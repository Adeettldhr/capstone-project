import asyncio
import uuid
import uvicorn
from dataclasses import dataclass, field
from fastapi import FastAPI, Request, HTTPException

# We use a dataclass to package the request data together.
# order=True allows the PriorityQueue to sort these objects automatically.
@dataclass(order=True)
class RequestItem:
    # The queue sorts entirely based on this integer. Lower number = higher priority.
    priority: int
    
    # compare=False tells the queue to ignore these fields during sorting.
    # This is mandatory because asyncio.Future objects cannot be compared mathematically.
    # Without this, if two items have the exact same priority, the queue will crash 
    # trying to compare their req_id or future.
    req_id: str = field(compare=False)
    prompt: str = field(compare=False)
    future: asyncio.Future = field(compare=False)

app = FastAPI()

# A thread-safe queue that automatically orders items by their priority number.
request_queue = asyncio.PriorityQueue()

# The maximum number of tasks the GPU can handle at exactly the same time.
MAX_CONCURRENT_REQUESTS = 5

# A Semaphore is like a bouncer at a club with a strict capacity.
# It holds a set number of "tickets" (MAX_CONCURRENT_REQUESTS). 
# A task must take a ticket to run, and return it when done.
gpu_semaphore = None 

class PredictorMock:
    @staticmethod
    def rank_request(prompt: str) -> int:
        # Simple logic to assign priority: shorter prompts get processed first (priority 1).
        length = len(prompt.split())
        if length < 10: return 1
        elif length < 30: return 2
        else: return 3

class CostAnalyzer:
    @staticmethod
    def evaluate_eviction(prompt: str) -> str:
        # Simulates the math deciding if it's faster to drop a request and make the user 
        # ask again later, or to pause it and move it to system RAM (PCIe Swap).
        tokens = len(prompt.split())
        recompute_cost = tokens * 0.5
        swap_cost = 80.0
        
        if recompute_cost < swap_cost:
            return "DROP_AND_RECOMPUTE"
        else:
            return "SAVE_AND_SWAP"

async def mock_vllm_engine(prompt: str):
    # Simulates the actual text generation on the GPU.
    generation_time = len(prompt.split()) * 0.1
    await asyncio.sleep(max(0.5, generation_time)) 
    return f"[GPU SIMULATION SUCCESS] Generated output for prompt: '{prompt}'"

async def process_task(item: RequestItem):
    """Handles the actual execution of a single request."""
    try:
        # Wait for the AI model to finish generating text
        result_text = await mock_vllm_engine(item.prompt)
        
        # If the user hasn't disconnected, send the generated text back to them
        if not item.future.done():
            item.future.set_result(result_text)
            
    except Exception as e:
        # If the AI model crashes, pass the error back to the user so they know it failed
        if not item.future.done():
            item.future.set_exception(e)
            
    finally:
        # CRITICAL: This ensures that no matter what happens (success or crash),
        # the "ticket" is returned to the bouncer so the next request in line can start.
        gpu_semaphore.release()
        
        # Tell the queue this specific item is fully completed
        request_queue.task_done()

async def dynamic_dispatcher():
    """
    The Gatekeeper. This runs in an infinite background loop.
    It constantly watches the queue and controls traffic to the GPU.
    """
    while True:
        # 1. Pull the highest priority request out of the waiting line.
        item = await request_queue.get()
        
        # 2. Check if the GPU is completely full right now.
        if gpu_semaphore.locked():
            
            # The GPU is full. We have to decide what to do with this request.
            decision = CostAnalyzer.evaluate_eviction(item.prompt)
            
            if decision == "DROP_AND_RECOMPUTE":
                print(f"⚠️ [THROTTLE] Cache full. [Recompute < Swap]. Dropping request {item.req_id[:6]}")
                
                # We decided it's too expensive to hold this in memory. 
                # We trigger an error inside the user's connection so they get a "Too Many Requests" response.
                if not item.future.done():
                    item.future.set_exception(RuntimeError("429_TOO_MANY_REQUESTS"))
                
                # Mark it done in the queue and skip to the next iteration of the loop
                request_queue.task_done()
                continue
            else:
                print(f"⏳ [THROTTLE] Cache full. [Swap < Recompute]. Holding request {item.req_id[:6]}...")
                
                # We decided to keep it. The dispatcher pauses exactly here and waits 
                # until a task finishes and returns a ticket to the semaphore.
                await gpu_semaphore.acquire()
        else:
            # The GPU is NOT full. Take a ticket immediately.
            await gpu_semaphore.acquire()
        
        # 3. We have a ticket. Send the request to the GPU to be processed in the background.
        print(f"✅ [DISPATCH] Processing request {item.req_id[:6]}... (Priority {item.priority})")
        
        # create_task runs process_task concurrently, allowing this loop to immediately 
        # move on to grab the next request in the queue.
        asyncio.create_task(process_task(item))

@app.on_event("startup")
async def startup_event():
    global gpu_semaphore
    # We must create the Semaphore here so it hooks into FastAPI's async event loop
    gpu_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    # Start the background traffic controller
    asyncio.create_task(dynamic_dispatcher())
    print("🚀 Linux Middleware Gatekeeper Initialized. Awaiting traffic...")

@app.post("/generate")
async def generate(request: Request):
    # 1. Parse the incoming HTTP request
    data = await request.json()
    prompt = data.get("prompt", "Default short prompt")
    
    # 2. Determine how important this request is
    priority = PredictorMock.rank_request(prompt)
    req_id = str(uuid.uuid4())
    
    # 3. Create a Future. This is an empty box. We will give this box to the queue,
    # and we will pause this HTTP endpoint until the dispatcher puts the final answer inside the box.
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    
    # 4. Package everything up and put it in the priority queue
    item = RequestItem(priority=priority, req_id=req_id, prompt=prompt, future=future)
    await request_queue.put(item)
    
    print(f"📥 [QUEUED] Request {req_id[:6]} added with Priority {priority}.")
    
    try:
        # 5. Pause and wait for the empty box (future) to be filled with the result
        result = await future
        return {"id": req_id, "priority_assigned": priority, "text": result}
        
    except RuntimeError as e:
        # 6. If the dispatcher put a "429_TOO_MANY_REQUESTS" error in the box instead of a result,
        # catch it and translate it into a proper HTTP status code for the user.
        if "429" in str(e):
            raise HTTPException(status_code=429, detail="Server overloaded. Request dropped per cost-analysis.")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)