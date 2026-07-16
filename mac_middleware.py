import asyncio
import uuid
import uvicorn
from dataclasses import dataclass, field
from fastapi import FastAPI, Request, HTTPException

@dataclass(order=True)
class RequestItem:
    priority: int
    req_id: str = field(compare=False)
    prompt: str = field(compare=False)
    future: asyncio.Future = field(compare=False)

app = FastAPI()

request_queue = asyncio.PriorityQueue()

# Robust concurrency management using asyncio primitives
MAX_CONCURRENT_REQUESTS = 5
gpu_semaphore = None  # Must be initialized inside the async event loop

class PredictorMock:
    @staticmethod
    def rank_request(prompt: str) -> int:
        length = len(prompt.split())
        if length < 10: return 1
        elif length < 30: return 2
        else: return 3

class CostAnalyzer:
    @staticmethod
    def evaluate_eviction(prompt: str) -> str:
        tokens = len(prompt.split())
        recompute_cost = tokens * 0.5
        swap_cost = 80.0
        
        if recompute_cost < swap_cost:
            return "DROP_AND_RECOMPUTE"
        else:
            return "SAVE_AND_SWAP"

async def mock_vllm_engine(prompt: str):
    generation_time = len(prompt.split()) * 0.1
    # Enforce a minimum sleep so concurrency overlap is easily visible during testing
    await asyncio.sleep(max(0.5, generation_time)) 
    return f"[GPU SIMULATION SUCCESS] Generated output for prompt: '{prompt}'"

async def process_task(item: RequestItem):
    """Executes the task and safely releases the GPU slot when finished."""
    try:
        result_text = await mock_vllm_engine(item.prompt)
        if not item.future.done():
            item.future.set_result(result_text)
    except Exception as e:
        if not item.future.done():
            item.future.set_exception(e)
    finally:
        # Crucial: Release the slot so the dispatcher can send the next request
        gpu_semaphore.release()
        request_queue.task_done()

async def dynamic_dispatcher():
    """
    The Gatekeeper. 
    Replaces the global counter and sleep-loops with an asyncio.Semaphore.
    """
    while True:
        # 1. Grab the highest priority item from the queue
        item = await request_queue.get()
        
        # 2. Check if the GPU is at maximum capacity
        if gpu_semaphore.locked():
            decision = CostAnalyzer.evaluate_eviction(item.prompt)
            
            if decision == "DROP_AND_RECOMPUTE":
                print(f"⚠️ [THROTTLE] Cache full. [Recompute < Swap]. Dropping request {item.req_id[:6]}")
                
                # Fix: Resolve the future with an error so the client HTTP request ends
                if not item.future.done():
                    item.future.set_exception(RuntimeError("429_TOO_MANY_REQUESTS"))
                request_queue.task_done()
                continue
            else:
                print(f"⏳ [THROTTLE] Cache full. [Swap < Recompute]. Holding request {item.req_id[:6]}...")
                # Fix: Block cleanly until a slot opens. No busy-wait loops.
                await gpu_semaphore.acquire()
        else:
            # We have immediate capacity
            await gpu_semaphore.acquire()
        
        # 3. Dispatch the task concurrently
        print(f"✅ [DISPATCH] Processing request {item.req_id[:6]}... (Priority {item.priority})")
        asyncio.create_task(process_task(item))

@app.on_event("startup")
async def startup_event():
    global gpu_semaphore
    # Initialize the semaphore here so it attaches to the running async loop
    gpu_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    asyncio.create_task(dynamic_dispatcher())
    print("🚀 Linux Middleware Gatekeeper Initialized. Awaiting traffic...")

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "Default short prompt")
    
    priority = PredictorMock.rank_request(prompt)
    req_id = str(uuid.uuid4())
    
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    
    item = RequestItem(priority=priority, req_id=req_id, prompt=prompt, future=future)
    await request_queue.put(item)
    
    print(f"📥 [QUEUED] Request {req_id[:6]} added with Priority {priority}.")
    
    try:
        # Wait for the dispatcher/worker to fulfill the future
        result = await future
        return {"id": req_id, "priority_assigned": priority, "text": result}
    except RuntimeError as e:
        # Catch the exception set by the dispatcher and translate it to an HTTP error
        if "429" in str(e):
            raise HTTPException(status_code=429, detail="Server overloaded. Request dropped per cost-analysis.")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)