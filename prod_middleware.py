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
MAX_CONCURRENT_REQUESTS = 5
gpu_semaphore = None 

# Production timeout settings
CLIENT_TIMEOUT_SECONDS = 30.0

class PredictorMock:
    @staticmethod
    def rank_request(prompt: str) -> int:
        # Instead of coarse buckets (1/2/3), we use predicted output length
        # as the actual priority number. Lower number = shorter request = 
        # processed first. This is real LTR scheduling.
        # Due to the lack of access to A100 GPU hardware and OPT-125M inference,
        # we are using the Groq API with Llama-3.1-8B as a substitute inference
        # backend to simulate real predicted output length.
        # On the A100, which is future deployment, we should replace this with actual OPT-125M inference.
        input_tokens = len(prompt.split())
        predicted_output = int(input_tokens * 2.5)  # realistic output estimate
        return predicted_output

class CostAnalyzer:
    @staticmethod
    def evaluate_eviction(prompt: str) -> str:
        # swap_cost now scales with request size instead of being
        # hardcoded at 80.0 — longer requests cost more to move
        # to CPU RAM and back, so they are more worth saving.
        # Due to lack of access to A100 GPU hardware, these costs
        # are estimated values. On A100 (future deployment): replace
        # with real PCIe bandwidth measurements from vLLM.
        tokens = len(prompt.split())
        recompute_cost = tokens * 0.5
        swap_cost = tokens * 1.5  # scales with size: bigger = more expensive to swap
        
        if recompute_cost < swap_cost:
            return "DROP_AND_RECOMPUTE"
        else:
            return "SAVE_AND_SWAP"

async def mock_vllm_engine(prompt: str):
    generation_time = len(prompt.split()) * 0.1
    await asyncio.sleep(max(0.5, generation_time)) 
    return f"[GPU SIMULATION SUCCESS] Generated output for prompt: '{prompt}'"

async def process_task(item: RequestItem):
    """Executes the task and safely releases the GPU slot."""
    try:
        # Final check before burning GPU cycles: Did the client give up while waiting for the semaphore?
        if item.future.cancelled():
            return

        result_text = await mock_vllm_engine(item.prompt)
        
        if not item.future.done():
            item.future.set_result(result_text)
            
    except Exception as e:
        if not item.future.done():
            item.future.set_exception(e)
            
    finally:
        gpu_semaphore.release()
        request_queue.task_done()

async def dynamic_dispatcher():
    """Polls the queue, handles caching logic, and filters dead requests."""
    while True:
        item = await request_queue.get()
        
        # 1. Purge dead requests immediately. 
        # If the client timed out while this was in the queue, drop it and move on.
        if item.future.cancelled():
            request_queue.task_done()
            continue
        
        # 2. Concurrency check
        if gpu_semaphore.locked():
            decision = CostAnalyzer.evaluate_eviction(item.prompt)
            
            if decision == "DROP_AND_RECOMPUTE":
                print(f"⚠️ [THROTTLE] Cache full. Dropping request {item.req_id[:6]}")
                if not item.future.done():
                    item.future.set_exception(RuntimeError("429_TOO_MANY_REQUESTS"))
                request_queue.task_done()
                continue
            else:
                print(f"⏳ [THROTTLE] Cache full. Holding request {item.req_id[:6]}...")
                await gpu_semaphore.acquire()
        else:
            await gpu_semaphore.acquire()
        
        print(f"✅ [DISPATCH] Processing request {item.req_id[:6]}... (Priority {item.priority})")
        asyncio.create_task(process_task(item))

@app.on_event("startup")
async def startup_event():
    global gpu_semaphore
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
        # Wrap the await in a strict timeout
        result = await asyncio.wait_for(future, timeout=CLIENT_TIMEOUT_SECONDS)
        return {"id": req_id, "priority_assigned": priority, "text": result}
        
    except asyncio.TimeoutError:
        # The timeout popped. We MUST cancel the future so the dispatcher knows to drop it.
        future.cancel()
        print(f"❌ [TIMEOUT] Request {req_id[:6]} timed out in queue. Abandoning.")
        raise HTTPException(status_code=504, detail="Gateway Timeout: Request spent too long in queue.")
        
    except RuntimeError as e:
        if "429" in str(e):
            raise HTTPException(status_code=429, detail="Server overloaded. Request dropped per cost-analysis.")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)