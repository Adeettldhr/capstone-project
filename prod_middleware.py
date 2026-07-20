import asyncio
import uuid
import uvicorn
import os
from dataclasses import dataclass, field
from fastapi import FastAPI, Request, HTTPException
from groq import Groq
from dotenv import load_dotenv

# Groq API client — replaces both OPT-125M predictor and vLLM engine
# Due to lack of access to A100 GPU hardware, we use Groq's free API
# which runs Llama-3.1-8B in the cloud as our inference backend.

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

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
        # Uses Groq API with Llama-3.1-8B to predict response length.
        # This replaces OPT-125M inference which requires A100 access.
        # Shorter predicted output = higher priority in the LTR queue.
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "user",
                        "content": f"Reply with only a single number. How many words will your answer be if I asked you this: {prompt}"
                    }
                ],
                max_tokens=10
            )
            predicted_length = int(response.choices[0].message.content.strip())
            return predicted_length
        except:
            # Fallback if Groq call fails for any reason
            input_tokens = len(prompt.split())
            return int(input_tokens * 2.5)

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
    # Real LLM inference via Groq API running Llama-3.1-8B.
    # This replaces the simulated asyncio.sleep() from before.
    # Due to lack of A100 access, Groq serves as our inference backend.
    # On A100 (future deployment): replace with vllm.AsyncLLMEngine.
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        None,
        lambda: groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
    )
    return response.choices[0].message.content
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

@app.get("/status")
async def get_status():
    # Live monitoring endpoint — shows current system health.
    # Useful for demoing the middleware and checking if it's working.
    return {
        "system": "LTR Middleware Running",
        "gpu_slots": {
            "total": MAX_CONCURRENT_REQUESTS,
            "in_use": MAX_CONCURRENT_REQUESTS - gpu_semaphore._value if gpu_semaphore else 0,
            "available": gpu_semaphore._value if gpu_semaphore else MAX_CONCURRENT_REQUESTS
        },
        "queue": {
            "waiting": request_queue.qsize(),
        },
        "info": "Visit /docs to send prompts and test the middleware"
    }


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