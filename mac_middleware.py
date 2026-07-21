import asyncio
import os
import uuid
import uvicorn
import httpx
from contextlib import asynccontextmanager
from pydantic import BaseModel
from dataclasses import dataclass, field
from fastapi import FastAPI, Request, HTTPException

# ---------------------------------------------------------
# CONFIGURATION & STATE
# ---------------------------------------------------------
ENVIRONMENT = os.getenv("ENVIRONMENT", "LOCAL")
CLIENT_TIMEOUT_SECONDS = 30.0
MAX_CONCURRENT_REQUESTS = 5

request_queue = asyncio.PriorityQueue()
gpu_semaphore = None 

# Strict input validation prevents malformed payloads from crashing the parser
class GenerateRequest(BaseModel):
    prompt: str

@dataclass(order=True)
class RequestItem:
    priority: int
    req_id: str = field(compare=False)
    prompt: str = field(compare=False)
    future: asyncio.Future = field(compare=False)

# ---------------------------------------------------------
# PREDICTORS & CACHE LOGIC
# ---------------------------------------------------------
class Predictor:
    @staticmethod
    async def rank_request(prompt: str) -> int:
        """
        Dynamically routes ranking logic based on the environment.
        MUST be async to prevent blocking the event loop during HTTP calls.
        """
        input_tokens = len(prompt.split())
        
        if ENVIRONMENT == "PRODUCTION":
            # Real LTR scheduling using external API (e.g., Groq/Llama-3.1-8B)
            # async with httpx.AsyncClient() as client:
            #     response = await client.post(...)
            
            # Simulated Groq response delay and calculation
            await asyncio.sleep(0.1) 
            predicted_output = int(input_tokens * 2.5)
            return predicted_output
            
        else:
            # Local fast mock using coarse buckets (1/2/3)
            if input_tokens < 10: return 1
            elif input_tokens < 30: return 2
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

# ---------------------------------------------------------
# WORKERS & DISPATCHER
# ---------------------------------------------------------
async def mock_vllm_engine(prompt: str):
    generation_time = len(prompt.split()) * 0.1
    await asyncio.sleep(max(0.5, generation_time)) 
    return f"[SIMULATION SUCCESS | ENV: {ENVIRONMENT}] Generated output for prompt."

async def process_task(item: RequestItem):
    try:
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
    while True:
        item = await request_queue.get()
        
        if item.future.cancelled():
            request_queue.task_done()
            continue
        
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

# ---------------------------------------------------------
# API ENDPOINTS & LIFESPAN
# ---------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global gpu_semaphore
    gpu_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    asyncio.create_task(dynamic_dispatcher())
    print(f"🚀 {ENVIRONMENT} Gatekeeper Initialized. Awaiting traffic...")
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/generate")
async def generate(request: GenerateRequest):
    priority = await Predictor.rank_request(request.prompt)
    req_id = str(uuid.uuid4())
    
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    
    item = RequestItem(priority=priority, req_id=req_id, prompt=request.prompt, future=future)
    await request_queue.put(item)
    
    print(f"📥 [QUEUED] Request {req_id[:6]} added with Priority {priority}.")
    
    try:
        result = await asyncio.wait_for(future, timeout=CLIENT_TIMEOUT_SECONDS)
        return {"id": req_id, "priority_assigned": priority, "text": result}
        
    except asyncio.TimeoutError:
        future.cancel()
        print(f"❌ [TIMEOUT] Request {req_id[:6]} timed out in queue. Abandoning.")
        raise HTTPException(status_code=504, detail="Gateway Timeout: Request spent too long in queue.")
        
    except RuntimeError as e:
        if "429" in str(e):
            raise HTTPException(status_code=429, detail="Server overloaded. Request dropped per cost-analysis.")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)