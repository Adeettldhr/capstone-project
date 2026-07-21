import asyncio
import uuid
import uvicorn
import os
from dataclasses import dataclass, field
from fastapi import FastAPI, Request, HTTPException
from contextlib import asynccontextmanager
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = AsyncGroq(api_key=GROQ_API_KEY)

@dataclass(order=True)
class RequestItem:
    priority: int
    req_id: str = field(compare=False)
    prompt: str = field(compare=False)
    future: asyncio.Future = field(compare=False)

request_queue = asyncio.PriorityQueue()
MAX_CONCURRENT_REQUESTS = 5
gpu_semaphore = None 
CLIENT_TIMEOUT_SECONDS = 30.0

class PredictorMock:
    @staticmethod
    async def rank_request(prompt: str) -> int:
        try:
            response = await groq_client.chat.completions.create(
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
            input_tokens = len(prompt.split())
            return int(input_tokens * 2.5)

class CostAnalyzer:
    @staticmethod
    def evaluate_eviction(prompt: str) -> str:
        tokens = len(prompt.split())
        recompute_cost = tokens * 0.5
        swap_cost = tokens * 1.5 
        
        if recompute_cost < swap_cost:
            return "DROP_AND_RECOMPUTE"
        else:
            return "SAVE_AND_SWAP"

async def mock_vllm_engine(prompt: str):
    response = await groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    return response.choices[0].message.content

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    global gpu_semaphore
    gpu_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    asyncio.create_task(dynamic_dispatcher())
    print("🚀 Linux Middleware Gatekeeper Initialized. Awaiting traffic...")
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/status")
async def get_status():
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
    
    priority = await PredictorMock.rank_request(prompt)
    req_id = str(uuid.uuid4())
    
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    
    item = RequestItem(priority=priority, req_id=req_id, prompt=prompt, future=future)
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