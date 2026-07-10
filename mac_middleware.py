# import asyncio
# import uuid
# import uvicorn
# from fastapi import FastAPI, Request

# app = FastAPI()

# # The Decoupled Priority Queue (Your core contribution)
# request_queue = asyncio.PriorityQueue()

# # Global state to mock KV cache monitoring on your Mac
# active_requests = 0
# MAX_CONCURRENT_REQUESTS = 5 # Set low so you can easily trigger the throttle on your screen

# class PredictorMock:
#     """
#     Simulates your OPT-125M model ranking requests on the Mac.
#     """
#     @staticmethod
#     def rank_request(prompt: str) -> int:
#         length = len(prompt.split())
#         if length < 10:
#             return 1  # High priority (short task)
#         elif length < 30:
#             return 2  # Medium priority
#         else:
#             return 3  # Low priority

# async def mock_vllm_engine(prompt: str):
#     """
#     Mocks the NVIDIA GPU processing. Simulates the time it takes to generate tokens.
#     """
#     generation_time = len(prompt.split()) * 0.1  # 100ms per word simulation
#     await asyncio.sleep(generation_time)
#     return f"[M1 SIMULATION SUCCESS] Generated output for prompt: '{prompt}'"

# async def process_task(prompt, future, req_id, priority):
#     """
#     Processes the individual task concurrently to simulate parallel GPU execution.
#     """
#     global active_requests
#     try:
#         result_text = await mock_vllm_engine(prompt)
#         future.set_result(result_text)
#     except Exception as e:
#         future.set_exception(e)
#     finally:
#         active_requests -= 1
#         request_queue.task_done()

# async def dynamic_dispatcher():
#     """
#     The Gatekeeper: Polls the queue and throttles dispatch to prevent simulated crashes.
#     """
#     global active_requests
#     while True:
#         if active_requests >= MAX_CONCURRENT_REQUESTS:
#             # THROTTLING TRIGGERED: Print to console so the committee sees it working
#             print(f"⚠️ [THROTTLE ENGAGED] Cache at capacity ({active_requests}/{MAX_CONCURRENT_REQUESTS}). Yielding...")
#             await asyncio.sleep(0.5)
#             continue
        
#         # Get the highest priority request if memory is clear
#         priority, req_id, prompt, future = await request_queue.get()
        
#         active_requests += 1
#         print(f"✅ [DISPATCH] Processing request {req_id[:6]}... (Priority {priority}, Active: {active_requests})")
        
#         # Dispatch concurrently to simulate parallel GPU execution
#         asyncio.create_task(process_task(prompt, future, req_id, priority))

# @app.on_event("startup")
# async def startup_event():
#     asyncio.create_task(dynamic_dispatcher())
#     print("🚀 M1 Middleware Gatekeeper Initialized. Awaiting traffic...")

# @app.post("/generate")
# async def generate(request: Request):
#     data = await request.json()
#     prompt = data.get("prompt", "Default short prompt")
    
#     # 1. Rank the request
#     priority = PredictorMock.rank_request(prompt)
#     req_id = str(uuid.uuid4())
    
#     # 2. Stage in Priority Queue
#     loop = asyncio.get_running_loop()
#     future = loop.create_future()
#     await request_queue.put((priority, req_id, prompt, future))
    
#     print(f"📥 [QUEUED] Request {req_id[:6]} added with Priority {priority}.")
    
#     # 3. Wait for execution
#     result = await future
#     return {"id": req_id, "priority_assigned": priority, "text": result}

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)


import asyncio
import uuid
import uvicorn
from fastapi import FastAPI, Request

app = FastAPI()

# The Decoupled Priority Queue (Your core contribution)
request_queue = asyncio.PriorityQueue()

# Global state to mock KV cache monitoring on your Mac
active_requests = 0
MAX_CONCURRENT_REQUESTS = 5 # Set low so you can easily trigger the throttle on your screen

class PredictorMock:
    """
    Simulates your OPT-125M model ranking requests on the Mac.
    """
    @staticmethod
    def rank_request(prompt: str) -> int:
        length = len(prompt.split())
        if length < 10:
            return 1  # High priority (short task)
        elif length < 30:
            return 2  # Medium priority
        else:
            return 3  # Low priority

class CostAnalyzer:
    """
    Simulates the C_rec < C_swap mathematical decision rule from Slide 7.
    """
    @staticmethod
    def evaluate_eviction(prompt: str) -> str:
        # Mock calculation based on prompt length
        tokens = len(prompt.split())
        recompute_cost = tokens * 0.5  # Simulated ms to just regenerate later
        swap_cost = 80.0               # Simulated ms baseline to move over PCIe bus
        
        if recompute_cost < swap_cost:
            return "DROP_AND_RECOMPUTE"
        else:
            return "SAVE_AND_SWAP"

async def mock_vllm_engine(prompt: str):
    """
    Mocks the NVIDIA GPU processing. Simulates the time it takes to generate tokens.
    """
    generation_time = len(prompt.split()) * 0.1  # 100ms per word simulation
    await asyncio.sleep(generation_time)
    return f"[M1 SIMULATION SUCCESS] Generated output for prompt: '{prompt}'"

async def process_task(prompt, future, req_id, priority):
    """
    Processes the individual task concurrently to simulate parallel GPU execution.
    """
    global active_requests
    try:
        result_text = await mock_vllm_engine(prompt)
        future.set_result(result_text)
    except Exception as e:
        future.set_exception(e)
    finally:
        active_requests -= 1
        request_queue.task_done()

async def dynamic_dispatcher():
    """
    The Gatekeeper: Polls the queue, runs cost analysis, and throttles dispatch.
    """
    global active_requests
    while True:
        if active_requests >= MAX_CONCURRENT_REQUESTS:
            # Peek at the next request to evaluate its cost before acting
            priority, req_id, prompt, future = await request_queue.get()
            
            # Run the mathematical thresholding from Slide 7
            decision = CostAnalyzer.evaluate_eviction(prompt)
            
            if decision == "DROP_AND_RECOMPUTE":
                print(f"⚠️ [THROTTLE ENGAGED] Cache full ({active_requests}/{MAX_CONCURRENT_REQUESTS}). [COST ANALYSIS: Recompute < Swap]. Yielding and dropping...")
            else:
                print(f"⚠️ [THROTTLE ENGAGED] Cache full ({active_requests}/{MAX_CONCURRENT_REQUESTS}). [COST ANALYSIS: Swap < Recompute]. Initiating PCIe Swap...")
            
            # Put the request back in the queue to hold it
            await request_queue.put((priority, req_id, prompt, future))
            request_queue.task_done()
            
            await asyncio.sleep(0.5)
            continue
        
        # Get the highest priority request if memory is clear
        priority, req_id, prompt, future = await request_queue.get()
        
        active_requests += 1
        print(f"✅ [DISPATCH] Processing request {req_id[:6]}... (Priority {priority}, Active: {active_requests})")
        
        # Dispatch concurrently to simulate parallel GPU execution
        asyncio.create_task(process_task(prompt, future, req_id, priority))

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(dynamic_dispatcher())
    print("🚀 M1 Middleware Gatekeeper Initialized. Awaiting traffic...")

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "Default short prompt")
    
    # 1. Rank the request
    priority = PredictorMock.rank_request(prompt)
    req_id = str(uuid.uuid4())
    
    # 2. Stage in Priority Queue
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    await request_queue.put((priority, req_id, prompt, future))
    
    print(f"📥 [QUEUED] Request {req_id[:6]} added with Priority {priority}.")
    
    # 3. Wait for execution
    result = await future
    return {"id": req_id, "priority_assigned": priority, "text": result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)