
# Import asyncio for asynchronous programming, task management, queues, and semaphores
import asyncio

# Import uuid to generate a unique ID for every incoming request
import uuid

# Import uvicorn to run the FastAPI application
import uvicorn

# Import os to access environment variables
import os

# Import dataclass utilities to create a structured request object
from dataclasses import dataclass, field

# Import FastAPI components for creating the API and handling HTTP requests/errors
from fastapi import FastAPI, Request, HTTPException

# Import asynccontextmanager to manage application startup and shutdown lifecycle
from contextlib import asynccontextmanager

# Import the asynchronous Groq API client
from groq import AsyncGroq

# Import load_dotenv to load environment variables from a .env file
from dotenv import load_dotenv


# Load environment variables from the .env file
load_dotenv()

# Retrieve the Groq API key from the environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Create an asynchronous Groq client using the API key
groq_client = AsyncGroq(api_key=GROQ_API_KEY)


# Define a data structure representing a request waiting to be processed
# order=True allows Python to compare requests based on priority
@dataclass(order=True)
class RequestItem:
    # Priority value used by the PriorityQueue
    # Lower numbers are processed before higher numbers
    priority: int

    # Unique ID assigned to the request
    # compare=False prevents this field from affecting priority ordering
    req_id: str = field(compare=False)

    # The user's input prompt
    # compare=False prevents the prompt from affecting priority ordering
    prompt: str = field(compare=False)

    # Future object used to send the processing result back to the waiting request
    # compare=False prevents the future from affecting priority ordering
    future: asyncio.Future = field(compare=False)


# Create a priority queue to store incoming requests
# Requests with lower priority values are processed first
request_queue = asyncio.PriorityQueue()

# Maximum number of requests that can be processed concurrently
MAX_CONCURRENT_REQUESTS = 5

# Semaphore used to limit the number of concurrent requests
# It is initialized during application startup
gpu_semaphore = None

# Maximum amount of time a client request can wait for a result
CLIENT_TIMEOUT_SECONDS = 30.0


# Mock predictor used to estimate the expected response length of a request
class PredictorMock:

    # Predict the approximate number of words/tokens expected in the response
    @staticmethod
    async def rank_request(prompt: str) -> int:
        try:
            # Ask the Groq model to predict the expected answer length
            response = await groq_client.chat.completions.create(
                # Specify the Groq model being used
                model="llama-3.1-8b-instant",

                # Send the prediction request to the model
                messages=[
                    {
                        "role": "user",
                        "content": f"Reply with only a single number. How many words will your answer be if I asked you this: {prompt}"
                    }
                ],

                # Limit the prediction response to a small number of tokens
                max_tokens=10
            )

            # Convert the model's response into an integer
            # This integer is used as the request priority
            predicted_length = int(response.choices[0].message.content.strip())

            # Return the predicted response length
            return predicted_length

        except:
            # If the API request fails, estimate the response length
            # based on the number of words in the input prompt
            input_tokens = len(prompt.split())

            # Estimate the output length as 2.5 times the input length
            return int(input_tokens * 2.5)


# Analyze whether a request should be dropped or held when resources are full
class CostAnalyzer:

    # Evaluate the cost of recomputing a request versus temporarily saving it
    @staticmethod
    def evaluate_eviction(prompt: str) -> str:

        # Count the number of words in the prompt
        tokens = len(prompt.split())

        # Estimate the cost of processing the request again from scratch
        recompute_cost = tokens * 2.0

        # Estimate the cost of moving/saving the request
        # 30.0 represents a fixed data transfer overhead
        # 0.2 represents the cost associated with each token
        swap_cost = 30.0 + (tokens * 0.2)

        # If recomputing is cheaper than swapping,
        # recommend dropping the request and recomputing later
        if recompute_cost < swap_cost:
            return "DROP_AND_RECOMPUTE"

        # Otherwise, recommend holding/saving the request
        else:
            return "SAVE_AND_SWAP"


# Simulate an LLM inference engine
async def mock_vllm_engine(prompt: str):

    # Send the user's prompt to the Groq API
    response = await groq_client.chat.completions.create(
        # Specify the Groq model
        model="llama-3.1-8b-instant",

        # Send the actual user prompt to the model
        messages=[{"role": "user", "content": prompt}],

        # Limit the generated response to 500 tokens
        max_tokens=500
    )

    # Return the generated text from the first response choice
    return response.choices[0].message.content


# Process a single request from the priority queue
async def process_task(item: RequestItem):
    try:
        # Check whether the client cancelled the request
        # If it was cancelled, stop processing it
        if item.future.cancelled():
            return

        # Send the prompt to the mock inference engine
        result_text = await mock_vllm_engine(item.prompt)

        # If the future is still active, send the generated result back to it
        if not item.future.done():
            item.future.set_result(result_text)

    # Handle any errors that occur while processing the request
    except Exception as e:

        # If the future is still active, send the exception to the waiting request
        if not item.future.done():
            item.future.set_exception(e)

    finally:
        # Release the semaphore slot after processing is complete
        gpu_semaphore.release()

        # Notify the priority queue that this task has finished
        request_queue.task_done()


# Continuously monitor the queue and dispatch requests for processing
async def dynamic_dispatcher():

    # Keep the dispatcher running continuously
    while True:

        # Wait for the next request from the priority queue
        item = await request_queue.get()

        # Check if the client cancelled the request while it was waiting
        if item.future.cancelled():

            # Mark the queue task as completed
            request_queue.task_done()

            # Skip this cancelled request
            continue

        # Check whether all available processing slots are currently occupied
        if gpu_semaphore.locked():

            # Analyze the cost of dropping versus holding the request
            decision = CostAnalyzer.evaluate_eviction(item.prompt)

            # If recomputing is considered cheaper, drop the request
            if decision == "DROP_AND_RECOMPUTE":

                # Log that the request is being dropped due to throttling
                print(f"⚠️ [THROTTLE] Cache full. Dropping request {item.req_id[:6]}")

                # Send an overload error to the waiting request
                if not item.future.done():
                    item.future.set_exception(RuntimeError("429_TOO_MANY_REQUESTS"))

                # Mark the queue task as completed
                request_queue.task_done()

                # Continue processing the next queued request
                continue

            # If saving/swapping is considered cheaper, hold the request
            else:
                # Log that the request is being held
                print(f"⏳ [THROTTLE] Cache full. Holding request {item.req_id[:6]}...")

                # Wait until a processing slot becomes available
                await gpu_semaphore.acquire()

        # If the semaphore is not locked, acquire an available processing slot
        else:
            await gpu_semaphore.acquire()

        # Log that the request is being dispatched for processing
        print(f"✅ [DISPATCH] Processing request {item.req_id[:6]}... (Priority {item.priority})")

        # Start processing the request asynchronously
        # This allows the dispatcher to continue handling other requests
        asyncio.create_task(process_task(item))


# Manage the FastAPI application's startup and shutdown lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):

    # Access the global semaphore variable
    global gpu_semaphore

    # Create a semaphore with a maximum of 5 concurrent requests
    gpu_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # Start the background request dispatcher
    asyncio.create_task(dynamic_dispatcher())

    # Print a startup message
    print("🚀 Linux Middleware Gatekeeper Initialized. Awaiting traffic...")

    # Allow the FastAPI application to start
    yield


# Create the FastAPI application and attach the lifecycle manager
app = FastAPI(lifespan=lifespan)


# Define an endpoint for checking the current middleware status
@app.get("/status")
async def get_status():

    # Return information about the middleware and current resource usage
    return {
        # Name of the running system
        "system": "LTR Middleware Running",

        # Information about the available GPU processing slots
        "gpu_slots": {

            # Total number of available concurrent request slots
            "total": MAX_CONCURRENT_REQUESTS,

            # Calculate how many slots are currently being used
            "in_use": MAX_CONCURRENT_REQUESTS - gpu_semaphore._value if gpu_semaphore else 0,

            # Show the number of currently available processing slots
            "available": gpu_semaphore._value if gpu_semaphore else MAX_CONCURRENT_REQUESTS
        },

        # Information about requests waiting in the queue
        "queue": {

            # Return the number of requests currently waiting
            "waiting": request_queue.qsize(),
        },

        # Provide information about where users can test the API
        "info": "Visit /docs to send prompts and test the middleware"
    }


# Define the endpoint used to submit prompts for generation
@app.post("/generate")
async def generate(request: Request):

    # Read the JSON request body
    data = await request.json()

    # Get the prompt from the request
    # Use a default prompt if no prompt is provided
    prompt = data.get("prompt", "Default short prompt")

    # Use the predictor to estimate the request's priority
    priority = await PredictorMock.rank_request(prompt)

    # Generate a unique ID for this request
    req_id = str(uuid.uuid4())

    # Get the current asyncio event loop
    loop = asyncio.get_running_loop()

    # Create a Future object that will eventually contain the request result
    future = loop.create_future()

    # Create a RequestItem containing all information needed to process the request
    item = RequestItem(
        priority=priority,
        req_id=req_id,
        prompt=prompt,
        future=future
    )

    # Add the request to the priority queue
    await request_queue.put(item)

    # Log that the request has been successfully queued
    print(f"📥 [QUEUED] Request {req_id[:6]} added with Priority {priority}.")

    try:
        # Wait for the request to finish processing
        # Cancel the wait if it takes longer than the configured timeout
        result = await asyncio.wait_for(
            future,
            timeout=CLIENT_TIMEOUT_SECONDS
        )

        # Return the request ID, assigned priority, and generated response
        return {
            "id": req_id,
            "priority_assigned": priority,
            "text": result
        }

    # Handle requests that exceed the allowed waiting time
    except asyncio.TimeoutError:

        # Cancel the future so the request is no longer expected by the client
        future.cancel()

        # Log that the request timed out
        print(f"❌ [TIMEOUT] Request {req_id[:6]} timed out in queue. Abandoning.")

        # Return an HTTP 504 Gateway Timeout response
        raise HTTPException(
            status_code=504,
            detail="Gateway Timeout: Request spent too long in queue."
        )

    # Handle requests that were explicitly dropped due to overload
    except RuntimeError as e:

        # Check whether the error represents a 429 overload condition
        if "429" in str(e):

            # Return an HTTP 429 Too Many Requests response
            raise HTTPException(
                status_code=429,
                detail="Server overloaded. Request dropped per cost-analysis."
            )

        # Return a generic HTTP 500 error for other runtime errors
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# Start the Uvicorn server when this file is run directly
if __name__ == "__main__":

    # Run the FastAPI application
    # Listen on all network interfaces using port 8000
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )

