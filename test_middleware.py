
# Import asyncio for running asynchronous tasks and concurrent requests
import asyncio

# Import aiohttp for making asynchronous HTTP requests
import aiohttp

# Import time for time-related operations
# Note: It is currently imported but not used in this script
import time

# Import requests for making synchronous HTTP requests
import requests


# Base URL of the locally running FastAPI middleware server
BASE_URL = "http://127.0.0.1:8000"


# Define a collection of test prompts with labels
# These prompts are divided into short, medium, and long examples
TEST_PROMPTS = [
    # Short prompts
    ("short_1", "Hi"),
    ("short_2", "What is 2 + 2?"),
    ("short_3", "Name a color."),

    # Medium-length prompts
    ("medium_1", "Explain what a neural network is."),
    ("medium_2", "What are the differences between Python and JavaScript?"),

    # Long prompts
    ("long_1", "Write a detailed essay on the history of artificial intelligence covering every decade from the 1950s to today."),
    ("long_2", "Explain in detail how transformers work in deep learning including self-attention and positional encoding."),
]


# Function used to check the current status of the middleware server
def check_status():
    try:
        # Send a GET request to the /status endpoint
        response = requests.get(f"{BASE_URL}/status")

        # Convert the JSON response into a Python dictionary
        data = response.json()

        # Print a formatted status heading
        print(f"\n📊 STATUS:")

        # Display the number of currently occupied GPU/request slots
        # along with the total number of available slots
        print(f"   GPU slots in use: {data['gpu_slots']['in_use']}/{data['gpu_slots']['total']}")

        # Display the number of requests currently waiting in the queue
        print(f"   Requests waiting in queue: {data['queue']['waiting']}\n")

    # Handle any error that occurs while checking the server status
    except Exception as e:

        # Print the error message
        print(f"❌ Failed to get status: {e}")


# Asynchronous function used to send one prompt to the middleware
async def send_prompt(session, label, prompt):

    # Print the prompt being sent
    # If the prompt is longer than 50 characters, only show the first 50 characters
    print(
        f"📤 Sending [{label}]: '{prompt[:50]}...' "
        if len(prompt) > 50
        else f"📤 Sending [{label}]: '{prompt}'"
    )

    try:
        # Send an asynchronous POST request to the /generate endpoint
        # The prompt is sent as JSON data
        # The request has a 60-second client-side timeout
        async with session.post(
            f"{BASE_URL}/generate",
            json={"prompt": prompt},
            timeout=60
        ) as response:

            # Check if the server successfully processed the request
            if response.status == 200:

                # Convert the JSON response into a Python dictionary
                data = await response.json()

                # Print the assigned priority and a shortened version of the response
                print(
                    f"✅ [{label}] Priority={data['priority_assigned']} | "
                    f"Response: {data['text'][:80]}..."
                )

            # Handle requests that were dropped because the server is overloaded
            elif response.status == 429:

                # Inform the user that the request was dropped
                # because the cost analysis determined recomputing was cheaper
                print(
                    f"⚠️  [{label}] DROPPED — server overloaded "
                    f"(cost analysis: recompute cheaper)"
                )

            # Handle requests that spent too much time waiting in the queue
            elif response.status == 504:

                # Inform the user that the request timed out
                print(
                    f"❌ [{label}] TIMEOUT — spent too long in queue"
                )

            # Handle any other unexpected HTTP response status
            else:

                # Print the HTTP status code returned by the server
                print(
                    f"❌ [{label}] Error: {response.status}"
                )

    # Handle errors that occur while sending or processing the request
    except Exception as e:

        # Print the error associated with the specific request
        print(
            f"❌ [{label}] Failed: {e}"
        )


# Main asynchronous function that runs the complete test
async def main():

    # Print a visual separator at the beginning of the test
    print("\n" + "="*55)

    # Print the test name
    print("  FDU Capstone — Concurrent LTR Middleware Test")

    # Print another visual separator
    print("="*55)


    # STEP 1: Check the middleware status before sending requests
    print("\n[STEP 1] Checking system status...")

    # Call the function that retrieves and displays the current server status
    check_status()


    # STEP 2: Send all test prompts concurrently
    print("[STEP 2] Blasting mixed prompts concurrently...\n")
    

    # Create an asynchronous HTTP client session
    # The same session is reused for all requests
    async with aiohttp.ClientSession() as session:

        # Create one asynchronous task for each test prompt
        # Each task calls send_prompt() with its corresponding label and prompt
        tasks = [
            send_prompt(session, label, prompt)
            for label, prompt in TEST_PROMPTS
        ]

        # Run all request tasks concurrently
        # asyncio.gather() waits until all requests have completed
        await asyncio.gather(*tasks)


    # STEP 3: Check the middleware status after all requests finish
    print("\n[STEP 3] Final status check...")

    # Display the final server and queue status
    check_status()


    # Print the closing separator
    print("="*55)

    # Indicate that the test has completed
    print("  Test complete! Check the middleware terminal logs.")

    # Print the final separator
    print("="*55)


# Run the main asynchronous function only when this file is executed directly
if __name__ == "__main__":

    # Start the asyncio event loop and execute the main test function
    asyncio.run(main())

