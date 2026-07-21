import asyncio
import aiohttp
import time
import requests

BASE_URL = "http://127.0.0.1:8000"

TEST_PROMPTS = [
    ("short_1", "Hi"),
    ("short_2", "What is 2 + 2?"),
    ("short_3", "Name a color."),
    ("medium_1", "Explain what a neural network is."),
    ("medium_2", "What are the differences between Python and JavaScript?"),
    ("long_1", "Write a detailed essay on the history of artificial intelligence covering every decade from the 1950s to today."),
    ("long_2", "Explain in detail how transformers work in deep learning including self-attention and positional encoding."),
]

def check_status():
    try:
        response = requests.get(f"{BASE_URL}/status")
        data = response.json()
        print(f"\n📊 STATUS:")
        print(f"   GPU slots in use: {data['gpu_slots']['in_use']}/{data['gpu_slots']['total']}")
        print(f"   Requests waiting in queue: {data['queue']['waiting']}\n")
    except Exception as e:
        print(f"❌ Failed to get status: {e}")

async def send_prompt(session, label, prompt):
    print(f"📤 Sending [{label}]: '{prompt[:50]}...' " if len(prompt) > 50 else f"📤 Sending [{label}]: '{prompt}'")
    try:
        async with session.post(f"{BASE_URL}/generate", json={"prompt": prompt}, timeout=60) as response:
            if response.status == 200:
                data = await response.json()
                print(f"✅ [{label}] Priority={data['priority_assigned']} | Response: {data['text'][:80]}...")
            elif response.status == 429:
                print(f"⚠️  [{label}] DROPPED — server overloaded (cost analysis: recompute cheaper)")
            elif response.status == 504:
                print(f"❌ [{label}] TIMEOUT — spent too long in queue")
            else:
                print(f"❌ [{label}] Error: {response.status}")
    except Exception as e:
        print(f"❌ [{label}] Failed: {e}")

async def main():
    print("\n" + "="*55)
    print("  FDU Capstone — Concurrent LTR Middleware Test")
    print("="*55)

    print("\n[STEP 1] Checking system status...")
    check_status()

    print("[STEP 2] Blasting mixed prompts concurrently...\n")
    
    async with aiohttp.ClientSession() as session:
        tasks = [send_prompt(session, label, prompt) for label, prompt in TEST_PROMPTS]
        await asyncio.gather(*tasks)

    print("\n[STEP 3] Final status check...")
    check_status()
    print("="*55)
    print("  Test complete! Check the middleware terminal logs.")
    print("="*55)

if __name__ == "__main__":
    asyncio.run(main())