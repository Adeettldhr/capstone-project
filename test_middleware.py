"""
Test Script for LTR Middleware
Sends a mix of short and long prompts to prove:
1. LTR scheduling - short requests get higher priority
2. Groq API is connected — real responses come back
3. Dynamic thresholding implementation — DROP/SWAP decisions are logged
(We have to run this in a separate terminal while the prod_middleware is running:)
    uvicorn prod_middleware:app --reload
"""

import requests
import time

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
    response = requests.get(f"{BASE_URL}/status")
    data = response.json()
    print(f"\n📊 STATUS:")
    print(f"   GPU slots in use: {data['gpu_slots']['in_use']}/{data['gpu_slots']['total']}")
    print(f"   Requests waiting in queue: {data['queue']['waiting']}\n")

def send_prompt(label, prompt):
    print(f"📤 Sending [{label}]: '{prompt[:50]}...' " if len(prompt) > 50 else f"📤 Sending [{label}]: '{prompt}'")
    try:
        response = requests.post(
            f"{BASE_URL}/generate",
            json={"prompt": prompt},
            timeout=60
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✅ [{label}] Priority={data['priority_assigned']} | Response: {data['text'][:80]}...")
        elif response.status_code == 429:
            print(f"⚠️  [{label}] DROPPED — server overloaded (cost analysis: recompute cheaper)")
        elif response.status_code == 504:
            print(f"❌ [{label}] TIMEOUT — spent too long in queue")
        else:
            print(f"❌ [{label}] Error: {response.status_code}")
    except Exception as e:
        print(f"❌ [{label}] Failed: {e}")
        print("   → Make sure middleware is running: uvicorn prod_middleware:app --reload")

print("\n" + "="*55)
print("  FDU Capstone — LTR Middleware Test")
print("="*55)

print("\n[STEP 1] Checking system status...")
check_status()

print("[STEP 2] Sending mixed prompts (short + long)...\n")
for label, prompt in TEST_PROMPTS:
    send_prompt(label, prompt)
    time.sleep(1)

print("\n[STEP 3] Final status check...")
check_status()

print("="*55)
print("  Test complete!")
print("  Check the middleware terminal for DROP/SWAP/DONE logs")
print("="*55)