# Capstone Defense: Decoupled Gatekeeper Middleware

## Overview
This repository contains the asynchronous middleware component (`prod_middleware.py`) for our capstone project: **Overcoming Overfitting in LTR-Based Scheduling for Large Language Models**.

Standard LLM serving engines (e.g., vLLM) utilizing First-Come-First-Serve (FCFS) scheduling suffer from Head-of-Line (HOL) blocking and fatal memory allocation faults (OOM crashes) during peak traffic surges. This middleware acts as a **Decoupled Gatekeeper**, physically isolating the HTTP traffic layer from the fragile C++ VRAM memory manager.

### Core Architecture
*   **Predictive Ranking:** Due to lack of access to A100 GPU hardware and OPT-125M inference, we substitute OPT-125M with the Groq API running Llama-3.1-8B to evaluate generative complexity and rank incoming requests in an `asyncio.PriorityQueue`. On A100 (future deployment): replace with actual OPT-125M inference trained on LMSYS-Chat-1M with L2 regularization.
*   **Dynamic Thresholding:** Calculates real-time Key-Value (KV) cache limits to evaluate the trade-off between recomputation cost ($C_{Rec}$) and PCIe swap latency ($C_{Swap}$). Swap cost now scales dynamically with request size instead of a hardcoded value.
*   **Active Admission Control:** Drops computationally cheap requests during extreme concurrency spikes (returning HTTP 429) or yields them cleanly to prevent hardware saturation and segmentation faults.

## Prerequisites
- Python 3.10+
- Git
- API Framework: FastAPI & Uvicorn
- Groq API Key (free at console.groq.com)

## Local Setup Instructions

**1. Clone the repository**
```bash
git clone https://github.com/Adeettldhr/capstone-project.git
cd capstone-project
```

**2. Create and activate a virtual environment**
```bash
python3 -m venv env
source env/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up your Groq API Key**

Create a `.env` file in the project root:
```bash
touch .env
```
Add your Groq API key inside it:
GROQ_API_KEY=your_groq_api_key_here
Note: The `.env` file is listed in `.gitignore` and will never be pushed to GitHub.

**5. Start the Middleware Server**
```bash
uvicorn prod_middleware:app --reload
```
*(The server will boot on `http://0.0.0.0:8000` and initialize the asynchronous gatekeeper.)*

**6. Run the Test Script**

In a separate terminal window:
```bash
python test_middleware.py
```
This sends a mix of short and long prompts through the middleware to verify LTR scheduling, Groq API connection, and dynamic thresholding are all working correctly.

**7. Monitor the System Live**

Visit in your browser:
http://127.0.0.1:8000/status
http://127.0.0.1:8000/docs
- `/status` shows GPU slots in use and requests waiting in queue
- `/docs` gives an interactive API explorer to send prompts manually

## Authors
- Adeet Tuladhar
- V. Dheeraj Kalapati

Fairleigh Dickinson University, Vancouver
