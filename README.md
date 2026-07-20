# Capstone Defense: Decoupled Gatekeeper Middleware

## Overview
This repository contains the asynchronous middleware component (`mac_middleware.py`) for our capstone project: **Overcoming Overfitting in LTR-Based Scheduling for Large Language Models**. 

Standard LLM serving engines (e.g., vLLM) utilizing First-Come-First-Serve (FCFS) scheduling suffer from Head-of-Line (HOL) blocking and fatal memory allocation faults (OOM crashes) during peak traffic surges. This middleware acts as a **Decoupled Gatekeeper**, physically isolating the HTTP traffic layer from the fragile C++ VRAM memory manager. 

### Core Architecture
*   **Predictive Ranking:** Integrates an OPT-125M predictor (trained on LMSYS-Chat-1M with L2 regularization) to evaluate generative complexity and rank incoming requests in an `asyncio.PriorityQueue`.
*   **Dynamic Thresholding:** Calculates real-time Key-Value (KV) cache limits to evaluate the trade-off between recomputation cost ($C_{Rec}$) and PCIe swap latency ($C_{Swap}$).
*   **Active Admission Control:** Drops computationally cheap requests during extreme concurrency spikes (returning HTTP 429) or yields them cleanly to prevent hardware saturation and segmentation faults.

## Prerequisites
- Python 3.10+
- Git
- API Framework: FastAPI & Uvicorn

## Local Setup Instructions

**1. Clone the repository**
```bash
git clone [https://github.com/Adeettldhr/capstone-project.git](https://github.com/Adeettldhr/capstone-project.git)
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

## Usage

**1. Start the Middleware Server**
Ensure your virtual environment is active, then initialize the FastAPI application and background dispatcher:
```bash
python mac_middleware.py
```
*(The server will boot on `http://0.0.0.0:8000` and initialize the asynchronous gatekeeper.)*
