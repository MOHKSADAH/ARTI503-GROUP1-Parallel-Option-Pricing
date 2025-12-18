# Accelerating Financial Option Pricing Using Parallel Monte Carlo Methods

**Course:** ARTI503 Parallel Architecture and Programming  
**Term:** Term 1 - 2025-2026  
**Stage:** 3 - Implementation and Parallelization  
**Institution:** Imam Abdulrahman Bin Faisal University  

## 📄 Project Overview
This repository contains the source code for our research on accelerating European option pricing using Monte Carlo simulations. The project compares a baseline **Sequential** implementation against a **Parallel** implementation using Python's `multiprocessing` library.

The simulation models European Call and Put options using Geometric Brownian Motion (GBM). Our parallel approach utilizes a **reduction pattern** to eliminate race conditions and achieve high scalability without synchronization overhead.

## 👥 Group Members
* Mohammad Khalil Al-Sadah
* Saad Ali Al-Ghamdi
* Abdulrahman Sultan Alotaibi
* Mohammed Jasim Almarzooq
* Shaya Saeed Alrashdi
* Khalid Mohammed Alhunief
* Khalifah Zakaria Alkhalifah
* Amjad Abdullah Al-Laif
* Mohammed Abdullah Alahmari

**Advisor:** Yasir Alguwaifli

## 📂 Repository Structure
The project is organized into two main modules:

```text
/
├── sequential/
│   └── monte_carlo_sequential.py   # Baseline serial implementation
├── parallel/
│   └── monte_carlo_parallel.py     # Optimized parallel implementation with multiprocessing
└── README.md
```

## 🛠️ Environment & Requirements
The code was developed and benchmarked using the following specifications:

* **Hardware:** AMD Ryzen 7 9700X (8 Cores, 16 Threads), 32GB DDR5 RAM
* **OS:** Windows 11
* **Language:** Python 3.13.5
* **Libraries:** `numpy` (Version 2.3.4), `scipy`

### Installation
To install the required dependencies:
```bash
pip install numpy scipy
```

## 🚀 How to Run

### 1. Sequential Benchmark
To run the baseline Monte Carlo simulation and generate Stage 1 benchmark data:
```bash
python sequential/monte_carlo_sequential.py
```
*Output: Calculates prices for 10k to 1M paths and saves `benchmark_results.json`.*

### 2. Parallel Benchmark
To run the parallel implementation (Stage 3) which tests scaling across 1, 2, 4, 6, and 8 cores:
```bash
python parallel/monte_carlo_parallel.py
```
*Output: Generates speedup/efficiency metrics and saves `parallel_benchmark_results.json`.*

## 📊 Key Performance Results
Our implementation demonstrated significant performance gains on the AMD Ryzen 9700X:

* **Speedup:** Achieved **5.18x** speedup on 8 cores for 1,000,000 simulation paths.
* **Efficiency:** Maintained **88.1%** parallel efficiency on 4 cores.
* **Scalability:** Strong scaling observed up to 6 cores, with memory bandwidth becoming a bottleneck at 8 cores.
* **Accuracy:** Monte Carlo estimates converged to the analytical Black-Scholes value with <0.15% error.

## 📝 Methodology
* **Parallel Framework:** Python `multiprocessing.Pool`.
* **Technique:** Map-Reduce pattern where each worker calculates payoffs independently.
* **RNG Handling:** Independent Random Number Generator streams (unique seeds per path) to ensure statistical validity and thread safety.

## 📄 License
This project is submitted for the ARTI503 course at IAU.
