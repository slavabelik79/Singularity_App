# Singularity Forecaster v2.1 (Monte Carlo Edition)

An advanced analytical tool and simulation engine designed to forecast the arrival of Artificial General Intelligence (AGI) and Superintelligence (ASI). Unlike simple exponential models, this script uses a multi-factor approach to simulate the complex dynamics of AI progress.

## 🚀 Overview

The model simulates thousands of possible future trajectories (Monte Carlo method) starting from a 2026 baseline. It accounts for hardware growth, algorithmic efficiency, and "recursive self-improvement," while simultaneously factoring in real-world bottlenecks like the "Data Wall," energy constraints, and regulatory damping.

## 🧠 Methodology & Key Features

This script implements several advanced forecasting methods:

### 1. Logistic Scaling Laws (Paradigm-Based)
Traditional models assume infinite power-law scaling. Version 2.1 introduces Logistic Saturation:
* Paradigm Ceilings: Each AI paradigm (e.g., Transformers, Diffusion) has an inherent capability ceiling.
* Paradigm Shifts: The model simulates random "breakthrough events" that shift the ceiling upward, allowing for sudden leaps in progress.

### 2. Continuous Inference Scaling
Inspired by recent breakthroughs, the model treats Inference-Time Compute as a continuous multiplier. As models gain capability, the effectiveness of additional "thinking time" increases, creating a non-linear boost to problem-solving abilities.

### 3. Technical & Physical Barriers
* Data Wall: Starting in 2026, the scarcity of high-quality human-generated data begins to slow down both hardware utilization and algorithmic breakthroughs.
* Energy Wall: Post-2027, the model introduces a damping factor representing the difficulty of scaling data centers due to power grid limitations.

### 4. Non-Technical "Soft" Barriers
The simulation includes "Human-in-the-loop" constraints:
* Regulatory Damping: Simulates the slowing effect of laws (e.g., EU AI Act) and export controls.
* Alignment Pauses: After reaching AGI-level capability, there is a probabilistic chance of a "safety moratorium" — a pause in deployment to ensure the system is safe.

### 5. Recursive Self-Improvement (RSI)
Once AI reaches a "researcher" level (Capability Score > 2.0), it begins to contribute to its own algorithmic development. This creates a feedback loop that can lead to a "hard takeoff" toward ASI.

### 6. Monte Carlo Simulation
The script runs 3,000+ independent trajectories with randomized variables (doubling times, breakthrough probabilities, damping rates) to produce a probabilistic distribution:
* 10th Percentile: Aggressive/Optimistic scenario.
* 50th Percentile: Median/Most Likely scenario.
* 90th Percentile: Conservative/Delayed scenario.

## 🛠 Operational Definitions

* Baseline (1.0): Frontier models of early 2026 (estimated ~10^27.5 FLOPs training compute).
* AGI (10.0): Reliable performance on ARC-AGI-2, autonomous PhD-level scientific research.
* ASI (1000.0): Intelligence surpassing total human collective capability across all domains.

## 📊 How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### CLI Version
```bash
python singularity_v2_1.py
```

### Web Dashboard (Flask)
A 4‑chart interactive dashboard is included:

1. Start the server:
 ```bash
 python app.py
 ```
2. Open http://localhost:5000 in your browser.

The dashboard provides:
* **Histogram** – AGI/ASI distribution across Monte Carlo runs.
* **Capability trajectory** – median growth with 10–90% confidence bands.
* **Cumulative probability** – P(AGI ≤ X years) and P(ASI ≤ X years).
* **Sensitivity analysis** – impact of each parameter (hardware, algorithms, paradigm ceiling, data wall).

## ⚠️ Disclaimer
This is a mathematical model for educational and forecasting purposes. It does not account for black-swan geopolitical events (e.g., global conflicts) or sudden extreme changes in hardware manufacturing capabilities.

---
Author: slavabelik79  
Model Version: 2.1 (Corrected Physics & Non-Technical Barriers)