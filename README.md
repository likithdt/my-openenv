---
title: Data Integrity Lab
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
---

# 📊 Data Integrity Lab (OpenEnv)
### **Autonomous Reinforcement Learning Environment for Data Auditing**

[![OpenEnv Spec](https://img.shields.io/badge/OpenEnv-v2.0-blue)](https://github.com/openenv/spec)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: FastAPI](https://img.shields.io/badge/Framework-FastAPI-009688)](https://fastapi.tiangolo.com/)

## 🎯 Motivation
Data cleaning is a manual, $100B+ bottleneck in the AI lifecycle. **Data Integrity Lab** transforms static data cleaning into a dynamic **Markov Decision Process (MDP)**. By modeling data health as a continuous "Integrity Index," this environment allows RL agents to learn optimal, cost-efficient cleaning strategies that adapt to any dataset distribution.

---

## 🏗️ Technical Architecture
The environment is built on a **Modular Auditor Pattern**, ensuring a strict separation between the environment state and the evaluation logic:

1.  **The Core (`gym_env.py`)**: A deterministic state machine that manages data transitions and maintains episode boundaries.
2.  **The Evaluator (`calculate_integrity`)**: A programmatic grader computing the **Heuristic Integrity Index ($I$)**:
    $$I = (1.0 - \frac{\text{Duplicates}}{\text{Total Rows}}) \times (1.0 - \frac{\text{Null Cells}}{\text{Total Cells}})$$
3.  **The Interface (`app.py`)**: A high-performance FastAPI wrapper compliant with the **OpenEnv v2.0** specification.
4.  **Reward Shaping**: To optimize for both accuracy and efficiency, we implement a **Dense Reward Function**:
    $$R_t = (I_{t} - I_{t-1}) \times 100 - \text{Step Penalty}$$

---

## 🕹️ Environment Specification

### **Action Space (Discrete)**
Agents interact via the `/step` endpoint using the following atomic operations:
* `drop_duplicates`: Eliminates redundant row vectors to improve uniqueness.
* `fill_median`: Imputes missing numerical values using statistical medians for stability.
* `drop_nulls`: Removes incomplete observations to harden the dataset.

### **Observation Space (Structured)**
A rich state representation is returned after every action to provide the agent with environmental context:
* **health_score**: A float [0.0–1.0] representing the current "Integrity Index".
* **summary**: Descriptive statistics (mean, std, min, max) of the current data features.
* **sample_rows**: A 5-row contextual window of the raw dataframe for LLM-based reasoning.

### **🏆 Tasks & Difficulty Progression**
| Task | Difficulty | Description | Target Score |
| :--- | :--- | :--- | :--- |
| **Easy** | Low | Single-step duplicate removal from small sets. | 1.0 |
| **Medium** | Moderate | Mixed noise (Nulls + Duplicates) requiring 1-2 steps. | 1.0 |
| **Hard** | High | Scaled (100+ rows) real-world distribution with complex noise. | 1.0 |

---

## 🚀 Getting Started

### **Local Deployment (Docker)**
Ensure Docker Desktop is running and execute:
```bash
# Build the container
docker build -t data-integrity-lab .

# Run the environment
docker run -p 8000:8000 data-integrity-lab
