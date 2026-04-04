---
title: Data Integrity Lab
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
---

# 📊 Data Integrity Lab: Autonomous RL Cleaning Environment
**Project Category:** AI for Data Engineering / Automated Machine Learning (AutoML)

## 🚀 The Vision
Data Integrity Lab is a **Universal Reinforcement Learning (RL) Gym** designed to automate the most tedious part of data science: Data Cleaning. Unlike traditional scripts, this environment uses a **Heuristic Integrity Index** to evaluate data health, allowing RL agents to learn optimal cleaning strategies for *any* dataset.

---

## 🧠 Core Architecture: The Integrity Index ($I$)
To move beyond hardcoded logic, we implemented a mathematical Auditor that calculates a global quality score for any uploaded CSV:

$$I = \text{Uniqueness} \times \text{Completeness}$$

* **Uniqueness ($u$):** $1.0 - (\text{Duplicate Rows} / \text{Total Rows})$
* **Completeness ($c$):** $1.0 - (\text{Null Cells} / \text{Total Cells})$

**The RL Feedback Loop:**
The Environment issues a **Reward** based on the delta of $I$ after every action:
$$Reward = (I_{after} - I_{before}) \times 100$$

---

## 🛠️ Key Features (Round 2: Universal Edition)
- **Dynamic Data Injection:** Use the `/upload` endpoint to transform the gym into a custom environment for *your* specific data.
- **Data-Agnostic Rewards:** The AI agent maximizes the "Integrity Index," making it compatible with Medical, Financial, or Retail datasets.
- **FastAPI Backend:** A high-performance, containerized API ready for large-scale deployment.
- **Interactive Swagger Docs:** Test the environment's "Step" and "Reset" functions in real-time via the `/docs` UI.

---

## 📂 Project Structure
```text
/server
  ├── app.py      # FastAPI Server & File Upload System
  ├── env.py      # The RL Gym Logic (Universal Data Auditor)
  └── models.py   # Pydantic Schemas for Data Validation
Dockerfile        # Containerization for Hugging Face Spaces
requirements.txt  # Project Dependencies

## 🤖 OpenEnv Specification

### Action Space
The agent can perform **Discrete(3)** actions via the `/step` endpoint:
- `drop_duplicates`: Removes redundant rows ($I \uparrow$).
- `fill_median`: Imputes missing numerical values ($I \uparrow$).
- `drop_nulls`: Removes rows with missing values ($I \uparrow$).

### Observation Space
A dictionary returned after every step:
- `health_score`: Float [0.0 - 1.0] (The Grader's metric).
- `summary`: Statistical distribution of the current data.
- `sample_rows`: A 5-row preview for context.

### 🏆 Tasks & Graders
1. **Easy**: Remove duplicates from a 3-row set.
2. **Medium**: Handle missing values and duplicates simultaneously.
3. **Hard**: Scaled cleaning on user-uploaded CSV data.

# 📊 Data Integrity Lab (OpenEnv)

## 🎯 Motivation
Data cleaning is a manual, $100B+ problem. This environment simulates a real-world "Data Auditor" task where an AI agent must identify and fix structural issues (duplicates, nulls, outliers) to achieve 100% data integrity.

## 🕹️ Spaces
### Action Space (Discrete)
- `drop_duplicates`: Removes redundant rows.
- `fill_median`: Imputes missing numeric data.
- `drop_nulls`: Removes rows with incomplete information.

### Observation Space (Box/Dict)
- `health_score`: 0.0 to 1.0 (The grader metric).
- `summary`: Statistical distribution (Pandas describe).
- `sample_rows`: A 5-row glimpse of the raw data.

## 🏆 Baseline Scores
| Task | Strategy | Score |
| :--- | :--- | :--- |
| Easy | GPT-4o (Zero-shot) | 1.0 |
| Medium | GPT-4o (Zero-shot) | 1.0 |
| Hard | GPT-4o (Few-shot) | 0.85 |

## 🏗️ Technical Architecture
The environment follows a **Modular Auditor Pattern**:
1. **The Core (`gym_env.py`)**: Implements an OpenAI Gym-like interface with deterministic state resets.
2. **The Evaluator (`calculate_integrity`)**: A programmatic grader that computes the density of duplicates and nulls as a continuous metric ($0.0 \rightarrow 1.0$).
3. **The Interface (`app.py`)**: A FastAPI wrapper that exposes the environment via a REST API, compliant with the **OpenEnv v2.0** specification.
4. **Reward Shaping**: Implements a dense reward function ($R_t = \Delta \text{Score} - \text{Cost}$) to incentivize both accuracy and efficiency.

