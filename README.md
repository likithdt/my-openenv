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