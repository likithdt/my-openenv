---
title: Data Integrity Lab
emoji: 📊
colorFrom: blue
colorTo: pink
sdk: docker
app_port: 8000
---

# Data Integrity Lab (OpenEnv)
### Autonomous Reinforcement Learning Environment for Data Auditing

[![OpenEnv Spec](https://img.shields.io/badge/OpenEnv-v2.0-blue)](https://github.com/openenv/spec)
[![Framework: FastAPI](https://img.shields.io/badge/Framework-FastAPI-009688)](https://fastapi.tiangolo.com/)

## Motivation
Data cleaning is a manual bottleneck in the AI lifecycle. Data Integrity Lab transforms static data cleaning into a dynamic Markov Decision Process (MDP).

## Technical Architecture
1. **The Core (gym_env.py)**: A deterministic state machine managing data transitions.
2. **The Evaluator (calculate_integrity)**: Computes the Integrity Index (I).
3. **The Interface (app.py)**: High-performance FastAPI wrapper compliant with OpenEnv v2.0.

## Environment Specification
* **drop_duplicates**: Eliminates redundant row vectors.
* **fill_median**: Imputes missing numerical values.
* **drop_nulls**: Removes incomplete observations.

## Getting Started
Ensure Docker is running and execute:
```bash
docker build -t data-integrity-lab .
docker run -p 8000:8000 data-integrity-lab