import os
import requests
import textwrap
import time
from typing import List, Optional, Dict
from openai import OpenAI

API_BASE_URL = "https://router.huggingface.co/v1"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN")
BASE_URL = os.getenv("BASE_URL", "https://likithdt-data-integrity-lab.hf.space")

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)

def get_llm_action(observation: Dict) -> str:
    """Uses direct requests to bypass potential local networking issues with the OpenAI client."""
    score = observation.get('health_score', 0.0)
    goal = observation.get('goal', 'Clean the data.')
    
    prompt = f"Role: Data Quality Agent. Score: {score}. Goal: {goal}. Respond with only: drop_duplicates, drop_nulls, or fill_median."

    clean_api = API_BASE_URL.split(']')[0].split('(')[0].strip()
    full_url = f"{clean_api}/chat/completions"

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10,
        "temperature": 0
    }
    
    headers = {
        "Authorization": f"Bearer {HF_TOKEN.strip()}", 
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            full_url, 
            json=payload, 
            headers=headers, 
            timeout=30
        )
        response.raise_for_status()
        res_json = response.json()
        return res_json['choices'][0]['message']['content'].strip().lower()
    except Exception as e:
        print(f"[DEBUG] LLM API Call Failed: {e}", flush=True)
        return "none"

def run_benchmark():
    if not HF_TOKEN or "your_actual_token" in HF_TOKEN:
        print("[DEBUG] CRITICAL: HF_TOKEN is not set correctly. Use '$env:HF_TOKEN=\"your_key\"' in PowerShell.")
        return

    tasks = ["easy", "medium", "hard"]
    
    for task_id in tasks:
        log_start(task=task_id, env="data-integrity-lab", model=MODEL_NAME)
        
        rewards = []
        steps_taken = 0
        success = False
        
        try:
            clean_base = BASE_URL.strip("[]() ")
            reset_res = requests.post(f"{clean_base}/reset", params={"task_id": task_id}, timeout=15)
            reset_res.raise_for_status()
            obs = reset_res.json()
            
            for step in range(1, 9):
                action_cmd = get_llm_action(obs)
                
                step_res = requests.post(f"{clean_base}/step", json={"command": action_cmd}, timeout=15)
                step_res.raise_for_status()
                result = step_res.json()
                
                reward = result.get('reward', 0.0)
                done = result.get('done', False)
                obs = result.get('observation', {})
                error = result.get('error', None)
                
                rewards.append(reward)
                steps_taken = step
                
                log_step(step=step, action=action_cmd, reward=reward, done=done, error=error)
                
                if done:
                    success = (obs.get('health_score', 0.0) >= 0.99)
                    break
                    
        except Exception as e:
            print(f"[DEBUG] Environment Connection Error: {e}", flush=True)
        
        log_end(success=success, steps=steps_taken, rewards=rewards)

if __name__ == "__main__":
    run_benchmark()
