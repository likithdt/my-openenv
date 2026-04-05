import os
import requests
import textwrap
from typing import List, Optional, Dict
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

BASE_URL = "https://likithdt-data-integrity-lab.hf.space"

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)

def get_llm_action(client: OpenAI, observation: Dict) -> str:
    feedback = observation.get('goal', 'No feedback.')
    score = observation.get('health_score', 0.0)
    
    prompt = textwrap.dedent(f"""
        Role: Data Quality Agent
        Objective: Reach Integrity Score 1.0.
        Current Score: {score}
        Feedback: {feedback}
        
        Instructions:
        - If 'Found duplicates' > 0, use 'drop_duplicates'.
        - If 'Found nulls' > 0, use 'drop_nulls'.
        - If nulls are 0 and duplicates are 0, you are done.
        
        Available Commands: ['drop_duplicates', 'drop_nulls', 'fill_median']
        Respond with ONLY the command name.
    """).strip()

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=50
        )
        return (completion.choices[0].message.content or "none").strip().lower()
    except Exception as e:
        return f"error: {str(e)}"

def run_benchmark():
    if not HF_TOKEN:
        print("[DEBUG] Error: HF_TOKEN environment variable is missing.")
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    tasks = ["easy", "medium", "hard"]
    
    for task_id in tasks:
        log_start(task=task_id, env="data-integrity-lab", model=MODEL_NAME)
        
        rewards = []
        steps_taken = 0
        success = False
        
        try:
            reset_res = requests.post(f"{BASE_URL}/reset", params={"task_id": task_id}, timeout=10)
            obs = reset_res.json()
            
            for step in range(1, 9):
                action_cmd = get_llm_action(client, obs)
                
                step_res = requests.post(f"{BASE_URL}/step", json={"command": action_cmd}, timeout=10)
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
            print(f"[DEBUG] Execution Error on task {task_id}: {e}")
        
        log_end(success=success, steps=steps_taken, rewards=rewards)

if __name__ == "__main__":
    run_benchmark()
