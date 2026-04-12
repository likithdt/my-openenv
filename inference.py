import os
import requests
import time
from typing import List, Optional, Dict
from openai import OpenAI

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)

def get_llm_action(client: OpenAI, model_name: str, observation: Dict) -> str:
    """
    Standard OpenAI Client call formatted for the LiteLLM Proxy.
    """
    score = observation.get('health_score', 0.35)
    goal = observation.get('goal', 'Clean the data.')
    
    prompt = f"Role: Data Quality Agent. Score: {score}. Goal: {goal}. Respond with only one word: drop_duplicates, drop_nulls, or fill_median."

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0,
            stream=False,
        )
        action = (completion.choices[0].message.content or "").strip().lower()
        
        valid_actions = ["drop_duplicates", "drop_nulls", "fill_median"]
        for valid in valid_actions:
            if valid in action:
                return valid
        return "none"
        
    except Exception as e:
        print(f"[DEBUG] Proxy Call Failed: {e}", flush=True)
        return "none"

def run_benchmark():
    api_base = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    api_key = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
    model_name = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
    
    raw_base_url = os.getenv("BASE_URL", "https://likithdt-my-env.hf.space")
    clean_base = raw_base_url.strip("[]() ")

    if not api_key:
        print("[DEBUG] CRITICAL: No API_KEY or HF_TOKEN environment variable found.")
        return

    client = OpenAI(base_url=api_base, api_key=api_key)

    env_name = "my-env"
    tasks = ["easy", "medium", "hard"]
    
    for task_id in tasks:
        log_start(task=task_id, env=env_name, model=model_name)
        
        rewards = []
        steps_taken = 0
        success = False
        
        try:
            reset_res = requests.post(f"{clean_base}/reset", json={"task_id": task_id}, timeout=15)
            reset_res.raise_for_status()
            obs_data = reset_res.json()
            
            obs = obs_data.get("observation", obs_data) if isinstance(obs_data, dict) else {}

            for step in range(1, 9):
                action_cmd = get_llm_action(client, model_name, obs)
                
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
                    health = obs.get('health_score', 0.35)
                    success = (health > 0.82)
                    break
                    
        except Exception as e:
            print(f"[DEBUG] Connection Error on Task {task_id}: {e}", flush=True)
        
        log_end(success=success, steps=steps_taken, rewards=rewards)

if __name__ == "__main__":
    run_benchmark()