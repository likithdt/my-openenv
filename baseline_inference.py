import os
import requests
from openai import OpenAI
from dotenv import load_dotenv

# 1. Load Environment Variables from .env file
load_dotenv()

# 2. Configuration
BASE_URL = "https://likithdt-data-integrity-lab.hf.space"
HF_TOKEN = os.getenv("HF_TOKEN")

# Pick a reliable model from the HF Router
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

# Initialize the client pointing to Hugging Face's Router
client = OpenAI(
    base_url="https://router.huggingface.co/v1", 
    api_key=HF_TOKEN
)

def get_llm_action(observation: dict) -> str:
    """
    Analyzes the observation and environment feedback to choose an action.
    """
    # Extract the 'goal' field where we stored the duplicate/null counts
    feedback = observation.get('goal', 'No feedback available.')
    score = observation.get('health_score', 0.0)
    
    prompt = f"""
    ROLE: Expert Data Cleaning Agent
    OBJECTIVE: Reach Integrity Score 1.0
    
    CURRENT STATE:
    - Integrity Score: {score}
    - Environment Feedback: {feedback}
    - Data Summary: {observation.get('summary')}
    
    INSTRUCTIONS:
    1. Read the 'Environment Feedback' carefully. 
    2. If it says 'Found X duplicates' where X > 0, use 'drop_duplicates'.
    3. If it says 'Found X nulls' where X > 0, use 'drop_nulls'.
    4. Do NOT repeat an action if the feedback says 'No change'.
    
    Available Commands: ['drop_duplicates', 'fill_median', 'drop_nulls']
    
    Response format: Respond with ONLY the command name.
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        # Clean the output to ensure it's just the command
        return response.choices[0].message.content.strip().lower()
    except Exception as e:
        print(f"💥 LLM API Error: {e}")
        return "none"

def run_task(task_id: str):
    print(f"\n🚀 Testing Task: {task_id.upper()} using {MODEL_ID}")
    
    # --- RESET ---
    try:
        res = requests.post(f"{BASE_URL}/reset", params={"task_id": task_id}, timeout=15)
        if res.status_code != 200:
            print(f"❌ Reset Failed: {res.text}")
            return
        obs = res.json()
    except Exception as e:
        print(f"💥 Connection Error during Reset: {e}")
        return

    # --- STEP LOOP ---
    steps = 0
    while obs.get('health_score', 0.0) < 0.99 and steps < 5:
        # 1. Ask LLM for action
        action_cmd = get_llm_action(obs)
        print(f"Step {steps+1}: LLM decided to '{action_cmd}'")
        
        # 2. Execute Action in Environment
        try:
            step_res = requests.post(f"{BASE_URL}/step", json={"command": action_cmd}, timeout=15)
            
            if step_res.status_code != 200:
                print(f"❌ Step API Error {step_res.status_code}: {step_res.text}")
                break

            result = step_res.json()
            
            # OpenEnv Compliance Check: Handle different response structures
            if 'observation' in result:
                obs = result['observation']
            else:
                obs = result # Fallback if response is flat
                
            print(f"   -> Resulting Score: {obs.get('health_score', 'Unknown')}")
            steps += 1
            
        except Exception as e:
            print(f"💥 Connection Error during Step: {e}")
            break

    # --- FINAL GRADER ---
    final_score = obs.get('health_score', 0.0)
    if final_score >= 0.99:
        print(f"✅ SUCCESS: Task '{task_id}' solved in {steps} steps!")
    else:
        print(f"❌ FAILED: Task '{task_id}' ended with score {final_score}")

if __name__ == "__main__":
    if not HF_TOKEN:
        print("❌ ERROR: HF_TOKEN not found! Check your .env file.")
    else:
        # Verify both tasks to ensure zero-shot generalization
        for task in ["easy", "medium","hard"]:
            run_task(task)
