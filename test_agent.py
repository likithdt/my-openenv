import requests
import time

# Update this with your Hugging Face Space URL
BASE_URL = "https://likithdt-data-integrity-lab.hf.space"

def run_inference():
    print(f"🚀 Starting Inference Test on {BASE_URL}")
    
    # 1. Reset to 'Easy' Task
    print("\n--- Step 1: Resetting to EASY task ---")
    res = requests.post(f"{BASE_URL}/reset", params={"task_id": "easy"})
    data = res.json()
    print(f"Initial Health Score: {data['health_score']}")

    # 2. Take Action: drop_duplicates
    print("\n--- Step 2: Agent taking action 'drop_duplicates' ---")
    action = {"command": "drop_duplicates"}
    res = requests.post(f"{BASE_URL}/step", json=action)
    result = res.json()
    
    # 3. Check Final Score
    final_score = result['observation']['health_score']
    print(f"Final Health Score: {final_score}")
    
    if final_score >= 0.99:
        print("\n✅ SUCCESS: Agent solved the environment!")
    else:
        print("\n❌ FAILED: Score below threshold.")

if __name__ == "__main__":
    run_inference()
    