import requests
import time

# Ensure there is NO trailing slash here
BASE_URL = "https://likithdt-data-integrity-lab.hf.space"

def run_inference():
    # 1. Reset
    res = requests.post(f"{BASE_URL}/reset", params={"task_id": "easy"})
    # 2. Step
    # Explicitly define the JSON payload to match CleanAction model
    action = {"command": "drop_duplicates", "target_column": None}
    res = requests.post(f"{BASE_URL}/step", json=action)
    print(f"🚀 Starting Inference Test on {BASE_URL}")
    
    try:
        # 1. Reset to 'Easy' Task
        # Note: Using params={} ensures the query string is formatted correctly
        print("\n--- Step 1: Resetting to EASY task ---")
        res = requests.post(f"{BASE_URL}/reset", params={"task_id": "easy"}, timeout=10)
        
        if res.status_code != 200:
            print(f"❌ Server Error {res.status_code}: {res.text}")
            return

        data = res.json()
        # Accessing nested health_score if it's inside the observation
        initial_score = data.get('health_score')
        print(f"Initial Health Score: {initial_score}")

        # 2. Take Action: drop_duplicates
        print("\n--- Step 2: Agent taking action 'drop_duplicates' ---")
        action = {"command": "drop_duplicates"}
        res = requests.post(f"{BASE_URL}/step", json=action, timeout=10)
        
        if res.status_code != 200:
            print(f"❌ Step Error: {res.text}")
            return

        result = res.json()
        
        # 3. Check Final Score (Looking inside the observation dictionary)
        final_score = result['observation']['health_score']
        print(f"Final Health Score: {final_score}")
        
        if final_score >= 0.99:
            print("\n✅ SUCCESS: Agent solved the environment!")
            print(f"Full History: {result.get('history')}")
        else:
            print(f"\n⚠️ PARTIAL SUCCESS: Score improved to {final_score}")

    except Exception as e:
        print(f"💥 Runtime Error: {e}")

if __name__ == "__main__":
    run_inference()
