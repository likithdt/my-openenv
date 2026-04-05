import requests
import time

BASE_URL = "https://likithdt-data-integrity-lab.hf.space"

def run_inference():
    res = requests.post(f"{BASE_URL}/reset", params={"task_id": "easy"})
    action = {"command": "drop_duplicates", "target_column": None}
    res = requests.post(f"{BASE_URL}/step", json=action)
    print(f"🚀 Starting Inference Test on {BASE_URL}")
    
    try:
        print("\n--- Step 1: Resetting to EASY task ---")
        res = requests.post(f"{BASE_URL}/reset", params={"task_id": "easy"}, timeout=10)
        
        if res.status_code != 200:
            print(f"❌ Server Error {res.status_code}: {res.text}")
            return

        data = res.json()
        initial_score = data.get('health_score')
        print(f"Initial Health Score: {initial_score}")

        print("\n--- Step 2: Agent taking action 'drop_duplicates' ---")
        action = {"command": "drop_duplicates"}
        res = requests.post(f"{BASE_URL}/step", json=action, timeout=10)
        
        if res.status_code != 200:
            print(f"❌ Step Error: {res.text}")
            return

        result = res.json()
        
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
