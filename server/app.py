import uvicorn
import pandas as pd
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from server.gym_env import DataCleaningEnv
from server.models import CleanAction

# 1. Initialize the FastAPI app
app = FastAPI(
    title="Data Integrity Lab - Pro Edition",
    description="A Universal RL Gym for Automated Data Cleaning",
    version="2.0.0"
)

# 2. Global Environment Instance
# Starts with a default messy sample, ready to be overwritten by uploads
env_instance = DataCleaningEnv()

@app.get("/")
def home():
    """
    Main Landing Page: Fixes 'Not Found' and shows current Data Health.
    """
    return {
        "status": "Online",
        "project": "Data Integrity Lab (Dynamic Edition)",
        "current_integrity_index": env_instance.calculate_integrity(),
        "total_steps_taken": env_instance.step_count,
        "message": "Visit /docs to upload your own CSV and start the RL cleaning process."
    }

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """
    THE FINALIST FEATURE: Dynamically re-initializes the gym with user data.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .csv file.")
    
    try:
        contents = await file.read()
        # Read the CSV into a DataFrame
        new_df = pd.read_csv(io.BytesIO(contents))
        
        # Overwrite the global instance with the new data
        global env_instance
        env_instance = DataCleaningEnv(df=new_df)
        
        return {
            "message": f"Successfully loaded {file.filename}",
            "rows": len(new_df),
            "columns": list(new_df.columns),
            "initial_integrity": env_instance.calculate_integrity(),
            "observation": env_instance.reset()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")

@app.post("/reset")
def reset_endpoint():
    """Resets the environment to its initial (uploaded) state."""
    return env_instance.reset()

@app.post("/step")
def step_endpoint(action: CleanAction):
    """Executes a cleaning command and returns the calculated Reward."""
    return env_instance.step(action)

def main():
    # Hugging Face Spaces require the app to listen on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
    