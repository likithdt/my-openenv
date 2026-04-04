import uvicorn
import pandas as pd
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from server.env import DataCleaningEnv
from server.models import CleanAction

# Initialize the FastAPI app
app = FastAPI(title="Data Integrity Lab - Pro Edition")

# Initialize the environment with a default messy dataset
env_instance = DataCleaningEnv()

@app.get("/")
def home():
    """
    Main landing page showing current Data Health.
    """
    return {
        "status": "Online",
        "project": "Data Integrity Lab (Universal Edition)",
        "current_integrity_index": env_instance.calculate_integrity(),
        "observation": env_instance.reset()
    }

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """
    THE FINALIST FEATURE: Upload any CSV file to clean it!
    This proves your AI works on ANY data, not just hardcoded samples.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV.")
    
    try:
        contents = await file.read()
        # Load the uploaded data into a Pandas DataFrame
        new_df = pd.read_csv(io.BytesIO(contents))
        
        # Re-initialize the environment with the NEW data
        global env_instance
        env_instance = DataCleaningEnv(df=new_df)
        
        return {
            "message": f"Successfully loaded {file.filename}",
            "rows": len(new_df),
            "initial_integrity": env_instance.calculate_integrity(),
            "observation": env_instance.reset()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")

@app.post("/reset")
def reset_endpoint():
    """Resets the current dataset to its uploaded state."""
    return env_instance.reset()

@app.post("/step")
def step_endpoint(action: CleanAction):
    """Executes a cleaning command and returns the reward (Integrity Improvement)."""
    return env_instance.step(action)

def main():
    # Hugging Face requires port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()