import uvicorn
import pandas as pd
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from server.gym_env import DataCleaningEnv # Ensure this matches your filename
from server.models import CleanAction

app = FastAPI(title="Data Integrity Lab - Pro Edition")

# Initialize the environment
env_instance = DataCleaningEnv()

@app.get("/")
def home():
    """
    This fixes the 'Not Found' error by providing a landing page.
    """
    return {
        "status": "Online",
        "project": "Data Integrity Lab (Dynamic Edition)",
        "current_integrity": env_instance.calculate_integrity(),
        "state": env_instance.state()
    }

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type.")
    
    try:
        contents = await file.read()
        new_df = pd.read_csv(io.BytesIO(contents))
        
        global env_instance
        env_instance = DataCleaningEnv(df=new_df)
        
        return {
            "message": f"Successfully loaded {file.filename}",
            "initial_integrity": env_instance.calculate_integrity(),
            "observation": env_instance.reset()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset")
def reset_endpoint():
    return env_instance.reset()

@app.post("/step")
def step_endpoint(action: CleanAction):
    return env_instance.step(action)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    