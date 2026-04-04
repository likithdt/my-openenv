import uvicorn
import pandas as pd
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from server.gym_env import DataCleaningEnv
from server.models import CleanAction

app = FastAPI(
    title="Data Integrity Lab",
    docs_url="/",       
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the environment globally
env_instance = DataCleaningEnv()

@app.get("/health")
async def health():
    """Explicitly defined root to kill the 404 error."""
    return {
        "status": "Online",
        "integrity_index": env_instance.calculate_integrity(),
        "info": "Navigate to /docs to upload and clean data."
    }

@app.get("/history")
async def get_history():
    """
    THE AUDIT TRAIL: Shows every cleaning step taken by the AI.
    """
    return {
        "total_steps": env_instance.step_count,
        "final_integrity": env_instance.calculate_integrity(),
        "log": env_instance.history
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
        return {"message": "Success", "initial_integrity": env_instance.calculate_integrity()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset")
async def reset():
    return env_instance.reset()

@app.post("/step")
async def step(action: CleanAction):
    return env_instance.step(action)

# This is the CRITICAL part for Hugging Face
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
