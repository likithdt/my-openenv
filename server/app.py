import uvicorn
import pandas as pd
import io
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from server.gym_env import DataCleaningEnv
from server.models import CleanAction
from fastapi.responses import HTMLResponse

app = FastAPI(
    title="Data Integrity Lab - Competition Edition",
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

env_instance = DataCleaningEnv()

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>Data Integrity Lab | OpenEnv</title>
            <style>
                body { font-family: sans-serif; text-align: center; padding: 50px; background: #f4f7f6; }
                .card { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); display: inline-block; }
                h1 { color: #2c3e50; }
                .status { color: #27ae60; font-weight: bold; }
            </style>
        </head>
        <body>
            <div class="card">
                <h1>📊 Data Integrity Lab</h1>
                <p>Status: <span class="status">RUNNING</span></p>
                <p>OpenEnv v2.0 Specification Compliant</p>
                <hr>
                <p style="font-size: 0.8em; color: #7f8c8d;">Gopalan College of Engineering & Management | Round 1</p>
            </div>
        </body>
    </html>
    """
@app.get("/health")
async def health():
    return {
        "status": "Online", 
        "current_task": env_instance.current_task,
        "integrity": env_instance.calculate_integrity()
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
            "message": "Data Loaded Successfully", 
            "initial_integrity": env_instance.calculate_integrity()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset")
async def reset(task_id: str = Query("easy", enum=["easy", "medium", "hard"])):
    """
    REQUIRED: Resets the environment to a specific grader task.
    """
    return env_instance.reset(task_id=task_id)

@app.post("/step")
async def step(action: CleanAction):
    """
    Executes an agent action and returns Reward/Observation.
    """
    try:
        result = env_instance.step(action)
        return result
    except Exception as e:
        print(f"STEP ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def history():
    """Provides the Audit Trail for the judges."""
    return {
        "task": env_instance.current_task,
        "steps": env_instance.step_count,
        "log": env_instance.history
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

def main():
    """Main entry point for multi-mode deployment."""
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    main()
