import uvicorn
import os
import sys
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# Ensure server directory is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

from gym_env import DataCleaningEnv

app = FastAPI()
env_instance = DataCleaningEnv()
env_instance.reset(task_id="easy")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# API for your new UI to call
@app.get("/health")
async def health():
    return {
        "integrity": env_instance.calculate_integrity(),
        "steps": env_instance.step_count,
        "data": env_instance.df.head(10).to_dict(orient='records')
    }

@app.post("/reset")
async def reset(request: Request):
    data = await request.json()
    env_instance.reset(task_id=data.get("task_id", "easy"))
    return {"status": "success"}

@app.post("/step")
async def step(request: Request):
    action = await request.json()
    return env_instance.step(action)

# Serve your custom UI file here
@app.get("/{path:path}")
async def serve_ui(path: str):
    ui_path = os.path.join(CURRENT_DIR, "index.html")
    if os.path.exists(ui_path):
        return FileResponse(ui_path)
    return HTMLResponse("<h1>UI File Not Found</h1><p>Please ensure index.html is in the server/ folder.</p>")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
    