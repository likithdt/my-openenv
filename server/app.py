import uvicorn
import os
import sys
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# Ensure server directory is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

from gym_env import DataCleaningEnv

app = FastAPI()
env_instance = DataCleaningEnv()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
async def health():
    return {
        "integrity": env_instance.calculate_integrity(),
        "steps": env_instance.step_count,
        "data": env_instance.df.head(10).to_dict(orient='records') if not env_instance.df.empty else []
    }

@app.get("/{path:path}")
async def serve_ui(path: str):
    ui_path = os.path.join(CURRENT_DIR, "index.html")
    if os.path.exists(ui_path):
        return FileResponse(ui_path)
    return HTMLResponse("<h1>UI File Not Found</h1>")

@app.get("/state")
async def get_state():
    return env_instance.state()

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    content = await file.read()
    decoded = content.decode('utf-8')
    obs = env_instance.load_user_data(decoded)
    return {"status": "success", "observation": obs}

@app.post("/reset")
async def reset(request: Request):
    env_instance.reset()
    return {"status": "success"}

@app.post("/step")
async def step(request: Request):
    action = await request.json()
    return env_instance.step(action)

@app.get("/{path:path}")
async def serve_ui(path: str):
    ui_path = os.path.join(CURRENT_DIR, "index.html")
    if os.path.exists(ui_path):
        return FileResponse(ui_path)
    return HTMLResponse("<h1>UI File Not Found</h1><p>Upload index.html to the server folder.</p>")

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()