import uvicorn
import os
import sys
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# Ensure the server directory is in the path for imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from gym_env import DataCleaningEnv #

# Create the app with a specific root_path to handle Hugging Face proxying
app = FastAPI(title="Data Integrity Lab", root_path="") 

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env_instance = DataCleaningEnv() #

# Catch-all route to diagnose where the request is actually landing
@app.get("/{full_path:path}", response_class=HTMLResponse)
async def catch_all(request: Request, full_path: str):
    return f"""
    <html>
        <body style="font-family: sans-serif; text-align: center; padding: 50px;">
            <h1>📊 Data Integrity Lab: Online</h1>
            <p style="color: green; font-weight: bold;">Server is responding!</p>
            <p>You tried to access: <code>/{full_path}</code></p>
            <hr>
            <p>API is active at <code>/reset</code> and <code>/step</code></p>
            <p style="font-size: 0.8em;">Likith D T | Gopalan College</p>
        </body>
    </html>
    """

@app.post("/reset")
async def reset():
    env_instance.reset() #
    return {"status": "success"}

@app.post("/step")
async def step(request: Request):
    data = await request.json() 
    return env_instance.step(data) 
