"""
FastAPI server for the Data Integrity Lab (my-env) OpenEnv environment.

Endpoints (OpenEnv simulation-mode contract):
    POST /reset   — reset environment, return initial observation
    POST /step    — apply cleaning action, return observation + reward + done
    GET  /state   — return current environment state
    GET  /health  — {"status": "healthy"}  (required by openenv validate)
    POST /upload  — load a user CSV (optional convenience endpoint)
    GET  /{path}  — serve UI

Each task score (health_score in every observation) is always strictly in
the open interval (0.01, 0.99), satisfying the Phase 2 validator requirement
that task scores are strictly between 0 and 1.
"""

import uvicorn
import os
import sys
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

try:
    from gym_env import DataCleaningEnv
except ImportError:
    try:
        from server.gym_env import DataCleaningEnv
    except ImportError:
        from .gym_env import DataCleaningEnv

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Data Integrity Lab",
    description=(
        "An OpenEnv environment where agents learn to clean dirty datasets "
        "using RL-compatible step/reset/state APIs."
    ),
    version="1.0.0",
)

env_instance = DataCleaningEnv()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Required OpenEnv endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """
    Health check endpoint.
    openenv validate requires: {"status": "healthy"}
    """
    return {"status": "healthy"}


@app.get("/state")
async def get_state():
    """Return current environment state."""
    return env_instance.state()


@app.post("/reset")
async def reset(request: Request):
    """
    Reset the environment for a given task.
    Body (JSON, optional): {"task_id": "easy" | "medium" | "hard"}
    Returns: {"observation": <DataObservation dict>}
    """
    try:
        data = await request.json()
        task_id = data.get("task_id", "easy") if isinstance(data, dict) else "easy"
    except Exception:
        task_id = "easy"

    obs = env_instance.reset(task_id=task_id)

    # Serialize Pydantic model to plain dict for JSON response
    obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else dict(obs)

    return JSONResponse({
        "status": "success",
        "observation": obs_dict,
    })


@app.post("/step")
async def step(request: Request):
    """
    Apply one cleaning action.
    Body (JSON): {"command": "drop_duplicates" | "fill_median" | "drop_nulls",
                  "target_column": "<col>" (optional)}
    Returns: {"observation": ..., "reward": float, "done": bool,
              "score": float, "history": [...]}

    score is always strictly within (0.01, 0.99).
    """
    action = await request.json()
    result = env_instance.step(action)

    # Serialize nested Pydantic observation model
    obs = result.get("observation")
    if hasattr(obs, "model_dump"):
        result["observation"] = obs.model_dump()

    return JSONResponse(result)


# ---------------------------------------------------------------------------
# Optional / convenience endpoints
# ---------------------------------------------------------------------------

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """Upload a CSV to use as the environment dataset."""
    content = await file.read()
    decoded = content.decode("utf-8")
    obs     = env_instance.load_user_data(decoded)
    obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else dict(obs)
    return JSONResponse({"status": "success", "observation": obs_dict})


@app.get("/{path:path}")
async def serve_ui(path: str):
    """Serve the web UI if present."""
    ui_path = os.path.join(CURRENT_DIR, "index.html")
    if os.path.exists(ui_path):
        return FileResponse(ui_path)
    return HTMLResponse(
        "<h1>Data Integrity Lab</h1>"
        "<p>POST /reset to start. POST /step to act. GET /state for state.</p>"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()