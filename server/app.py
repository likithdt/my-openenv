import uvicorn
from fastapi import FastAPI
from server.env import DataCleaningEnv
from server.models import CleanAction

app = FastAPI()
env_instance = DataCleaningEnv()

# 1. The Home Route
@app.get("/")
def home():
    return {"status": "Online", "data": env_instance.reset()}

# 2. The 'Web' Route (To stop those 404 logs you saw)
@app.get("/web")
def web_proxy():
    return {"status": "Redirected", "data": env_instance.reset()}

# 3. The API Endpoints
@app.post("/reset")
def reset_endpoint():
    return env_instance.reset()

@app.post("/step")
def step_endpoint(action: CleanAction):
    return env_instance.step(action)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)