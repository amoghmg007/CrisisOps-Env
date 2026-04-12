from fastapi import FastAPI
from pydantic import BaseModel
from env import CrisisOpsEnv, Action

app = FastAPI()
env_instance = CrisisOpsEnv()

@app.get("/")
def read_root():
    return {"status": "ok", "message": "CrisisOps Env Server Running"}

@app.get("/health")
def health():
    return {"status": "healthy", "env": "CrisisOpsEnv"}

@app.post("/reset")
def reset(task: str = None):
    obs = env_instance.reset(task=task)
    return obs.dict()

@app.post("/step")
def step(action: Action):
    obs, reward, done, info = env_instance.step(action)
    return {
        "observation": obs.dict(),
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state")
def state():
    return env_instance.state()

@app.get("/tasks")
def get_tasks():
    return env_instance.get_tasks()

@app.get("/grade")
def get_grade(task: str = None):
    valid_tasks = ["recon", "defense", "recovery"]
    if task and task not in valid_tasks:
        return {"error": "invalid task"}
        
    # Determine the grader logic based on the task name or current env state
    target_task = task or env_instance.task_level
    
    score = 0.0
    if target_task == "recon":
        score = env_instance.grade_recon()
    elif target_task == "defense":
        score = env_instance.grade_defense()
    elif target_task == "recovery":
        score = env_instance.grade_recovery()
    
    return {
        "score": score,
        "breakdown": env_instance.get_grade_breakdown(task=target_task)
    }

@app.get("/breakdown")
def get_breakdown(task: str = None):
    return env_instance.get_grade_breakdown(task=task)

@app.get("/grade_recon")
def grade_recon_endpoint():
    return {"score": env_instance.grade_recon()}

@app.get("/grade_defense")
def grade_defense_endpoint():
    return {"score": env_instance.grade_defense()}

@app.get("/grade_recovery")
def grade_recovery_endpoint():
    return {"score": env_instance.grade_recovery()}

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
