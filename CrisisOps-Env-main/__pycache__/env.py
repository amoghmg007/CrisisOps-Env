from pydantic import BaseModel

class Observation(BaseModel):
    task: str
    feedback: str

class Action(BaseModel):
    code: str
    explanation: str

class AntigravityEnv:
    def __init__(self):
        self.current_task = ""
        self.done = False

    def reset(self):
        self.current_task = "Write a function to return sum of two numbers"
        self.done = False
        return Observation(task=self.current_task, feedback="Start coding")

    def step(self, action: Action):
        score = 0.0

        if "def" in action.code:
            score += 0.4

        if "return" in action.code:
            score += 0.3

        if len(action.explanation) > 10:
            score += 0.3

        if score >= 0.8:
            self.done = True
            feedback = "Good job!"
        else:
            feedback = "Improve your solution"

        return {
            "observation": Observation(task=self.current_task, feedback=feedback),
            "reward": score,
            "done": self.done,
            "info": {}
        }

    def state(self):
        return {
            "task": self.current_task,
            "done": self.done
        }