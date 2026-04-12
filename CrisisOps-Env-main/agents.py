import random
from env import DefenderMove

class BaselineDefender:
    def __init__(self, mode="random"):
        self.mode = mode
    
    def get_move(self, obs, step):
        if self.mode == "random":
            # Bad baseline (wastes budget on wrong things)
            if obs.task == "recon":
                return DefenderMove.defend if step % 2 == 0 else DefenderMove.allocate
            elif obs.task == "defense":
                return DefenderMove.allocate if step % 2 == 0 else DefenderMove.scan
            else:
                return DefenderMove.defend if step % 2 == 0 else DefenderMove.scan
        elif self.mode == "greedy":
            # Ensure "survives moderately" profile
            if obs.task == "recon":
                return DefenderMove.scan if step % 3 == 0 else DefenderMove.defend
            elif obs.task == "defense":
                if obs.budget >= 1 and step % 2 == 0:
                    return DefenderMove.defend
                return DefenderMove.scan
            else:
                if obs.budget >= 2 and obs.system_integrity < 0.7:
                    return DefenderMove.allocate
                return DefenderMove.scan
        return DefenderMove.scan
