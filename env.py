from pydantic import BaseModel, Field
from enum import Enum
from typing import Tuple, List, Optional
import random

# ------------------ ENUMS ------------------

class DefenderMove(str, Enum):
    scan = "scan"
    defend = "defend"
    allocate = "allocate"

# ------------------ MODELS ------------------

class Observation(BaseModel):
    task: str
    phase_summary: str
    system_integrity: float
    visible_threat_level: Optional[float]
    budget: int
    recent_events_log: List[str]

class Action(BaseModel):
    def_move: DefenderMove = Field(description="The move chosen by the Defender agent")

# ------------------ ENV ------------------

class CrisisOpsEnv:
    def __init__(self, seed: int = None):
        self.max_steps = 15
        self.tasks = ["recon", "defense", "recovery"]
        self.current_task_index = 0
        self.seed_val = seed
        if seed is not None:
            random.seed(seed)

        self.step_count = 0
        self.done = False

        self.system_integrity = 1.0
        self.threat = 0.5 
        
        self.budget = 10
        self.recent_events = []
        
        self.scan_count = 0
        self.defend_count = 0
        self.allocate_count = 0
        
        self.last_action = None
        
        self.integrity_history = [1.0]
        self.visible_threat_history = []
        self.budget_history = [10]
        
        self.task_level = self.tasks[0]

    def reset(self, task: str = None, seed: int = None) -> Observation:
        if seed is not None:
            self.seed_val = seed
            random.seed(seed)
        if task and task in self.tasks:
            self.current_task_index = self.tasks.index(task)
        else:
            self.current_task_index = 0
        return self._init_task(self.tasks[self.current_task_index])

    def _init_task(self, task) -> Observation:
        self.step_count = 0
        self.done = False
        
        self.system_integrity = 1.0
        # If recovery task, start with degraded integrity
        if task == "recovery":
            self.system_integrity = 0.5
            
        self.threat = 0.3
        self.budget = 10
        
        self.recent_events = []
        
        self.scan_count = 0
        self.defend_count = 0
        self.allocate_count = 0
        
        self.last_action = None
        
        self.integrity_history = [self.system_integrity]
        self.visible_threat_history = []
        self.budget_history = [10]
        self.task_level = task

        return self._get_obs()

    def _get_obs(self) -> Observation:
        # Partial Observability: threat is only visible if last action was scan
        vis_threat = round(self.threat, 2) if self.last_action == DefenderMove.scan else None
        
        return Observation(
            task=self.task_level,
            phase_summary=f"Current Objective: {self.task_level.capitalize()}",
            system_integrity=round(self.system_integrity, 2),
            visible_threat_level=vis_threat,
            budget=self.budget,
            recent_events_log=self.recent_events[-3:]
        )

    def state(self) -> dict:
        return {
            "task": self.task_level,
            "step": self.step_count,
            "integrity": self.system_integrity,
            "done": self.done
        }

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        if self.done:
            return self._get_obs(), 0.0, True, {}

        self.step_count += 1
        move_def = action.def_move
        self.last_action = move_def
        
        # 1. Attacker is now a Deterministic Rule-Based Process
        if self.step_count % 2 == 0:
            self.threat += 0.03
            att_type = "escalated"
        else:
            self.threat += 0.02
            att_type = "stealthy"
            
        self.threat = min(2.0, self.threat)

        # 2. Process Defender Action Tradeoffs
        actual_damage = 0.0
        action_success = False
        
        # Tradeoffs logic:
        # scan: info vs no protection (costs 1)
        # defend: safety vs no recovery (costs 1)
        # allocate: recovery vs budget cost (costs 2)

        if move_def == DefenderMove.scan:
            if self.budget >= 1:
                self.budget -= 1
                self.scan_count += 1
                action_success = True
                self.recent_events.append("System scan executed. Threat visibility restored.")
            else:
                self.recent_events.append("Scan failed: Insufficient budget.")
                
        elif move_def == DefenderMove.defend:
            if self.budget >= 1:
                self.budget -= 1
                self.defend_count += 1
                action_success = True
                self.threat = max(0.0, self.threat - 0.25)
                self.recent_events.append("Defensive measures deployed. Threat reduced.")
            else:
                self.recent_events.append("Defense failed: Insufficient budget.")
                
        elif move_def == DefenderMove.allocate:
            if self.budget >= 2:
                self.budget -= 2
                self.allocate_count += 1
                action_success = True
                self.system_integrity += 0.25 # Recovery
                self.recent_events.append("Resources allocated. Integrity recovered.")
            else:
                self.recent_events.append("Allocation failed: Insufficient budget.")

        # 3. Process Attack Damage
        # Base damage is proportional to the threat level
        dmg = self.threat * 0.05
        
        if move_def == DefenderMove.defend and action_success:
            dmg *= 0.2 # Defend mitigates 80% damage
            self.recent_events.append(f"Attack mitigated. Minor damage taken: {dmg:.2f}")
        else:
            self.recent_events.append(f"System undefended. Direct attack damage: {dmg:.2f}")
        
        self.system_integrity -= dmg
        self.system_integrity = max(0.0, min(1.0, self.system_integrity))
        
        # Track history
        self.integrity_history.append(self.system_integrity)
        self.budget_history.append(self.budget)
        
        if move_def == DefenderMove.scan and action_success:
            self.visible_threat_history.append(self.threat)

        # 4. Check End Conditions
        if self.step_count >= self.max_steps or self.system_integrity <= 0.05:
            self.done = True

        info = {
            "task": self.task_level,
            "budget": self.budget
        }

        # Calculate step reward (incremental grade if needed, mostly for logging)
        score = self._compute_current_score()

        return self._get_obs(), score, self.done, info

    def _compute_current_score(self) -> float:
        # Generic helper to just give a step reward value, 
        # actual grade is computed via grading functions at the end.
        return self.system_integrity

    # ------------------ UNIFIED GRADERS ------------------
    # score = A * metric1 + B * metric2
    # passed = score >= threshold

    def grade_recon(self) -> float:
        """Goal: Maximize threat visibility."""
        scan_ratio = self.scan_count / max(1, self.step_count)
        threat_discovery = sum(self.visible_threat_history)
        score = (0.6 * min(1.0, scan_ratio * 2.0)) + (0.4 * min(1.0, threat_discovery * 0.5))
        score = score * 0.98 + 0.01
        return round(score, 4)

    def grade_defense(self) -> float:
        """Goal: Maintain system integrity against attacks."""
        avg_integrity = sum(self.integrity_history) / len(self.integrity_history)
        defend_ratio = self.defend_count / max(1, self.step_count)
        score = (0.4 * avg_integrity) + (0.6 * min(1.0, defend_ratio * 1.5))
        score = score * 0.98 + 0.01
        return round(score, 4)

    def grade_recovery(self) -> float:
        """Goal: Restore system after damage."""
        min_integrity = min(self.integrity_history)
        recovery_delta = self.system_integrity - min_integrity
        alloc_ratio = self.allocate_count / max(1, self.step_count)
        score = (0.6 * min(1.0, recovery_delta * 1.5)) + (0.4 * min(1.0, alloc_ratio * 2.0))
        score = score * 0.98 + 0.01
        return round(score, 4)

    def get_grade_breakdown(self, task: str = None) -> dict:
        """Returns details for transparent, deterministic evaluation."""
        target_task = task or self.task_level
        passed_threshold = 0.5
        
        if target_task == "recon":
            score = self.grade_recon()
            return {
                "score": score,
                "passed": score >= passed_threshold,
                "metrics": {
                    "scans_performed": self.scan_count,
                    "threat_discovery_total": round(sum(self.visible_threat_history), 2)
                }
            }
        elif target_task == "defense":
            score = self.grade_defense()
            avg_integrity = sum(self.integrity_history) / len(self.integrity_history)
            return {
                "score": score,
                "passed": score >= passed_threshold,
                "metrics": {
                    "avg_integrity": round(avg_integrity, 2),
                    "defenses_deployed": self.defend_count
                }
            }
        elif target_task == "recovery":
            score = self.grade_recovery()
            min_integrity = min(self.integrity_history)
            return {
                "score": score,
                "passed": score >= passed_threshold,
                "metrics": {
                    "recovery_delta": round(self.system_integrity - min_integrity, 2),
                    "allocations_made": self.allocate_count
                }
            }
        return {}

    def get_tasks(self) -> List[dict]:
        """Provides metadata for task discovery."""
        return [
            {
                "name": "recon", 
                "description": "Maximize threat visibility using strategic scanning.", 
                "grader": "grade_recon"
            },
            {
                "name": "defense", 
                "description": "Maintain high system integrity using active defenses.", 
                "grader": "grade_defense"
            },
            {
                "name": "recovery", 
                "description": "Restore compromised system integrity via resource allocation.", 
                "grader": "grade_recovery"
            }
        ]