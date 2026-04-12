from env import CrisisOpsEnv, Action, DefenderMove, AttackerMove
import json

def run_demo():
    print("--- 🚀 CrisisOps Environment Demo ---")
    env = CrisisOpsEnv()
    obs = env.reset()
    
    print("\n[INITIAL OBSERVATION]")
    print(json.dumps(obs.dict(), indent=2))
    
    # Simulate a few steps with manual/predefined actions
    demo_actions = [
        Action(def_move=DefenderMove.scan, att_move=AttackerMove.mislead),
        Action(def_move=DefenderMove.patch, att_move=AttackerMove.attack),
        Action(def_move=DefenderMove.allocate, att_move=AttackerMove.attack),
        Action(def_move=DefenderMove.defend, att_move=AttackerMove.attack)
    ]
    
    for i, action in enumerate(demo_actions):
        print(f"\n--- Step {i+1} ---")
        print(f"Action Taken: Defender={action.def_move.value}, Attacker={action.att_move.value}")
        
        obs, reward, done, info = env.step(action)
        
        print(f"Reward: {reward}")
        print(f"System Integrity: {obs.system_integrity}")
        print(f"Defender Budget: {obs.def_budget}")
        print(f"Observation Highlights: {obs.recent_attacks_log}")
        
        if done:
            print("\n[ENVIRONMENT DONE]")
            break

if __name__ == "__main__":
    run_demo()
