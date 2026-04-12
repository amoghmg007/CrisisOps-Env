import matplotlib.pyplot as plt
import os

def plot_simulation(env, task_name, run_id=""):
    """
    Generates a visualization of the simulation run.
    """
    steps = range(len(env.integrity_history))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot 1: Integrity
    ax1.plot(steps, env.integrity_history, label='System Integrity', color='blue', linewidth=2, marker='o')
    ax1.set_ylabel('Integrity (0.0 - 1.0)')
    ax1.set_title(f'CrisisOps Simulation: {task_name.title()} {run_id}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)
    
    # Plot 2: Budgets
    ax2.step(steps, env.budget_history, label='Defender Budget', color='green', where='post')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Budget Units')
    plt.figtext(0.15, 0.02, "CrisisOps Dynamic Benchmark Framework", color='gray')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(f"{os.path.dirname(__file__)}/plots/{task_name}_{run_id}.png")
    plt.close()
    return f"plots/{task_name}_{run_id}.png"

def plot_combined_integrity(histories_dict: dict, save_name: str):
    """
    Plots multiple agents' integrity history on the same graph to highlight skill gaps.
    """
    os.makedirs(f"{os.path.dirname(__file__)}/plots", exist_ok=True)
    plt.figure(figsize=(10, 5))
    
    colors = {"Random": "#EF4444", "Greedy": "#F59E0B", "LLM Agent (Strategic)": "#10B981"}
    markers = {"Random": "x", "Greedy": "s", "LLM Agent (Strategic)": "o"}
    
    for agent_name, integrity_history in histories_dict.items():
        color = colors.get(agent_name, "#ffffff")
        marker = markers.get(agent_name, ".")
        plt.plot(integrity_history, label=agent_name, color=color, marker=marker, linewidth=2, alpha=0.9 if agent_name == "LLM Agent (Strategic)" else 0.6)
        
    plt.title("Agent Intelligence Comparison: System Integrity Over Time", color='white', pad=20, fontsize=14)
    plt.xlabel("Step", color='white')
    plt.ylabel("System Integrity", color='white')
    plt.ylim(-0.05, 1.05)
    
    # Styling
    ax = plt.gca()
    ax.set_facecolor('#1e1e2e')
    plt.gcf().patch.set_facecolor('#1e1e2e')
    ax.spines['bottom'].set_color('gray')
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color('gray')
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors='gray')
    plt.grid(True, linestyle='--', alpha=0.2)
    plt.legend(facecolor='#2e2e3e', edgecolor='gray', labelcolor='white')
    
    plt.tight_layout()
    plt.savefig(f"{os.path.dirname(__file__)}/plots/{save_name}.png")
    plt.close()
    return f"plots/{save_name}.png"

def plot_comparison(results_data, task_name):
    """
    Overlays multiple agent performances on the same graph to highlight discovery.
    results_data: List of dicts with {label, integrity_history}
    """
    plt.figure(figsize=(10, 6))
    
    for res in results_data:
        steps = range(len(res['integrity_history']))
        plt.plot(steps, res['integrity_history'], label=f"{res['label']} Integrity", linewidth=2)
    
    plt.axhline(y=0.0, color='black', linestyle='-')
    plt.title(f"Strategic Comparison: {task_name.title()}")
    plt.ylabel("System Integrity")
    plt.xlabel("Steps")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    filename = f"plots/COMPARISON_{task_name}.png"
    plt.savefig(filename)
    plt.close()
    return filename
