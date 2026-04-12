---
title: CrisisOps Env Validation
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_file: inference.py
pinned: false
---

# 🚀 CrisisOps: Multi-Agent Strategic Threat Environment

# CrisisOps

> **CrisisOps transforms cybersecurity into a deterministic strategy game where intelligence is measurable, not assumed.**

**Game Loop Overview:**
`Each step: threat increases → agent chooses action → system updates → score accumulates.`

CrisisOps is a multi-agent cyber-defense environment designed exactly for the **"Agentic AI: Gaming & Multi-Agent"** track. It serves as a rigorous testing ground measuring how efficiently agents gather information, assign resources, and stave off compounding disaster in real-time.

## 🎯 Why This is a Good Benchmark
This environment clearly demonstrates strategic differentiation `Random < Greedy < LLM Agent` across all tasks.
This environment is a strong benchmark because it:
- Differentiates agent intelligence (Random < Greedy < LLM)
- Enforces trade-offs between information, defense, and recovery
- Operates under partial observability and resource constraints
- Uses deterministic evaluation for reproducibility

All environment transitions and evaluations are fully deterministic to ensure reproducible and fair benchmarking across agents.

## ⚙️ Core Mechanics
```text
+-------------------+
|   Blue Team (LLM) |
|   (budget, ops)   |
+---------+---------+
          | Action (Scan, Defend, Allocate)
          v
+---------+---------+
|    CrisisOpsEnv   | <--- [Deterministic Attacker Logic]
| (system_integrity,|
|  visible_threat,  |
|  budget)          |
+---------+---------+
          | Observation + Reward
          v
+---------+---------+
```

### ⚙️ Environment Dynamics
1. **Partial Observability**: The threat level of the attacker is completely hidden. The agent only receives `visible_threat_level` if their immediate `last_action` was `scan`.
2. **Resource-Constrained Decision-Making**: Every action the agent takes requires `budget`.
   - `scan` (Cost: 1) - **Tradeoff**: Offers information, but provides no physical protection from attacks.
   - `defend` (Cost: 1) - **Tradeoff**: Offers safety (mitigates attack impact), but does not recover missing integrity.
   - `allocate` (Cost: 2) - **Tradeoff**: Recovers system integrity natively, at a high budget cost without defense.
3. **Deterministic Attacker**: The environmental threat grows deterministically via set rules (`if step % 2 == 0...`). Damage scales on this internal threat loop mechanically, ensuring entirely fair deterministic capabilities for grading.

## 🔁 3 Structurally Identical Evaluation Tracks
All tasks use identical internal states (`integrity`, `threat`, `budget`) and actions. Only the optimization objectives change.

### Expected Agent Behavior
In a well-performing agent:
- **Recon:** frequent scans improve threat awareness
- **Defense:** balanced defense prevents collapse
- **Recovery:** timely allocation restores system integrity

1. **Recon Phase:** maximize threat awareness efficiently
2. **Defense Phase:** maintain system stability under pressure
3. **Recovery Phase:** restore integrity after degradation

## 🚀 Running the Benchmarkics
Each task uses identical, highly deterministic grading structures: `Score = A * metric_1 + B * metric_2`.

- **Recon Grading**: `(0.7 * avg_visible_threat) + (0.3 * scan_ratio)`
- **Defense Grading**: `(0.6 * avg_integrity) + (0.4 * terminal_integrity)`
- **Recovery Grading**: `(0.7 * recovery_delta) + (0.3 * terminal_integrity)`

## ▶️ Running Benchmarks locally
```bash
# Run deterministic tracking benchmarks over the 3 tasks
python inference.py

# Run standalone open-env production server
uvicorn server.app:app --host 0.0.0.0 --port 7860
```