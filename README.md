# warehouse-rl — Warehouse Order Picking with RL

## 1. Motivation

Modern fulfilment warehouses — particularly those operating under quick-commerce constraints (10–30 minute delivery windows) — face a fundamental operational challenge: **where should each product (SKU) be physically stored on the warehouse floor?**

This problem, known as *warehouse slotting*, directly determines how far human pickers must walk to collect items for each customer order. In a warehouse handling thousands of orders per day, even a 10% reduction in average picker travel distance translates to measurable reductions in fulfilment latency, labour cost, and energy consumption.

Existing industrial solutions are predominantly **static**: SKUs are assigned to slots once, based on historical demand frequency (ABC analysis), and reassigned only during scheduled reorganisation cycles. These approaches have two critical limitations:

1. **They ignore co-purchase correlations.** Items frequently ordered together should be co-located near the depot. A pure frequency-based approach places each SKU independently, ignoring joint demand structure.
2. **They cannot adapt to demand drift.** Consumer demand patterns shift continuously — seasonally, with promotions, and in response to market trends. A static assignment policy degrades in quality as the demand distribution changes.

This project formulates warehouse slotting as a **sequential decision-making problem** and trains a deep RL agent to discover slot assignment policies that outperform both random and frequency-based heuristics, while adapting dynamically to the demand distribution seen during an episode.

## 2) Repo layout
- `env/` — Gymnasium environment + synthetic demand generator
- `agents/` — Baselines + PPO inference wrapper
- `experiments/` — Training entry point + ablations
- `evaluate/` — Numerical benchmarking
- `visualise/` — Episode rendering (GIF)
- `configs/` — Single source of truth for hyperparams
- `notebooks/` — Analysis notebook (plots/tables placeholders)

## 3) Environment (what the agent learns)
- Grid world of size $N \times N$ with a depot at (0,0)
- Each episode has a sequence of order targets (cells)
- Actions: up/right/down/left
- When the agent reaches the target cell, the order is fulfilled automatically

Observation: `[agent_r, agent_c, target_r, target_c, remaining_orders]` normalized to `[0,1]`.

Rewards:
- Step penalty (always)
- +fulfill reward when reaching target
- Optional dense shaping: reward for reducing Manhattan distance to target

## 4) Install
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 5) Train PPO
Uses config in `configs/default.yaml`.
```bash
python -m experiments.train_ppo --config configs/default.yaml
```
Model outputs:
- `models/ppo_warehouse.zip`
- evaluation logs under `runs/ppo/`

## 6) Run ablations
Sparse vs dense reward:
```bash
python -m experiments.ablation_reward --config configs/default.yaml --total-timesteps 50000
```
5x5 vs 8x8 grid:
```bash
python -m experiments.ablation_grid --config configs/default.yaml --total-timesteps 50000
```

## 7) Benchmark all agents
```bash
python -m evaluate.benchmark --config configs/default.yaml --episodes 50 --ppo-model models/ppo_warehouse.zip
```
Writes `runs/benchmark.csv`.

## 8) Visualise an episode (GIF)
```bash
python -m visualise.render_episode --config configs/default.yaml --agent ppo --ppo-model models/ppo_warehouse.zip --out runs/episode.gif
```
(For `--agent random` or `heuristic`, PPO model isn’t required.)


