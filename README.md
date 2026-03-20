# warehouse-rl — Warehouse Order Picking with RL

## 1) What this is (TL;DR)
A small, self-contained RL project: a custom Gymnasium grid environment that simulates warehouse order picking, plus baselines (random + frequency heuristic) and PPO training/evaluation scripts.

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

## 9) How to talk about this on a resume (numbers to fill)
Replace brackets with your results from `runs/benchmark.csv` and ablation logs.
- PPO improves average episode reward by **[X%]** over random and **[Y%]** over heuristic.
- PPO achieves **[S%]** success rate (all orders fulfilled) within **[T]** max steps.
- Dense reward shaping reduces time-to-threshold by **[R%]** vs sparse reward.
- Scaling from 5x5 → 8x8 reduces performance by **[Δ]**, motivating curriculum/transfer.

## 10) Repro tips
- Keep `seed` fixed in `configs/default.yaml`
- Save models and resolved env config under `runs/`

