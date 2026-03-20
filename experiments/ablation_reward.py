import yaml
import copy
import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import env
from experiments.train_ppo import train

def run_reward_ablation():
    with open("configs/default.yaml") as f:
        base_cfg = yaml.safe_load(f)

    # Experiment A: dense reward
    cfg_dense = copy.deepcopy(base_cfg)
    cfg_dense["env"]["reward_mode"] = "dense"
    cfg_dense["training"]["run_name"] = "ppo_dense"
    with open("configs/ablation_dense.yaml", "w") as f:
        yaml.dump(cfg_dense, f)

    # Experiment B: sparse reward
    cfg_sparse = copy.deepcopy(base_cfg)
    cfg_sparse["env"]["reward_mode"] = "sparse"
    cfg_sparse["training"]["run_name"] = "ppo_sparse"
    with open("configs/ablation_sparse.yaml", "w") as f:
        yaml.dump(cfg_sparse, f)

    print("Running DENSE reward experiment...")
    train("configs/ablation_dense.yaml")

    print("Running SPARSE reward experiment...")
    train("configs/ablation_sparse.yaml")

if __name__ == "__main__":
    run_reward_ablation()