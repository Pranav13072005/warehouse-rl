import gymnasium as gym
import yaml
import os
import wandb
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import env
# import env  # registers WarehouseSlotting-v0
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

class WandbCallback(BaseCallback):
    """Log SB3 metrics to W&B at each rollout."""
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        if len(self.model.ep_info_buffer) > 0:
            ep_rewards = [ep["r"] for ep in self.model.ep_info_buffer]
            wandb.log({
                "rollout/ep_rew_mean": sum(ep_rewards) / len(ep_rewards),
                "rollout/n_updates": self.model.num_timesteps,
            })
        return True


def train(config_path: str = "configs/default.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    run_name = cfg["training"]["run_name"]

    # W&B init
    wandb.init(
        project="warehouse-rl",
        name=run_name,
        config=cfg,
        sync_tensorboard=True
    )

    os.makedirs(cfg["training"]["log_dir"], exist_ok=True)
    os.makedirs(cfg["training"]["model_dir"], exist_ok=True)

    # Vectorised envs (4 parallel) for faster data collection
    env_kwargs = cfg["env"]
    train_env = make_vec_env(
        "WarehouseSlotting-v0",
        n_envs=4,
        env_kwargs=env_kwargs
    )

    # Eval env (single, not vectorised)
    eval_env = Monitor(gym.make("WarehouseSlotting-v0", **env_kwargs))

    # Eval callback: saves best model automatically
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=cfg["training"]["model_dir"],
        log_path=cfg["training"]["log_dir"],
        eval_freq=cfg["training"]["eval_freq"],
        n_eval_episodes=cfg["training"]["n_eval_episodes"],
        deterministic=True,
        render=False
    )

    # PPO model
    model = PPO(
        policy=cfg["ppo"]["policy"],
        env=train_env,
        n_steps=cfg["ppo"]["n_steps"],
        batch_size=cfg["ppo"]["batch_size"],
        n_epochs=cfg["ppo"]["n_epochs"],
        learning_rate=cfg["ppo"]["learning_rate"],
        gamma=cfg["ppo"]["gamma"],
        gae_lambda=cfg["ppo"]["gae_lambda"],
        clip_range=cfg["ppo"]["clip_range"],
        ent_coef=cfg["ppo"]["ent_coef"],
        vf_coef=cfg["ppo"]["vf_coef"],
        max_grad_norm=cfg["ppo"]["max_grad_norm"],
        tensorboard_log=cfg["training"]["log_dir"],
        verbose=1
    )

    print(f"\n🚀 Starting PPO training: {run_name}")
    print(f"   Timesteps: {cfg['ppo']['total_timesteps']:,}")
    print(f"   Reward mode: {env_kwargs['reward_mode']}\n")

    model.learn(
        total_timesteps=cfg["ppo"]["total_timesteps"],
        callback=[eval_callback, WandbCallback()],
        progress_bar=True
    )

    model.save(os.path.join(cfg["training"]["model_dir"], f"{run_name}_final"))
    print(f"\n✅ Training complete. Model saved.")
    wandb.finish()


if __name__ == "__main__":
    train()