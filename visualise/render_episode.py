import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import gymnasium as gym
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import env
from stable_baselines3 import PPO
from env.demand_generator import DemandGenerator

def render_episode_gif(model_path="models/best_model", output="episode.gif"):
    env_instance = gym.make("WarehouseSlotting-v0")
    model = PPO.load(model_path, env=env_instance)

    obs, _ = env_instance.reset()
    raw_env = env_instance.unwrapped
    frames = []

    done = False
    step = 0
    while not done and step < 100:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_instance.step(action)
        done = terminated or truncated

        # Capture frame
        grid = raw_env.slot_contents.reshape(
            raw_env.grid_size, raw_env.grid_size
        ).copy()
        order = raw_env.current_order.copy()
        frames.append((grid.copy(), order.copy(), reward, info["episode_travel"]))
        step += 1

    # Animate
    fig, ax = plt.subplots(figsize=(6, 6))
    demand_weights = DemandGenerator(n_skus=raw_env.n_skus).get_demand_weights()

    def draw_frame(i):
        ax.clear()
        grid, order, rew, ep_travel = frames[i]
        order_skus = set(np.where(order)[0])

        for r in range(raw_env.grid_size):
            for c in range(raw_env.grid_size):
                sku = grid[r, c]
                if sku < 0:
                    color = "#f5f5f5"
                    label = ""
                elif sku in order_skus:
                    color = "#ff6b6b"   # red = in current order
                    label = str(sku)
                else:
                    # Heat by demand: popular = dark blue
                    intensity = demand_weights[sku] / demand_weights.max()
                    color = plt.cm.Blues(0.3 + 0.7 * intensity)
                    label = str(sku)

                rect = mpatches.FancyBboxPatch(
                    (c + 0.05, raw_env.grid_size - r - 0.95),
                    0.9, 0.9,
                    boxstyle="round,pad=0.05",
                    facecolor=color, edgecolor="white", linewidth=2
                )
                ax.add_patch(rect)
                if label:
                    ax.text(c + 0.5, raw_env.grid_size - r - 0.5, label,
                            ha="center", va="center", fontsize=9, fontweight="bold")

        # Depot marker
        ax.text(0.5, raw_env.grid_size - 0.5, "D",
                ha="center", va="center", fontsize=14,
                color="green", fontweight="bold")

        ax.set_xlim(0, raw_env.grid_size)
        ax.set_ylim(0, raw_env.grid_size)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Step {i+1} | Reward: {rew:.2f} | Episode travel: {ep_travel:.0f}\n"
                     f"Red = current order  |  Blue intensity = demand popularity",
                     fontsize=9)

    anim = animation.FuncAnimation(fig, draw_frame, frames=len(frames), interval=300)
    anim.save(output, writer="pillow", fps=3)
    print(f"Saved animation to {output}")
    plt.close()


if __name__ == "__main__":
    render_episode_gif()