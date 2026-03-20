import numpy as np
import gymnasium as gym
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import env
from stable_baselines3 import PPO
from agents.random_agent import RandomAgent
from agents.heuristic_agent import FrequencyHeuristicAgent

def evaluate_agent(agent, env_instance, n_episodes=50):
    """Run agent for n_episodes, return mean ± std of episode travel."""
    travels = []
    for _ in range(n_episodes):
        obs, _ = env_instance.reset()
        done = False
        while not done:
            action, _ = agent.predict(obs)
            obs, _, terminated, truncated, info = env_instance.step(action)
            done = terminated or truncated
        travels.append(info["episode_travel"])
    return np.mean(travels), np.std(travels)


def run_benchmark(model_path: str = "models/best_model"):
    env_instance = gym.make("WarehouseSlotting-v0")

    # Load trained PPO
    ppo_agent = PPO.load(model_path, env=env_instance)

    random_agent = RandomAgent(env_instance.action_space)
    heuristic_agent = FrequencyHeuristicAgent(env_instance.unwrapped)

    print("\n" + "="*50)
    print("BENCHMARK RESULTS (50 episodes each)")
    print("="*50)
    print(f"{'Agent':<20} {'Mean Travel':>12} {'Std Dev':>10} {'vs Random':>10}")
    print("-"*54)

    results = {}
    for name, agent in [
        ("Random", random_agent),
        ("Frequency Heuristic", heuristic_agent),
        ("PPO (ours)", ppo_agent)
    ]:
        mean, std = evaluate_agent(agent, env_instance)
        results[name] = mean
        print(f"{name:<20} {mean:>12.1f} {std:>10.1f}")

    # Print improvement percentages
    random_baseline = results["Random"]
    print("-"*54)
    for name in ["Frequency Heuristic", "PPO (ours)"]:
        pct = (random_baseline - results[name]) / random_baseline * 100
        print(f"{name} reduces travel by {pct:.1f}% vs Random")

    pct_vs_heuristic = (results["Frequency Heuristic"] - results["PPO (ours)"]) / \
                       results["Frequency Heuristic"] * 100
    print(f"PPO outperforms Heuristic by {pct_vs_heuristic:.1f}%")
    print("="*50)


if __name__ == "__main__":
    run_benchmark()