from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np

from src.agents.base_agent import BaseDQNAgent


def evaluate_agent(agent: BaseDQNAgent, episodes: int = 20, seed: int = 123):
    env = gym.make("MountainCar-v0")

    rewards = []
    successes = 0

    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = agent.act(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

        rewards.append(total_reward)
        successes += int(terminated)

    env.close()

    rewards = np.asarray(rewards, dtype=np.float32)

    return {
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
        "min_reward": float(rewards.min()),
        "max_reward": float(rewards.max()),
        "successes": int(successes),
        "episodes": int(episodes),
    }


def render_episode(
    agent: BaseDQNAgent,
    seed: int,
    output_dir: Path,
    name: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        "MountainCar-v0",
        render_mode="rgb_array",
    )

    state, _ = env.reset(seed=seed)

    frames = []
    terminated = False
    truncated = False

    while not (terminated or truncated):
        frames.append(env.render())
        action = agent.act(state, training=False)
        state, _, terminated, truncated, _ = env.step(action)

    frames.append(env.render())
    env.close()

    output_path = output_dir / f"{name}.mp4"
    imageio.mimsave(output_path, frames, fps=30)

    print(f"Saved video to {output_path}")
