import json
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from src.agents.base_agent import BaseDQNAgent


def shaped_reward(reward, state, new_state, gamma):
    modified_reward = reward + 300 * (
        gamma * abs(new_state[1]) - abs(state[1])
    )
    return modified_reward


def train(
    agent: BaseDQNAgent,
    episodes: int,
    seed: int,
    results_dir: Path,
    name: str
):
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make("MountainCar-v0")

    rewards_history = []
    losses_history = []
    epsilon_history = []

    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)

        episode_reward = 0.0
        episode_losses = []

        done = False

        while not done:
            action = agent.act(state, training=True)

            new_state, reward, terminated, truncated, _ = env.step(action)

            reward = shaped_reward(
                reward,
                state,
                new_state,
                agent.gamma,
            )

            done = terminated or truncated

            agent.remember(
                state,
                action,
                reward,
                new_state,
                done,
            )

            loss = agent.learn()
            if loss is not None:
                episode_losses.append(loss)

            agent.step_epsilon()

            state = new_state
            episode_reward += reward

        rewards_history.append(float(episode_reward))
        losses_history.append(
            float(np.mean(episode_losses))
            if episode_losses
            else np.nan
        )
        epsilon_history.append(float(agent.epsilon))

        if (episode + 1) % 25 == 0:
            print(
                f"Episode {episode + 1:4d} | "
                f"Reward {episode_reward:8.2f} | "
                f"Epsilon {agent.epsilon:.4f}"
            )

    env.close()

    agent.save(results_dir / f"{name}_model.pt")

    with open(
        results_dir / f"{name}_training_history.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "rewards": rewards_history,
                "losses": losses_history,
                "epsilon": epsilon_history,
            },
            f,
            indent=2,
        )

    _plot_history(
        rewards_history,
        losses_history,
        epsilon_history,
        plots_dir,
        name
    )


def _plot_history(rewards, losses, epsilon, plots_dir: Path, name: str):
    episodes = np.arange(1, len(rewards) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards)
    plt.title("Episode rewards")
    plt.xlabel("Episode")
    plt.ylabel("Modified reward")
    plt.tight_layout()
    plt.savefig(plots_dir / f"{name}_rewards.png")
    plt.close()

    moving = np.convolve(
        rewards,
        np.ones(20) / 20,
        mode="valid",
    )

    plt.figure(figsize=(10, 5))
    plt.plot(
        np.arange(20, len(rewards) + 1),
        moving,
    )
    plt.title("20-episode moving average")
    plt.xlabel("Episode")
    plt.ylabel("Average modified reward")
    plt.tight_layout()
    plt.savefig(plots_dir / f"{name}_moving_average.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, losses)
    plt.title("Training loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(plots_dir / f"{name}_loss.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, epsilon)
    plt.title("Epsilon")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.tight_layout()
    plt.savefig(plots_dir / f"{name}_epsilon.png")
    plt.close()
