import argparse
from pathlib import Path

from src.agents.base_agent import BaseDQNAgent
from src.agents import DoubleDQNAgent, PERDQNAgent, NStepDQNAgent, DuelingDQNAgent, NoisyNetDQNAgent, C51DQNAgent, RainbowDQNAgent
from src.evaluate import evaluate_agent, render_episode
from src.train import train
import random
import torch
import numpy as np

AGENTS = {"double_dqn": DoubleDQNAgent, "per": PERDQNAgent, "nstep_dqn": NStepDQNAgent, "dueling_dqn": DuelingDQNAgent, "noisy_net": NoisyNetDQNAgent, "c51": C51DQNAgent, "rainbow_dqn": RainbowDQNAgent}


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=AGENTS, default="double_dqn")
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "render"],
        default="train",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="default",
    )
    return parser

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def make_agent(agent_name="double_dqn") -> BaseDQNAgent:
    state_dim = 2
    action_dim = 3
    agent = AGENTS[agent_name](state_dim=state_dim, action_dim=action_dim)
    return agent



def main():
    parser = build_parser()
    args = parser.parse_args()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    agent = make_agent(args.agent)

    if args.mode == "train":
        train(
            agent=agent,
            episodes=300,
            seed=42,
            results_dir=results_dir,
            name=args.name,
        )
    elif args.mode == "evaluate":
        model_path = results_dir / f"{args.name}_model.pt"
        agent.load(model_path)
        metrics = evaluate_agent(agent, episodes=20, seed=123)
        print(metrics)
    elif args.mode == "render":
        model_path = results_dir / f"{args.name}_model.pt"
        agent.load(model_path)
        render_episode(agent, seed=123, output_dir=results_dir / "videos", name=args.name)


if __name__ == "__main__":
    main()
