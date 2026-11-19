import sys
import numpy as np
import torch
import argparse
from pathlib import Path

from environment.environment import EnvClass
from utils.game_runner import GameRunner
from net.net import NoisyDQNNet


def parse_args():
    parser = argparse.ArgumentParser(description="Run a Dino test game with a trained NoisyDQN model.")
    parser.add_argument("-gt", "--game_time", type=float, default=599.0, help="Game time limit in seconds (default: 599).")
    parser.add_argument("-ng", "--num_games", type=int, default=10, help="Number of consecutive games to play (default: 10).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Number of consecutive games to play (default: 10).")
    return parser.parse_args()

def main() -> int:
    """CLI entry point."""
    args = parse_args()

    env = EnvClass()
    env.init_environment()

    if torch.cuda.is_available():
        device = torch.device('cuda:0')     #gpu
    else:
        device = torch.device('cpu')        #cpu
    
    policy_net = NoisyDQNNet()
    checkpoint = torch.load('net/weights/model_dqn_noise.pt')
    #checkpoint = torch.load('/media/mainuser/DATA/RL/DinoGame/weights/model_dqn_noise_N_epoch.pt')
 
    policy_net.load_state_dict(checkpoint['model_state_dict'])
    policy_net.eval()


    runner = GameRunner(env, policy_net, device, args.verbose)
    scores = []
    games_duration = []
    for i in range(args.num_games):
        game_result = runner.run()
        scores.append(game_result['score'])
        games_duration.append(game_result['duration'])


    print(
        f"games={args.num_games} "
        f"score_mean={np.mean(scores):.1f} "
        f"score_max={max(scores)} "
        f"score_min={min(scores)} "
        f"game_time_mean={np.mean(games_duration):.1f}"
    )


if __name__ == "__main__":
    main()
