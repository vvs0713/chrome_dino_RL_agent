import os
import numpy as np
import argparse
import matplotlib.pyplot as plt

from environment.environment import EnvClass
from train.double_dqn_agent import DoubleDQNAgent, TrainConfig
from train.one_game import one_game, play_test_game


# near the top of train.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(PROJECT_ROOT, "train", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Train agent to play Dino game with a NoisyDQN model.")
    parser.add_argument("-te", "--train_episodes", type=int, default=1000, help="Total number of train episodes.")
    parser.add_argument("-ts", "--train_steps", type=int, default=100, help="Train steps per one episode.")
    parser.add_argument("-us", "--taget_net_update", type=int, default=10, help="Target net update frequency.")
    parser.add_argument("-t", "--test_game_freq", type=int, default=10, help="Perform test game frequency.")
    parser.add_argument("-tc", "--use_teacher", type=bool, default=False, help="Use teacher bot.")
    return parser.parse_args()

def main() -> int:
    """CLI entry point."""
    args = parse_args()

    env = EnvClass()
    env.init_environment()


    config = TrainConfig()
    agent = DoubleDQNAgent(config)

    if args.use_teacher:
        from teacher import AlgoBot, BotGameConfig
        use_teacher = True
        teacher = AlgoBot(BotGameConfig())
    else:
        use_teacher = False
        teacher = None

    episode_rewards_history = []
    episode_loss_history = []
    test_scores = []

    for episode in range(args.train_episodes):
        agent.memory, episode_reward = one_game(env, agent, agent.memory, teacher, use_teacher)
        
        losses_in_this_episode = []
        for _ in range(args.train_steps):
            loss_val = agent.train_step()
            if loss_val is not None:
                losses_in_this_episode.append(loss_val)
            
        if episode % args.taget_net_update == 0 and episode > 0:
            agent.update_target_network()

        episode_rewards_history.append(episode_reward)
        
        if losses_in_this_episode:
            mean_loss = np.mean(losses_in_this_episode)
            episode_loss_history.append(mean_loss)
        else:
            episode_loss_history.append(0)
        
        if episode % args.test_game_freq == 0:
            test_score, _ = play_test_game(env, agent.policy_net, agent.normalization_values, agent.params.device)

            test_scores.append(test_score)
            agent.policy_net.train()
                
            if  episode > 0:
                plt.figure(figsize=(15, 4))
                
                plt.subplot(1, 3, 1)
                plt.plot(episode_rewards_history, color='green', alpha=0.6)
                # Добавим скользящее среднее для наглядности тренда
                if len(episode_rewards_history) > 60:
                    plt.plot(np.convolve(episode_rewards_history, np.ones(20)/20, mode='valid'), color='darkgreen', linewidth=2)
                plt.title('Episode reward (points)')
                plt.xlabel('Episode')
                

                plt.subplot(1, 3, 2)
                plt.plot(episode_loss_history, color='red')
                plt.title('Mean episode loss')
                plt.xlabel('Episode')

                plt.subplot(1, 3, 3)
                plt.plot(test_scores, color='blue', marker='o', markersize=2)
                plt.title('Test reward')
                plt.xlabel('Every 5 episodes')
                
                plt.tight_layout()
                plt.savefig(os.path.join(PLOTS_DIR, 'training_stats.png'))
                plt.close()

                
            # Сохраняем контрольную точку весов
            agent.save_model()
            
            # Лог в консоль для контроля сходимости
            print(f"Эпизод {episode:4d} | Очки: {episode_reward:5.1f} | Тест Q-Net: {test_scores[-1]:5.1f} | Loss: {episode_loss_history[-1]:.4f} | Роль: {'BOT' if use_teacher else 'AGENT'}")

    env.stop_environment()
    

if __name__ == "__main__":
    main()










