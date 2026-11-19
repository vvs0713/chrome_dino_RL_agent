import copy
import time
from collections import deque
from typing import Any, Deque, Dict, List, Tuple

import numpy as np
import pyautogui
import torch

from utils.state_data import MakeStateData, MakeStateDataConfig
from train.memory import GlobalReplayBuffer
from train.double_dqn_agent import DoubleDQNAgent

def one_game(
    env: Any,
    agent: DoubleDQNAgent,
    memory: GlobalReplayBuffer,
    bot: Any,
    use_bot: bool,
) -> Tuple[GlobalReplayBuffer, float, float]:
    """
    Play a single game episode, collecting transitions with N-step returns.

    Args:
        env: Game environment wrapper (start, grab_image, apply_action, done_flag, refresh).
        agent: Double DQN agent used for action selection.
        memory: Global prioritized replay buffer to fill.
        bot: Teacher bot providing reference actions.
        use_bot: If True, the teacher plays the game; otherwise the agent does.

    Returns:
        memory: The same replay buffer (convenience for chaining).
        total_reward: Accumulated reward over the episode.
        game_time: Episode duration in seconds.
    """
    time.sleep(3)
    env.start()
    done: bool = False
    total_reward: float = 0.0
    data_class: MakeStateData = MakeStateData(MakeStateDataConfig())
    start_time: float = time.time()

    img: np.ndarray = copy.copy(env.grab_image())

    prev_actions: List[int] = [0 for _ in range(6)]

    res_dict: Dict[str, Any] = data_class.calc_state_variables(img, start_time)
    state_data: List[float] = [
        res_dict['barrier_distance'], res_dict['bird_distance'], res_dict['pos1'],
        res_dict['pos2'], res_dict['velocity'], res_dict['jump_time'],
        res_dict['duck_timer'], res_dict['game_time'],
    ] + prev_actions

    episode_stats: Dict[str, Any] = {'actions': [0, 0, 0, 0], 'steps': 0, 'done': []}

    # --- N-STEP RETURNS INITIALIZATION ---
    n_step: int = 10
    gamma: float = 0.999  # High gamma for a long horizon.
    n_step_buffer: Deque[Tuple[List[float], int, float, List[float], bool, int]] = deque(
        maxlen=n_step
    )
    # --------------------------------------

    k: int = 0
    while not done:
        if k == 5:
            k = 0
            # Prepare the state for NoisyNet (normalized, batched).
            state_data_normalized: np.ndarray = np.array(state_data) / agent.normalization_values
            state_data_tensor: torch.Tensor = torch.FloatTensor(state_data_normalized).view(1, -1)

            # Get an action from the network (NoisyNet generates its own noise in agent.act).
            action: int = agent.act(state_data_tensor)

            # Query the teacher bot ONLY for logging into the buffer (for supervised_loss).
            if use_bot:
                res_dict_bot: Dict[str, Any] = bot.select_action(img, start_time)
                bot_action: int = res_dict_bot['action']
                action = bot_action  # The teacher plays, guaranteeing rare rewards are reached.
            else:
                bot_action = 0
                action = agent.act(state_data_tensor)  # The NoisyNet plays, exploring the world.

            # Apply the action to the environment.
            env.apply_action(action)
            data_class.update_timers(action)

            prev_actions.append(action)
            prev_actions.pop(0)

            img = copy.copy(env.grab_image())
            done = env.done_flag

            # Reward: small positive per alive step, hard penalty on crash.
            if not done:
                reward: float = 0.1
            else:
                reward = -1.0  # Increased penalty so it outweighs the step bonuses.

            # Time limit (2 minutes).
            if (time.time() - start_time) > 120:
                done = True
                reward = 1.0

            total_reward += reward

            # Compute the next state.
            res_dict = data_class.calc_state_variables(img, start_time)
            next_state_data: List[float] = [
                res_dict['barrier_distance'], res_dict['bird_distance'], res_dict['pos1'],
                res_dict['pos2'], res_dict['velocity'], res_dict['jump_time'],
                res_dict['duck_timer'], res_dict['game_time'],
            ] + prev_actions

            # --- N-STEP RETURNS ASSEMBLY ---
            # Push the current single step into the temporary queue.
            n_step_buffer.append((state_data, action, reward, next_state_data, done, bot_action))

            # Once the queue is full (n_step steps ahead available), pack them for the ReplayBuffer.
            if len(n_step_buffer) == n_step:
                s_t, a_t, _, _, _, bot_a_t = n_step_buffer[0]  # Oldest step in the queue.

                # Accumulate the discounted reward over n_step steps.
                n_step_reward: float = 0.0
                for i in range(n_step):
                    n_step_reward += (gamma ** i) * n_step_buffer[i][2]  # index 2 is reward.
                    if n_step_buffer[i][4]:  # if the game ended inside the chain (done=True).
                        break

                # Extract the final (i-th) step parameters in the chain.
                s_next_n: List[float] = n_step_buffer[i][3]  # index 3 is next_state_data.
                done_n: bool = n_step_buffer[i][4]           # index 4 is done.

                # Push a single row linking s_t and s_next_n into the main ReplayBuffer.
                memory.push(
                    time.time() - start_time,
                    s_t, a_t, n_step_reward, s_next_n, done_n, bot_a_t,
                )

            # Move to the next step.
            state_data = next_state_data

            episode_stats['actions'][action] += 1
            episode_stats['steps'] += 1
            episode_stats['done'].append(done)
        else:
            k += 1

    # --- DRAIN THE REMAINING QUEUE (WHEN THE GAME IS OVER) ---
    # Fewer than n_step elements remain; flush them into the buffer before exiting.
    while len(n_step_buffer) > 0:
        s_t, a_t, _, _, _, bot_a_t = n_step_buffer.popleft()
        n_step_reward = 0.0
        for i in range(len(n_step_buffer) + 1):
            if i < len(n_step_buffer):
                n_step_reward += (gamma ** i) * n_step_buffer[i][2]
                s_next_n = n_step_buffer[i][3]
                done_n = n_step_buffer[i][4]
                if n_step_buffer[i][4]:
                    break
            else:
                # For the very last element, use its own data.
                n_step_reward += (gamma ** i) * reward
                s_next_n = next_state_data
                done_n = True

        memory.push(
            time.time() - start_time,
            s_t, a_t, n_step_reward, s_next_n, done_n, bot_a_t,
        )
    # ------------------------------------------------------

    game_time: float = time.time() - start_time
    pyautogui.keyUp('down', _pause=False)
    env.refresh()

    return memory, total_reward


def apply_model(
    state: List[float],
    model: torch.nn.Module,
    normalization_values: np.ndarray,
    device: torch.device,
) -> int:
    """
    Normalize the state, run it through the model, and return the greedy action.

    Args:
        state: Raw feature vector (length 14).
        model: Trained Q-network.
        normalization_values: Per-feature normalization constants.
        device: Torch device to run inference on.

    Returns:
        Selected action index.
    """
    state = np.array(state) / normalization_values

    data: torch.Tensor = torch.tensor(state).reshape(1, 1, 14)
    data = data.to(device).to(torch.float32)
    data = data.squeeze(1)
    pred: torch.Tensor = model.eval()(data)  # Shape: (1, 4) - logits for 4 actions.

    action: int = torch.argmax(pred).item()

    return action


def play_test_game(
    env: Any,
    model: torch.nn.Module,
    normalization_values: np.ndarray,
    device: torch.device,
    game_time_limit: int = 599,
) -> Tuple[int, bool]:
    """
    Run a single test episode with the trained model (or the bot as fallback).

    Args:
        env: Game environment wrapper.
        model: Trained policy network.
        bot: Teacher bot used when `time.time() - start_time < 0`.
        normalization_values: Per-feature normalization constants.
        device: Torch device to run inference on.
        game_time_limit: Maximum episode duration in seconds.

    Returns:
        score: Number of steps survived.
        game_complete: Whether the episode finished naturally (always True here).
    """
    time.sleep(1)
    env.start()
    done: bool = False
    img_buff: List[np.ndarray] = [np.zeros((170, 620, 3)) for _ in range(7)]
    img: np.ndarray = env.grab_image()
    img_buff.append(img.copy())
    score: int = 0
    start_time: float = time.time()
    actions_stats: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
    game_complete: bool = True
    prev_actions: List[int] = [0 for _ in range(6)]
    data_class: MakeStateData = MakeStateData(MakeStateDataConfig())

    # Switch to eval mode once instead of on every step.
    model.eval()

    while not done:
        res_dict: Dict[str, Any] = data_class.calc_state_variables(img, start_time)
        state_data: List[float] = [
            res_dict['barrier_distance'], res_dict['bird_distance'], res_dict['pos1'],
            res_dict['pos2'], res_dict['velocity'], res_dict['jump_time'],
            res_dict['duck_timer'], res_dict['game_time'],
        ] + prev_actions

        action = apply_model(state_data, model, normalization_values, device)

        data_class.update_timers(action)

        prev_actions.append(action)
        prev_actions.pop(0)

        env.apply_action(action)
        img = copy.copy(env.grab_image())
        img_buff.append(img.copy())
        done = env.done_flag
        score += 1
        actions_stats[action] += 1

        if time.time() - start_time > game_time_limit:
            done = True

        time.sleep(0.01)

    pyautogui.keyUp('down', _pause=False)
    env.refresh()

    return score, game_complete