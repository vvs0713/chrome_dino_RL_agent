import numpy as np
import numpy.typing as npt
import copy
import time
import pyautogui
import torch
from dataclasses import dataclass, field
from typing import Protocol

from utils.state_data import MakeStateData, MakeStateDataConfig
from net.net import NoisyDQNNet


def _default_normalization_values() -> npt.NDArray[np.float32]:
    """Default per-feature normalization denominators (8 state features + 6 previous actions)."""
    return np.array(
        [512, 125, 560, 1120, 970, 0.6, 0.6, 120] + [4] * 6,
        dtype=np.float32,
    )

@dataclass
class GameConfig:
    """Tunable parameters for a single test game."""

    game_time_limit: float = 599.0
    start_delay: float = 1.0
    state_dim: int = 14
    prev_actions_len: int = 6
    img_buff_maxlen: int = 7
    std_check_window: int = 4
    img_shape: tuple[int, int, int] = (170, 620, 3)

    normalization_values: npt.NDArray[np.float32] = field(
        default_factory=_default_normalization_values
    )

class GameRunner:
    """Runs the Dino game loop."""

    def __init__(
        self,
        env,
        model: torch.nn.Module,
        device: str,
        verbose: str = False
    ) -> None:
        self.env = env
        self.model = model
        self.verbose = verbose
        self.config = GameConfig()
        self.device = device
        self.model.to(self.device)

    def run(self) -> dict:
        """Play one game and return the result."""
        self.state_builder = MakeStateData(MakeStateDataConfig())
        
        cfg = self.config
        time.sleep(cfg.start_delay)
        self.env.start()

        done = False

        img_buff: list[npt.NDArray[np.uint8]] = [
            np.zeros(cfg.img_shape) for _ in range(cfg.img_buff_maxlen)
        ]
        img = self.env.grab_image()
        img_buff.append(img.copy())

        score = 0
        start_time = time.time()
        prev_actions: list[int] = [0 for _ in range(cfg.prev_actions_len)]

        while not done:
            res_dict = self.state_builder.calc_state_variables(img, start_time, resolution_scale_factor=self.env.k)
            state_data = [
                res_dict['barrier_distance'],
                res_dict['bird_distance'],
                res_dict['pos1'],
                res_dict['pos2'],
                res_dict['velocity'],
                res_dict['jump_time'],
                res_dict['duck_timer'],
                res_dict['game_time'],
            ] + prev_actions
            #if res_dict['game_time'] > 3.7 and res_dict['game_time'] < 4.7:
            #    print(state_data)
            action = self._apply_model(state_data)

            self.state_builder.update_timers(action)

            prev_actions.append(action)
            prev_actions.pop(0)

            if self.verbose:
                self._print_status(res_dict, action, prev_actions)
            
            self.env.apply_action(action)
            img = copy.copy(self.env.grab_image())
            img_buff.append(img.copy())
            done = self.env.done_flag
            score += 1

            if time.time() - start_time > cfg.game_time_limit:
                done = True

            #time.sleep(0.01)

        pyautogui.keyUp('down', _pause=False)
        self.env.refresh()

        return {
            'score': score,
            'duration': time.time() - start_time
        }
    
    def _apply_model(self, state):
        normalization_values = [512, 125, 560, 1120, 970, 0.6, 0.6, 120] + [4] * 6
        state = np.array(state) / normalization_values

        data = torch.tensor(state).reshape(1, 1, self.config.state_dim)
        data = data.to(self.device).to(torch.float32)
        data = data.squeeze(1)
        with torch.no_grad():
            pred = self.model(data)

        action = torch.argmax(pred).item()

        return action

    def _print_status(self, res_dict: dict[str, float], action: int,
                      prev_actions: list[int] | None = None) -> None:
        """Print a status line from the raw state dict (more robust than positional unpacking)."""
        action_names = {0: "none", 1: "jump", 2: "duck", 3: "stop_duck"}
        action_name = action_names.get(action, f"unknown({action})")
        ps = (
            f"barrier_distance: {res_dict['barrier_distance']:>5.1f} | "
            f"bird_distance: {res_dict['bird_distance']:>5.1f} | "
            f"pos1: {res_dict['pos1']:>5.1f} | "
            f"pos2: {res_dict['pos2']:>5.1f} | "
            f"velocity: {res_dict['velocity']:>4.2f} | "
            f"jump_time: {res_dict['jump_time']:>4.2f} | "
            f"duck_timer: {res_dict['duck_timer']:>4.2f} | "
            f"game_time: {res_dict['game_time']:>6.2f} | "
            f"prev_actions: {prev_actions or []} | "
            f"action: {action_name}({action})"
        )
        print(ps, flush=True)
