import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from net.net import NoisyLinear, NoisyDQNNet
from train.memory import GlobalReplayBuffer

@dataclass
class TrainConfig:
    action_dim: int = 4                     # размер пространства возможных действий       
    
    #SGD train parameters
    learning_rate: float = 9*1e-5           # коэффициент скорости обучения
    batch_size: int = 2028                  # размер батча
    margin = 5                              # величина штрафа
    lambda_coeff = 0.1                      # Вес учительского лосса 
    
    #RL parameters
    gamma: float = 0.995                    # Фактор дисконтирования
    epsilon: float = 1                      # Начальная вероятность случайного действия
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.9995           # Как быстро дисконтируем случайность
    replay_buffer_capicity: int = 500_000   # Размер буфера состояний
    
    if torch.cuda.is_available():
        device = torch.device('cuda:0')     #gpu
    else:
        device = torch.device('cpu')        #cpu
 
    load_weights = True                     #загружать ли предобученную модель


class DoubleDQNAgent:
    """Double DQN agent with NoisyNet, PER and optional expert supervision."""

    def __init__(self, params: Any) -> None:
        """
        Args:
            params: Hyperparameter/config object. Expected attributes:
                device, learning_rate, replay_buffer_capicity, batch_size,
                gamma, margin, lambda_coeff, per_beta (optional).
        """
        self.params: Any = params

        # Initialize TWO networks: policy and target.
        self.policy_net: NoisyDQNNet = NoisyDQNNet().to(self.params.device)
        self.target_net: NoisyDQNNet = NoisyDQNNet().to(self.params.device)

        # Load pretrained NoisyNet weights (structures match on disk and in memory).
        checkpoint: Dict[str, Any] = torch.load(
            'net/weights/model_dqn_noise.pt',
            map_location=self.params.device,
        )
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])

        # Sync weights at the start.
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target net is not trained by gradient descent.

        self.optimizer: optim.Adam = optim.Adam(
            self.policy_net.parameters(), lr=self.params.learning_rate
        )
        self.memory: GlobalReplayBuffer = GlobalReplayBuffer(
            capacity=self.params.replay_buffer_capicity
        )

        # Per-feature normalization constants.
        self.normalization_values: np.ndarray = np.array(
            [512, 125, 560, 1120, 970, 0.6, 0.6, 120] + [4 for _ in range(6)]
        )

        self.action_counts: np.ndarray = np.ones(4, dtype=np.float32)
        self.class_weights: np.ndarray = np.ones(4, dtype=np.float32)

    def freeze_unfreeze_weights(self, freeze: bool = True) -> None:
        """Freeze or unfreeze the mu parameters of the NoisyNet and rebuild the optimizer."""
        if freeze:
            set_noisy_weights_frozen(self.policy_net, freeze_mu=True)
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.policy_net.parameters()),
                lr=self.params.learning_rate,
            )
        else:
            set_noisy_weights_frozen(self.policy_net, freeze_mu=False)
            self.optimizer = torch.optim.Adam(
                self.policy_net.parameters(), lr=self.params.learning_rate
            )

    def act(self, state: torch.Tensor) -> int:
        """Select an action greedily with fresh noise in the NoisyNet."""
        state = state.squeeze(1).to(self.params.device).to(torch.float32)
        self.policy_net.reset_noise()
        with torch.no_grad():
            q_values: torch.Tensor = self.policy_net(state)
        return q_values.argmax().item()

    def train_step(self) -> Optional[float]:
        """
        One optimization step: Double DQN + supervised margin loss + PER.

        Returns:
            Total loss value, or None if the buffer does not yet hold a full batch.
        """
        if len(self.memory) < self.params.batch_size:
            return None  # Wait until enough experience is collected.

        # Importance-sampling exponent for PER.
        beta: float = getattr(self.params, 'per_beta', 0.4)

        # 1. Sample a batch (with PER metadata).
        (
            states, actions, rewards, next_states, dones, bot_actions,
            idxs, weights, time_origins, action_origins,
        ) = self.memory.sample(self.params.batch_size, beta=beta)

        # Normalize features.
        states = np.array(states) / self.normalization_values
        next_states = np.array(next_states) / self.normalization_values

        # Convert to tensors with explicit shape [batch_size, 14].
        states = torch.FloatTensor(states).to(self.params.device).view(self.params.batch_size, -1)
        next_states = torch.FloatTensor(next_states).to(self.params.device).view(self.params.batch_size, -1)

        actions = torch.LongTensor(actions).to(self.params.device).view(self.params.batch_size, 1)
        rewards = torch.FloatTensor(rewards).to(self.params.device)
        dones = torch.FloatTensor(dones).to(self.params.device)
        bot_actions = torch.LongTensor(bot_actions).to(self.params.device).view(self.params.batch_size, 1)

        # Importance-sampling weights as a tensor.
        weights = torch.FloatTensor(weights).to(self.params.device)

        # Mask: 1.0 where the agent's action matches the teacher's action.
        is_expert: torch.Tensor = (actions == bot_actions).float().squeeze(1)

        # Resample noise in both networks for this optimization step.
        self.policy_net.reset_noise()
        self.target_net.reset_noise()

        # 2. Current Q-values (shape: [batch_size, action_dim]).
        current_q_values: torch.Tensor = self.policy_net(states)

        # Q-values for the actions actually taken (shape: [batch_size]).
        current_action_q: torch.Tensor = current_q_values.gather(1, actions).squeeze(1)

        # 3. --- SUPERVISED LOSS (LARGE MARGIN) ---
        # Penalty matrix of shape [batch_size, action_dim] filled with `margin`.
        loss_margin: torch.Tensor = torch.full(
            current_q_values.size(), self.params.margin, device=self.params.device
        )
        # Zero out the penalty at the teacher's chosen action.
        loss_margin.scatter_(1, bot_actions, 0.0)

        # Upper bound: max_a [ Q(s, a) + margin ].
        q_plus_margin: torch.Tensor = current_q_values + loss_margin
        max_q_plus_margin, _ = torch.max(q_plus_margin, dim=1)

        # Raw network estimates for the teacher's actions.
        teacher_q: torch.Tensor = current_q_values.gather(1, bot_actions).squeeze(1)

        # Supervised loss, masked by `is_expert` (do not penalize the agent's own choices).
        supervised_loss_batch: torch.Tensor = max_q_plus_margin - teacher_q
        supervised_loss: torch.Tensor = torch.mean(supervised_loss_batch * is_expert)

        # 4. --- DOUBLE DQN LOGIC ---
        with torch.no_grad():
            # Step 1: Policy net selects the best action for the next state.
            next_state_actions: torch.Tensor = self.policy_net(next_states).argmax(
                dim=1, keepdim=True
            )
            # Step 2: Target net evaluates that action.
            next_q_values: torch.Tensor = self.target_net(next_states).gather(
                1, next_state_actions
            ).squeeze(1)
            # Step 3: N-step Bellman target (N=5).
            gamma_n: float = self.params.gamma ** 5
            expected_q_values: torch.Tensor = rewards + (
                gamma_n * next_q_values * (1 - dones)
            )

        # 5. Combined loss with PER importance weights.
        with torch.no_grad():
            # Raw TD errors for priority updates (no gradients).
            td_errors: np.ndarray = (
                torch.abs(current_action_q - expected_q_values).cpu().numpy().flatten()
            )

        # Per-sample SmoothL1 loss (no reduction) to apply IS weights.
        td_loss_batch: torch.Tensor = nn.SmoothL1Loss(reduction='none')(
            current_action_q, expected_q_values
        )
        td_loss: torch.Tensor = torch.mean(weights * td_loss_batch)

        # Combine Bellman loss and teacher supervision loss.
        total_loss: torch.Tensor = td_loss + self.params.lambda_coeff * supervised_loss

        # --- Optimization step ---
        self.optimizer.zero_grad()
        total_loss.backward()
        # Gradient clipping to protect the NoisyNet `mu` weights from explosion.
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        for module in self.policy_net.modules():
            if isinstance(module, NoisyLinear):
                module.weight_sigma.data.clamp_(max=0.1) # или 0.05
                module.bias_sigma.data.clamp_(max=0.1)

        # 6. --- UPDATE PRIORITIES IN THE BUFFER ---
        self.memory.update_priorities(idxs, time_origins, action_origins, td_errors)

        return total_loss.item()

    def update_target_network(self, tau: float = 0.005) -> None:
        """Hard-sync the target network with the policy network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self) -> None:
        """Save the policy network weights to disk."""
        torch.save(
            {'model_state_dict': self.policy_net.state_dict()},
            'net/weights/model_dqn_noise.pt',
        )


def set_noisy_weights_frozen(model: torch.nn.Module, freeze_mu: bool = True) -> None:
    """
    freeze_mu=True  -> Freezes the base weights (mu); only the noise (sigma) is trained.
    freeze_mu=False -> Unfreezes everything again for full training.
    """
    frozen_count: int = 0
    active_count: int = 0

    for name, param in model.named_parameters():
        if 'mu' in name:
            # Control gradients for the mean weights.
            param.requires_grad = not freeze_mu
            if freeze_mu:
                frozen_count += 1
            else:
                active_count += 1
        elif 'sigma' in name:
            # The noise must ALWAYS be trainable.
            param.requires_grad = True
            active_count += 1

    status: str = "FROZEN" if freeze_mu else "UNFROZEN"
    print(f"Base weights (mu) {status}. Active tensors: {active_count}, frozen: {frozen_count}")