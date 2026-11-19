import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """Linear layer with factorized Gaussian noise (Fortunato et al., "Noisy Networks for Exploration")."""

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5) -> None:
        super(NoisyLinear, self).__init__()

        self.in_features: int = in_features
        self.out_features: int = out_features
        self.std_init: float = std_init

        # Learnable parameters: mean values (mu)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))

        # Learnable parameters: standard deviations (sigma)
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Buffers for random noise (not updated by gradient descent)
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        """Initialize parameters as described in the original paper."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size: int) -> torch.Tensor:
        """Noise scaling function f(x) = sgn(x) * sqrt(|x|) used for factorized noise."""
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul(x.abs().sqrt())

    def reset_noise(self) -> None:
        """Generate a new noise matrix (call before each step/episode)."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        # Factorized noise: outer product of the two vectors
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # During training, add noise to weights and bias
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # During evaluation (eval), use only the mean parameter values
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)


class NoisyDQNNet(nn.Module):
    """DQN network with two input branches (first 8 features and next 6 features) and NoisyLinear layers."""

    def __init__(self, std_init: float = 0.5) -> None:
        super().__init__()

        # Replaced LazyLinear with exact sizes: x[:, :8] has 8 features
        self.input_1 = nn.Sequential(
            NoisyLinear(in_features=8, out_features=64, std_init=std_init),
            nn.LeakyReLU(),
        )

        # x[:, 8:14] has 6 features (indices 8, 9, 10, 11, 12, 13)
        self.input_2 = nn.Sequential(
            NoisyLinear(in_features=6, out_features=64, std_init=std_init),
            nn.LeakyReLU(),
        )

        # After concatenating v1 (64) and v2 (64), the model input has 128 features
        self.model = nn.Sequential(
            NoisyLinear(in_features=128, out_features=256, std_init=std_init),
            nn.LeakyReLU(),
            NoisyLinear(in_features=256, out_features=256, std_init=std_init),
            nn.LeakyReLU(),
            NoisyLinear(in_features=256, out_features=128, std_init=std_init),
            nn.LeakyReLU(),
            NoisyLinear(in_features=128, out_features=4, std_init=std_init),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Important: if x is passed without a batch dimension (i.e. a plain vector),
        # the code will fail at x[:, :8]. Make sure the input is a batch of shape [batch_size, 14].
        v1 = self.input_1(x[:, :8])
        v2 = self.input_2(x[:, 8:14])

        # Concatenate outputs along the feature dimension (dim=1)
        x_concat = torch.cat((v1, v2), dim=1)

        q_values = self.model(x_concat)
        return q_values

    def reset_noise(self) -> None:
        """
        Automatically walk all modules of the network, find NoisyLinear layers,
        and refresh their random noise matrices.
        """
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()