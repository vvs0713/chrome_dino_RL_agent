import random
from typing import Any, Dict, List, Tuple

import numpy as np


class SumTree:
    """Binary sum tree for fast priority sampling in O(log N)."""

    def __init__(self, capacity: int) -> None:
        """
        Args:
            capacity: Maximum number of leaves (stored transitions).
        """
        self.capacity: int = capacity
        # Stores priorities. Size: leaves + internal nodes.
        self.tree: np.ndarray = np.zeros(2 * capacity - 1)
        # Stores the actual data (transition tuples).
        self.data: np.ndarray = np.zeros(capacity, dtype=object)
        self.write: int = 0
        self.n_entries: int = 0

    def _propagate(self, idx: int, change: float) -> None:
        """Propagate a priority change from a leaf up to the root."""
        parent: int = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """Find the leaf index whose cumulative priority range contains `s`."""
        left: int = 2 * idx + 1
        right: int = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        """Return the sum of all priorities (root value)."""
        return self.tree[0]

    def add(self, p: float, data: Any) -> None:
        """Add a new transition with priority `p` to the next write position."""
        idx: int = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx: int, p: float) -> None:
        """Update the priority of the leaf at `idx` and propagate the change."""
        change: float = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, Any]:
        """
        Return (tree_index, priority, data) for the leaf containing `s`.
        """
        idx: int = self._retrieve(0, s)
        data_idx: int = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    def __len__(self) -> int:
        return self.n_entries


class SinglePrioritizedBuffer:
    """Prioritized Experience Replay (PER) buffer for a single action."""

    def __init__(self, capacity: int, alpha: float = 0.6) -> None:
        """
        Args:
            capacity: Maximum number of transitions.
            alpha: Priority exponent (0 = uniform, 1 = full prioritization).
        """
        self.tree: SumTree = SumTree(capacity)
        self.alpha: float = alpha
        self.epsilon: float = 0.01

    def push(self, error: float, data: Any) -> None:
        """Add a transition with priority derived from its TD error."""
        # New transitions receive maximum priority.
        p: float = (np.abs(error) + self.epsilon) ** self.alpha
        self.tree.add(p, data)

    def sample(self, n: int, beta: float = 0.4) -> Tuple[List[Any], List[int], np.ndarray]:
        """
        Sample `n` transitions using stratified sampling over priority segments.

        Args:
            n: Number of transitions to sample.
            beta: Importance-sampling exponent (annealed towards 1 over training).

        Returns:
            batch: List of transitions.
            idxs: List of tree indices (for later priority updates).
            is_weights: Normalized importance-sampling weights.
        """
        batch: List[Any] = []
        idxs: List[int] = []
        priorities: List[float] = []

        segment: float = self.tree.total() / n

        for i in range(n):
            a: float = segment * i
            b: float = segment * (i + 1)
            s: float = random.uniform(a, b)
            idx, p, data = self.tree.get(s)

            # Guard against empty cells when the buffer is not yet full.
            if data is None or (isinstance(data, float) and data == 0.0):
                # Fall back to a random already-filled entry.
                valid_idx: int = random.randint(0, self.tree.n_entries - 1)
                idx = valid_idx + self.tree.capacity - 1
                p = self.tree.tree[idx]
                data = self.tree.data[valid_idx]

            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        sampling_probabilities: np.ndarray = np.array(priorities) / (self.tree.total() + 1e-8)
        is_weights: np.ndarray = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        is_weights /= (is_weights.max() + 1e-8)

        return batch, idxs, is_weights

    def update_priorities(self, idxs: List[int], errors: List[float]) -> None:
        """Update priorities of sampled transitions after a gradient step."""
        for idx, error in zip(idxs, errors):
            p: float = (np.abs(error) + self.epsilon) ** self.alpha
            self.tree.update(idx, p)

    def __len__(self) -> int:
        return len(self.tree)


class ReplayBuffer:
    """Global buffer distributing batches across actions with PER support."""

    def __init__(self, capacity: int, alpha: float = 0.6) -> None:
        """
        Args:
            capacity: Total capacity, split evenly across the 4 action buffers.
            alpha: Priority exponent for PER.
        """
        sub_capacity: int = capacity // 4
        self.buffers: Dict[int, SinglePrioritizedBuffer] = {
            0: SinglePrioritizedBuffer(sub_capacity, alpha),
            1: SinglePrioritizedBuffer(sub_capacity, alpha),
            2: SinglePrioritizedBuffer(sub_capacity, alpha),
            3: SinglePrioritizedBuffer(sub_capacity, alpha),
        }

    def push(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
        bot_action: int,
        error: float = 1.0,
    ) -> None:
        """Store a transition in the buffer corresponding to `action`."""
        data: Tuple[Any, int, float, Any, bool, int] = (
            state, action, reward, next_state, done, bot_action
        )
        if action in self.buffers:
            self.buffers[action].push(error, data)

    def sample(
        self, batch_size: int, beta: float = 0.4
    ) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, List[int], np.ndarray, List[int]
    ]:
        """
        Sample a batch balanced across the 4 action buffers.

        Returns:
            states, actions, rewards, next_states, dones, bot_actions,
            idxs (tree indices), is_weights, actions_origin (buffer ids).
        """
        total_state: List[Any] = []
        total_action: List[int] = []
        total_reward: List[float] = []
        total_next_state: List[Any] = []
        total_done: List[bool] = []
        total_bot_action: List[int] = []

        all_idxs: List[int] = []
        all_weights: List[float] = []
        all_actions_origin: List[int] = []  # Track which action buffer each sample came from.

        # Compute per-action quotas.
        target_num: int = batch_size // 4
        collected: int = 0

        for act in [0, 1, 2, 3]:
            buf: SinglePrioritizedBuffer = self.buffers[act]
            n_samples: int = min(len(buf), target_num)
            if n_samples > 0:
                batch, idxs, weights = buf.sample(n_samples, beta)

                state, action, reward, next_state, done, bot_action = zip(*batch)
                total_state.extend(state)
                total_action.extend(action)
                total_reward.extend(reward)
                total_next_state.extend(next_state)
                total_done.extend(done)
                total_bot_action.extend(bot_action)

                all_idxs.extend(idxs)
                all_weights.extend(weights)
                all_actions_origin.extend([act] * n_samples)
                collected += n_samples

        # Assign the remainder to action 0 (no_action).
        no_action_num: int = batch_size - collected
        buf_zero: SinglePrioritizedBuffer = self.buffers[0]

        # Guard for early training when buffer 0 may still be empty.
        n_zero: int = min(len(buf_zero), no_action_num)
        if n_zero > 0:
            batch, idxs, weights = buf_zero.sample(n_zero, beta)
            state, action, reward, next_state, done, bot_action = zip(*batch)
            total_state.extend(state)
            total_action.extend(action)
            total_reward.extend(reward)
            total_next_state.extend(next_state)
            total_done.extend(done)
            total_bot_action.extend(bot_action)

            all_idxs.extend(idxs)
            all_weights.extend(weights)
            all_actions_origin.extend([0] * n_zero)

        return (
            np.array(total_state),
            np.array(total_action),
            np.array(total_reward, dtype=np.float32),
            np.array(total_next_state),
            np.array(total_done, dtype=np.uint8),
            np.array(total_bot_action),
            all_idxs,
            np.array(all_weights, dtype=np.float32),
            all_actions_origin,
        )

    def update_priorities(
        self, idxs: List[int], all_actions_origin: List[int], errors: List[float]
    ) -> None:
        """Update priorities after a gradient step, routing each sample to its buffer."""
        for idx, act, error in zip(idxs, all_actions_origin, errors):
            self.buffers[act].update_priorities([idx], [error])

    def __len__(self) -> int:
        return sum(len(self.buffers[i]) for i in range(4))


class GlobalReplayBuffer:
    """Global buffer distributing batches across 4 game-time intervals with PER support."""

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        time_limits: List[int] = [30, 60, 90, 120],
    ) -> None:
        """
        Args:
            capacity: Total capacity, split evenly across the 4 time buffers.
            alpha: Priority exponent for PER.
            time_limits: Upper time bounds (in seconds) for each of the 4 intervals.
        """
        # self.buffers holds 4 temporal blocks. Each block is a ReplayBuffer
        # that internally splits its capacity across the 4 actions.
        sub_capacity: int = capacity // 4
        self.buffers: Dict[int, ReplayBuffer] = {
            0: ReplayBuffer(sub_capacity, alpha),
            1: ReplayBuffer(sub_capacity, alpha),
            2: ReplayBuffer(sub_capacity, alpha),
            3: ReplayBuffer(sub_capacity, alpha),
        }
        self.time_limits: List[int] = time_limits

    def push(
        self,
        game_time: float,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
        bot_action: int,
        error: float = 1.0,
    ) -> None:
        """Store a transition in the temporal buffer matching `game_time`."""
        buf_num: int = 3  # Default interval if game_time > last limit.
        for n, t in enumerate(self.time_limits):
            if game_time <= t:
                buf_num = n
                break

        # Dispatch to the temporal buffer; it will route by action internally.
        self.buffers[buf_num].push(state, action, reward, next_state, done, bot_action, error)

    def sample(
        self, batch_size: int, beta: float = 0.4
    ) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, List[int], np.ndarray, List[int], List[int]
    ]:
        """
        Sample a batch balanced across the 4 temporal buffers.

        Returns:
            states, actions, rewards, next_states, dones, bot_actions,
            idxs, is_weights, time_origins, action_origins.
        """
        total_state: List[Any] = []
        total_action: List[int] = []
        total_reward: List[float] = []
        total_next_state: List[Any] = []
        total_done: List[bool] = []
        total_bot_action: List[int] = []

        all_idxs: List[int] = []
        all_weights: List[float] = []

        # Two trackers are needed to reverse-route priority updates:
        # 1. Which temporal buffer the sample came from (time_block).
        # 2. Which action buffer inside it the sample came from (action_origin).
        all_time_origins: List[int] = []
        all_action_origins: List[int] = []

        # Split the batch evenly across the 4 temporal intervals.
        target_per_time: int = batch_size // 4
        collected_total: int = 0

        for time_idx in range(4):
            buf_time: ReplayBuffer = self.buffers[time_idx]

            # Skip empty temporal buffers (we will backfill from filled ones).
            if len(buf_time) == 0:
                continue

            n_samples: int = min(len(buf_time), target_per_time)
            if n_samples > 0:
                (
                    res_s, res_a, res_r, res_ns, res_d, res_ba,
                    idxs, weights, act_origins,
                ) = buf_time.sample(n_samples, beta)

                total_state.extend(res_s)
                total_action.extend(res_a)
                total_reward.extend(res_r)
                total_next_state.extend(res_ns)
                total_done.extend(res_d)
                total_bot_action.extend(res_ba)

                all_idxs.extend(idxs)
                all_weights.extend(weights)

                all_time_origins.extend([time_idx] * n_samples)
                all_action_origins.extend(act_origins)
                collected_total += n_samples

        # Backfill if the batch is not fully filled (e.g., some intervals are empty).
        remainder: int = batch_size - collected_total
        if remainder > 0:
            for time_idx in range(4):
                buf_time = self.buffers[time_idx]
                if len(buf_time) >= remainder:
                    (
                        res_s, res_a, res_r, res_ns, res_d, res_ba,
                        idxs, weights, act_origins,
                    ) = buf_time.sample(remainder, beta)

                    total_state.extend(res_s)
                    total_action.extend(res_a)
                    total_reward.extend(res_r)
                    total_next_state.extend(res_ns)
                    total_done.extend(res_d)
                    total_bot_action.extend(res_ba)

                    all_idxs.extend(idxs)
                    all_weights.extend(weights)
                    all_time_origins.extend([time_idx] * remainder)
                    all_action_origins.extend(act_origins)
                    break

        return (
            np.array(total_state),
            np.array(total_action),
            np.array(total_reward, dtype=np.float32),
            np.array(total_next_state),
            np.array(total_done, dtype=np.uint8),
            np.array(total_bot_action),
            all_idxs,
            np.array(all_weights, dtype=np.float32),
            all_time_origins,
            all_action_origins,
        )

    def update_priorities(
        self,
        idxs: List[int],
        all_time_origins: List[int],
        all_action_origins: List[int],
        errors: List[float],
    ) -> None:
        """Update priorities in the specific temporal buffer and action buffer."""
        for idx, time_idx, act, error in zip(idxs, all_time_origins, all_action_origins, errors):
            # Drill into the temporal block, then into the action-specific updater.
            self.buffers[time_idx].update_priorities([idx], [act], [error])

    def __len__(self) -> int:
        # Global length is the sum of all elements across all temporal intervals.
        return sum(len(self.buffers[i]) for i in range(4))