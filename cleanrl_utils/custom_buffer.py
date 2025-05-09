from typing import NamedTuple
import numpy as np
import torch
from stable_baselines3.common.buffers import ReplayBuffer, ReplayBufferSamples


class SDReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    entropies: torch.Tensor            # ← NEW


class SDReplayBuffer(ReplayBuffer):
    """
    A 3‑line extension of SB3’s ReplayBuffer that keeps one extra float per transition.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entropy_buf = np.zeros((self.buffer_size, 1), dtype=np.float32)

    # --------------------------------------------------------------------- add
    def add(self, obs, next_obs, action, reward, done, infos):
        # 'infos' is a list when using a vector env
        entropies = np.array([info["entropy"] for info in infos], dtype=np.float32).reshape(-1, 1)
        for offset in range(len(entropies)):
            self.entropy_buf[(self.pos + offset) % self.buffer_size] = entropies[offset]
        super().add(obs, next_obs, action, reward, done, infos)

    # --------------------------------------------------------------- sampling
    def sample(self, batch_size, env=None):
        # identical to ReplayBuffer.sample but we pack entropy_buf, too
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        data = self._get_samples(batch_inds, env)
        return data

    def _get_samples(self, batch_inds: np.ndarray, env=None) -> SDReplayBufferSamples:
        # Randomly pick which parallel‑env transition to use
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(
                self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env
            )
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        obs = self._normalize_obs(self.observations[batch_inds, env_indices, :], env)
        actions = self.actions[batch_inds, env_indices, :]
        dones = (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1)
        rewards = self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env)
        ents = self.entropy_buf[batch_inds, env_indices].reshape(-1, 1)

        data = (obs, actions, next_obs, dones, rewards, ents)
        return SDReplayBufferSamples(*tuple(map(self.to_torch, data)))
