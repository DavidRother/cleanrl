#!/usr/bin/env python3
"""
This script evaluates and visualizes the reward surface of a PPO Actor network in the
MinAtar/SpaceInvaders-v1 environment. It perturbs the actor’s parameters along two
independently sampled filter-normalized random directions and computes the empirical
expected episodic return at each perturbation. The methodology is adapted from the techniques
in Li et al. (2018) as used in "Cliff Diving: Exploring Reward Surfaces in Reinforcement Learning Environments".

Usage:
    python evaluate_reward_surface.py --model_path /path/to/actor_checkpoint.pt
                                      [--episodes 5]
                                      [--grid_size 21]
                                      [--max_alpha 1.0]
"""

import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import gymnasium as gym
import matplotlib.pyplot as plt
import tqdm
import os

from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

# Set up device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------
# Helpers and Wrappers
# -------------------------------

def layer_init(layer, bias_const=0.0):
    """Initialize layer with Kaiming normal and constant bias."""
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ChannelFirstWrapper(gym.ObservationWrapper):
    """
    Gym wrapper to transpose observations from (H, W, C) to (C, H, W).
    """

    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        assert len(obs_shape) == 3, "Expected 3D observations (H, W, C)"
        c, h, w = obs_shape[2], obs_shape[0], obs_shape[1]
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(c, h, w), dtype=self.observation_space.dtype
        )

    def observation(self, observation):
        return np.transpose(observation, (2, 0, 1))


# -------------------------------
# Actor Network (Convolutional)
# -------------------------------
class Actor(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_shape = envs.single_observation_space.shape  # Expect shape (C, H, W)
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[0], 16, kernel_size=3, stride=1)),
            nn.Flatten(),
        )
        # Determine conv output dimension by a dummy forward pass.
        with torch.inference_mode():
            output_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]

        self.fc1 = layer_init(nn.Linear(output_dim, 128))
        self.fc_logits = layer_init(nn.Linear(128, envs.single_action_space.n))

    def forward(self, x):
        # Ensure x is a float tensor with the expected device
        # If input is a single observation without batch dimension, unsqueeze it.
        x = torch.as_tensor(x, dtype=torch.float32, device=device)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = F.relu(self.conv(x))
        x = F.relu(self.fc1(x))
        logits = self.fc_logits(x)
        return logits

    def get_action(self, x):
        logits = self(x)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        # Action probabilities and log probabilities might be useful for other metrics.
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        return action, log_prob, action_probs


# -------------------------------
# Filter-Normalized Random Directions
# -------------------------------
def get_random_directions(model):
    """
    Generate a dictionary of random perturbation directions with the same shape as the model parameters.
    """
    directions = {}
    for name, param in model.named_parameters():
        directions[name] = torch.randn_like(param)
    return directions


def normalize_direction(direction):
    """
    For each parameter tensor in the direction dictionary, apply normalization.

    For convolutional layers (tensor with >= 3 dimensions), normalize each filter (first dimension) independently.
    For other parameters, normalize the tensor globally.
    """
    epsilon = 1e-10
    normalized_direction = {}
    for name, tensor in direction.items():
        if tensor.ndim >= 3:  # Assume convolutional filters
            new_tensor = tensor.clone()
            # Normalize each filter (iterate over first dimension)
            for i in range(new_tensor.shape[0]):
                norm = new_tensor[i].norm()
                new_tensor[i] = new_tensor[i] / (norm + epsilon)
            normalized_direction[name] = new_tensor
        else:
            norm = tensor.norm()
            normalized_direction[name] = tensor / (norm + epsilon)
    return normalized_direction


def perturb_model(base_model, direction1, direction2, alpha, beta):
    """
    Create a perturbed model:
        new_params = original_params + alpha * direction1 + beta * direction2
    """
    perturbed_model = copy.deepcopy(base_model)
    for name, param in perturbed_model.named_parameters():
        param.data.add_(alpha * direction1[name] + beta * direction2[name])
    return perturbed_model


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = ChannelFirstWrapper(env)
        env = ClipRewardEnv(env)

        env.action_space.seed(seed)
        return env

    return thunk


# -------------------------------
# Reward Evaluation
# -------------------------------
def gather_dataset(model, env, num_episodes=5):
    """
    Evaluate the average episodic return of the given model.
    Uses the Gymnasium API: reset() returns (obs, info) and step() returns
    (obs, reward, terminated, truncated, info).
    """
    total_return = 0.0
    dataset = []
    obs, _ = env.reset()
    num_done = 0
    while not num_done == num_episodes:
        # Get action. Actor expects batched input.
        actions, _, _ = model.get_action(torch.Tensor(obs).to(device))
        actions = actions.detach().cpu().numpy()
        dataset.append((obs, actions))
        obs, rewards, terminations, truncations, infos = env.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                num_done += 1
    return dataset


def compute_loss_on_dataset(model, dataset):
    """
    Compute the surrogate loss on the fixed dataset.
    Here the loss is defined as the negative log-likelihood of the stored actions.
    """
    losses = []
    for obs, action in dataset:
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
        if obs_tensor.ndim == 3:
            obs_tensor = obs_tensor.unsqueeze(0)
        logits = model(obs_tensor)
        log_probs = F.log_softmax(logits, dim=1)
        loss = -log_probs[0, action]
        losses.append(loss.item())
    return np.mean(losses)


# -------------------------------
# Main Script to Visualize the Reward Surface
# -------------------------------
def main():

    model_path = "../runs_old/MinAtar/SpaceInvaders-v1__sac_min_atar_max_alpha_multi_run/final_model_seed_123456.pt"

    episodes = 10

    grid_size = 17

    max_alpha = 1.0

    env_id = "MinAtar/SpaceInvaders-v1"

    current_seed = 1

    run_name = "perturbation"

    # Create the environment. Ensure observations come as (C, H, W) by using the wrapper.

    # To construct our Actor we need an object with attributes 'single_observation_space' and 'single_action_space'.
    # We create a dummy container for that.
    envs = gym.vector.SyncVectorEnv([make_env(env_id, current_seed, 0, False, run_name)])

    # Instantiate the Actor and load the checkpoint.
    base_model = Actor(envs).to(device)
    state_dict = torch.load(model_path, map_location=device)
    base_model.load_state_dict(state_dict)
    base_model.eval()

    # Generate two independently sampled random directions and apply filter normalization.
    direction1 = normalize_direction(get_random_directions(base_model))
    direction2 = normalize_direction(get_random_directions(base_model))

    # Build a grid of perturbations.
    grid_size = grid_size
    alphas = np.linspace(-max_alpha, max_alpha, grid_size)
    betas = np.linspace(-max_alpha, max_alpha, grid_size)
    reward_grid = np.zeros((grid_size, grid_size))

    print("Evaluating reward surface over perturbation grid ...")
    for i, a in enumerate(tqdm.tqdm(alphas, desc="Alpha perturbation")):
        for j, b in enumerate(betas):
            perturbed_model = perturb_model(base_model, direction1, direction2, a, b)
            avg_reward = evaluate_reward(perturbed_model, envs, num_episodes=episodes)
            reward_grid[i, j] = avg_reward

    # Plot the reward surface.
    X, Y = np.meshgrid(betas, alphas)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, reward_grid, cmap='viridis', edgecolor='none', alpha=0.9)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    ax.set_title("3D Reward Surface for SAC Actor\n(MinAtar/SpaceInvaders-v1)")
    ax.set_xlabel("Beta Perturbation")
    ax.set_ylabel("Alpha Perturbation")
    ax.set_zlabel("Average Reward")
    plt.tight_layout()

    model_dir = os.path.dirname(model_path)
    save_filename = f"reward_surface_data_{model_path}.npz"
    save_path = os.path.join(model_dir, save_filename)
    np.savez(save_path, X=X, Y=Y, reward_grid=reward_grid)
    print(f"Reward surface data saved at: {save_path}")

    plot_filename = f"reward_surface_plot_{model_path}.png"
    plot_save_path = os.path.join(model_dir, plot_filename)
    plt.savefig(plot_save_path, dpi=300)
    print(f"Reward surface plot saved at: {plot_save_path}")

    plt.show()


if __name__ == "__main__":
    main()
