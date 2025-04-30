import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import tqdm

# Mixed-precision utilities
from torch.cuda.amp import autocast, GradScaler

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    capture_video: bool = False
    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 1000000
    num_envs: int = 1
    buffer_size: int = int(1e6)
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    learning_starts: int = 5e3
    policy_lr: float = 3e-4
    q_lr: float = 1e-3
    policy_frequency: int = 2
    target_network_frequency: int = 1
    alpha: float = 0.02
    action_sampling: int = 30
    delta_start: float = 0.3
    delta_end: float = 0.99999
    delta_fraction: float = 0.7
    lambda_lr: float = 1e-3
    lambda_init: float = 0.0

def kl_categorical_vs_uniform(p, n):
    if not (0 < p < 1):
        raise ValueError("p must be in (0,1)")
    if n < 2:
        raise ValueError("n must be at least 2")

    # term for the heavy action
    term1 = p * np.log(p * n)
    # term for the remaining n-1 actions
    term2 = (1 - p) * np.log((1 - p) * n / (n - 1))
    return term1 + term2


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


def kl_to_max_entropy(log_std: torch.Tensor) -> torch.Tensor:
    """
    Compute the KL divergence between each diagonal Gaussian
      P = N(mean, σ^2)    with σ = exp(log_std)
    and the maximum‐entropy Gaussian
      Q = N(mean, σ_max^2) with σ_max = exp(LOG_STD_MAX).

    Args:
        log_std: Tensor of shape [..., action_dim]

    Returns:
        kl: Tensor of shape [...] containing the KL(P||Q) per sample.
    """
    # current std
    sigma = torch.exp(log_std)
    # max‐entropy std
    sigma_max = torch.exp(torch.tensor(LOG_STD_MAX, device=log_std.device))
    # KL per dimension: log(σ_max/σ) + (σ^2)/(2 σ_max^2) − 1/2
    term1 = LOG_STD_MAX - log_std
    term2 = (sigma.pow(2)) / (2 * sigma_max.pow(2))
    kl_per_dim = term1 + term2 - 0.5
    # sum over action dimensions
    return kl_per_dim.sum(dim=-1)


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def sample_actions(self, x: torch.Tensor, n: int):
        """
        Given a batch of states x (shape [B, obs_dim]), sample n actions per state.
        Returns:
          actions_flat: Tensor of shape [B*n, action_dim]
          log_probs_flat: Tensor of shape [B*n, 1]
        """
        B = x.shape[0]
        # get policy parameters
        mean, log_std = self(x)                    # both [B, action_dim]

        log_std_const = torch.full_like(mean, LOG_STD_MAX, device=mean.device)
        std_const = log_std_const.exp()

        # expand to [B, n, action_dim]
        mean_exp = mean.unsqueeze(1).expand(-1, n, -1)
        std_exp  = std_const.unsqueeze(1).expand(-1, n, -1)

        # sample
        normal = torch.distributions.Normal(mean_exp, std_exp)
        x_t = normal.rsample()                     # [B, n, action_dim]
        y_t = torch.tanh(x_t)

        # rescale to action space
        actions = y_t * self.action_scale + self.action_bias

        # log prob with tanh correction
        log_prob = normal.log_prob(x_t)
        log_prob = log_prob - torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)  # [B, n, 1]

        # flatten to [B*n, ...]
        actions_flat = actions.view(B * n, -1)
        log_probs_flat = log_prob.view(B * n, 1)

        return actions_flat, log_probs_flat, mean, log_std


def delta_schedule(
    step: int,
    total_steps: int,
    delta_min: float,
    delta_max: float,
    power: float = 3.0,        # 1.0 → linear, >1.0 → slow-start, <1.0 → fast-start
) -> float:
    """
    Interpolates between `delta_min` (t=0) and `delta_max` (t=1) with a
    power-law on the *fraction of remaining entropy*.

        s = (step / total_steps) ** power
        return delta_min + (delta_max - delta_min) * s
    """
    s = (step / total_steps) ** power
    return delta_min + (delta_max - delta_min) * s


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # env setup
    envs = gym.vector.SyncVectorEnv([
        make_env(args.env_id, args.seed + i, i, args.capture_video, run_name)
        for i in range(args.num_envs)
    ])

    # instantiate networks
    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    # torch.compile for speed
    actor = torch.compile(actor)
    qf1 = torch.compile(qf1)
    qf2 = torch.compile(qf2)

    # optimizers
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    std_params = list(actor.fc_logstd.parameters())
    policy_params = [p for p in actor.parameters() if p not in std_params]
    actor_optimizer = optim.Adam(policy_params, lr=args.policy_lr)
    std_optimizer = optim.Adam(std_params, lr=args.policy_lr)

    # mixed precision scaler
    scaler = GradScaler()

    # replay buffer, logging, etc.
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    start_time = time.time()
    progress_bar = tqdm.trange(args.total_timesteps, desc="Training", dynamic_ncols=True)
    latest_return = None
    episode_returns = []
    episodic_lengths = []
    buffer_size = args.buffer_size
    obs, _ = envs.reset(seed=args.seed)

    for global_step in progress_bar:
        # collect
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(args.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.tensor(obs, device=device))
            actions = actions.cpu().numpy()
        next_obs, rewards, term, trunc, infos = envs.step(actions)
        rb.add(obs, next_obs, actions, rewards, term, infos)
        if "final_info" in infos:
            for info in infos["final_info"]:
                # Skip the envs that are not done
                if "episode" not in info:
                    continue
                episodic_return = info["episode"]["r"]
                episodic_length = info["episode"]["l"]
                latest_return = episodic_return
                writer.add_scalar(f"charts/episodic_return", episodic_return, global_step)
                writer.add_scalar(f"charts/episodic_length", episodic_length, global_step)

                episode_returns.append(episodic_return)
                episodic_lengths.append(episodic_length)
                if sum(episodic_lengths) > buffer_size:
                    episode_returns.pop(0)
                    episodic_lengths.pop(0)
                avg_return = sum(episode_returns) / len(episode_returns)
                writer.add_scalar(f"charts/episodic_return_avg", avg_return, global_step)
                break
        obs = next_obs

        # training
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            # reuse obs replication
            B = data.observations.shape[0]
            K = args.action_sampling
            obs_rep = data.observations.unsqueeze(1).repeat(1, K, 1).view(B*K, -1)

            delta_t = delta_schedule(
                step=global_step,
                total_steps=int(args.delta_fraction * args.total_timesteps),
                delta_min=kl_categorical_vs_uniform(args.delta_start, K),
                delta_max=kl_categorical_vs_uniform(args.delta_end, K)
            )

            # sample actions once
            actions_rep, _, mean_curr, log_std_pred = actor.sample_actions(data.observations, K)

            with autocast():
                # critic loss
                with torch.no_grad():
                    next_a, next_logp, _ = actor.get_action(data.next_observations)
                    target_q = torch.min(
                        qf1_target(data.next_observations, next_a),
                        qf2_target(data.next_observations, next_a)
                    )
                    next_q = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * target_q.view(-1)
                q1_val = qf1(data.observations, data.actions).view(-1)
                q2_val = qf2(data.observations, data.actions).view(-1)
                qf1_loss = F.mse_loss(q1_val, next_q)
                qf2_loss = F.mse_loss(q2_val, next_q)
                q_loss = qf1_loss + qf2_loss

                # hinge loss
                q1_rep = qf1(obs_rep, actions_rep).view(B, K)
                q2_rep = qf2(obs_rep, actions_rep).view(B, K)
                q_rep = torch.min(q1_rep, q2_rep)
                p = torch.softmax(q_rep / args.alpha, dim=1)
                u = torch.full_like(p, 1.0 / K)
                kl = (p * (torch.log(p + 1e-8) - torch.log(u))).sum(dim=1)
                hinge = torch.clamp(kl - delta_t, min=0.0).mean()
                primal_loss = q_loss + args.lambda_lr * hinge

                # policy + std loss (delayed)
                actor_loss = torch.tensor(0.0, device=device)
                std_loss = torch.tensor(0.0, device=device)
                if global_step % args.policy_frequency == 0:
                    pi, logp, _ = actor.get_action(data.observations)
                    min_q_pi = torch.min(qf1(data.observations, pi), qf2(data.observations, pi))
                    actor_loss = ((args.alpha * logp) - min_q_pi).mean()
                    # std loss from the same samples
                    a = actions_rep.view(B, K, -1)
                    mu = mean_curr.unsqueeze(1).detach()
                    diff2 = (a - mu).pow(2)
                    target_var = (p.unsqueeze(-1) * diff2).sum(dim=1)
                    target_log_std = 0.5 * torch.log(target_var + 1e-8)
                    std_loss = F.mse_loss(log_std_pred, target_log_std.detach())

            # optimizer steps
            scaler.scale(q_loss).backward()
            scaler.step(q_optimizer)
            scaler.update()

            if global_step % args.policy_frequency == 0:
                for _ in range(args.policy_frequency):
                    # perform policy and std updates policy_frequency times
                    scaler.scale(actor_loss + std_loss).backward()
                    scaler.step(actor_optimizer)
                    scaler.step(std_optimizer)
                    scaler.update()

            # soft target updates
            if global_step % args.target_network_frequency == 0:
                for p, tp in zip(qf1.parameters(), qf1_target.parameters()):
                    tp.data.copy_(args.tau * p.data + (1 - args.tau)
                                  * tp.data)
                for p, tp in zip(qf2.parameters(), qf2_target.parameters()):
                    tp.data.copy_(args.tau * p.data + (1 - args.tau)
                                  * tp.data)

    envs.close()
    writer.close()
