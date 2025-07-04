# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
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
import math


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.02
    """Entropy regularization coefficient."""
    action_sampling = 30
    delta_start: float = 0.3  # very exploratory at the beginning
    delta_end: float = 0.99999
    delta_fraction: float = 0.7  # finish annealing after 80 % of training
    lambda_lr: float = 1e-3            # step size for dual variable λ
    lambda_init: float = 0.0
    lambda_decay: float = 1e-3


def kl_categorical_vs_uniform(p, n):
    if not (0 < p < 1):
        raise ValueError("p must be in (0,1)")
    if n < 2:
        raise ValueError("n must be at least 2")
    term1 = p * np.log(p * n)
    term2 = (1 - p) * np.log((1 - p) * n / (n - 1))
    return term1 + term2


def diagonal_hinge_kl_loss(logstd, logstd_max, delta):
    var = torch.exp(2 * logstd)
    var_max = torch.exp(2 * logstd_max)
    dim = logstd.size(-1)
    kl = 0.5 * (torch.sum(var / var_max, dim=-1) + 2 * torch.sum(logstd_max - logstd, dim=-1) - dim)
    violation = F.relu(kl - delta)
    return violation.mean()


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
        std_const = log_std.exp()

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
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)

    std_params = list(actor.fc_logstd.parameters())
    policy_params = [p for p in actor.parameters() if all(p is not sp for sp in actor.fc_logstd.parameters())]

    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(policy_params, lr=args.policy_lr)
    std_optimizer = optim.Adam(std_params, lr=args.policy_lr)

    lambda_param = torch.tensor(args.lambda_init, dtype=torch.float32, device=device)

    envs.single_observation_space.dtype = np.float32
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

    action_dim = envs.single_action_space.shape[0]
    max_logstd_vec = torch.full((action_dim,), LOG_STD_MAX)
    min_logstd_vec = torch.full((action_dim,), LOG_STD_MIN)

    alpha = args.alpha  # softmax temperature
    delta_start = kl_categorical_vs_uniform(args.delta_start, args.action_sampling)
    delta_end = kl_categorical_vs_uniform(args.delta_end, args.action_sampling)
    beta_start = kl_to_max_entropy(max_logstd_vec).item()
    beta_end = kl_to_max_entropy(min_logstd_vec).item()
    beta_fraction = args.delta_fraction
    delta_fraction = args.delta_fraction
    print(f"KL start bound: {delta_start}")
    print(f"KL end bound: {delta_end}")

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in progress_bar:
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
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

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            delta_t = delta_schedule(
                step=global_step,
                total_steps=int(delta_fraction * args.total_timesteps),
                delta_min=delta_start,
                delta_max=delta_end,
                power=3,
            )
            beta_t = delta_schedule(
                step=global_step,
                total_steps=int(beta_fraction * args.total_timesteps),
                delta_min=beta_start,
                delta_max=beta_end,
                power=3,
            )
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            B = data.observations.shape[0]
            K = args.action_sampling
            # repeat each state K times
            obs_rep = data.observations.unsqueeze(1).repeat(1, K, 1).view(B * K, -1)
            # sample K actions per state
            actions_rep, logp_rep, mean_current, _ = actor.sample_actions(data.observations, args.action_sampling)
            # compute Q-values
            q1_rep = qf1(obs_rep, actions_rep.detach()).view(B, K)
            q2_rep = qf2(obs_rep, actions_rep.detach()).view(B, K)
            q_rep = torch.min(q1_rep, q2_rep)
            # form a discrete distribution over the K Q’s
            p = torch.softmax(q_rep / args.alpha, dim=1)  # shape [B, K]
            # uniform target
            u = torch.full_like(p, 1.0 / K)
            # KL(p || u) = sum p * (log p - log u)
            kl = (p * (torch.log(p + 1e-8) - torch.log(u))).sum(dim=1)
            violation = torch.clamp(kl - delta_t, min=0.0)
            hinge_mean = violation.mean()

            primal_loss = qf_loss + lambda_param.detach() * hinge_mean

            # optimize the model
            q_optimizer.zero_grad()
            primal_loss.backward()
            q_optimizer.step()

            with torch.no_grad():
                lambda_param += args.lambda_lr * hinge_mean
                lambda_param -= args.lambda_decay * lambda_param
                lambda_param.clamp_(min=0.0)

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):
                    pi, log_pi, log_std = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = (-min_qf_pi).mean()

                    # ── 2. fresh sampling for σ-loss  ───────────────────────────
                    actions_rep, _, mean_current, log_std_pred = actor.sample_actions(
                        data.observations, args.action_sampling
                    )
                    a = actions_rep.reshape(B, K, -1)
                    mu = mean_current[:, None, :].detach()
                    diff2 = (a - mu).pow(2)
                    q1_rep = qf1(obs_rep, actions_rep).view(B, K)
                    q2_rep = qf2(obs_rep, actions_rep).view(B, K)
                    p = torch.softmax(torch.min(q1_rep, q2_rep) / alpha, dim=1)
                    sigma2_hat = (p.unsqueeze(-1) * diff2).sum(dim=1)
                    target_log_std = 0.5 * torch.log(sigma2_hat + 1e-8)
                    std_loss = F.mse_loss(log_std_pred, target_log_std.detach())

                    # KL between current log std and LOG_STD_MAX
                    expanded_log_std_max = torch.full_like(log_std_pred, LOG_STD_MAX)
                    hinge_actor_std_loss = diagonal_hinge_kl_loss(log_std_pred, expanded_log_std_max, beta_t)

                    # ── 3. optimise both heads together  ───────────────────────
                    actor_optimizer.zero_grad()
                    std_optimizer.zero_grad()
                    (actor_loss + std_loss + hinge_actor_std_loss).backward()
                    std_optimizer.step()  # only fc_logstd params here
                    actor_optimizer.step()  # trunk + mean head

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                const = 0.5 * (1.0 + math.log(2 * math.pi))
                entropy_per_dim = log_std + const  # [B, action_dim]
                entropy_per_sample = entropy_per_dim.sum(dim=-1)  # [B]
                avg_entropy = entropy_per_sample.mean()
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/mean_entropy", avg_entropy, global_step)
                writer.add_scalar(f"charts/delta", delta_t, global_step)
                writer.add_scalar(f"charts/kl_mean", kl.mean().item(), global_step)
                writer.add_scalar(f"losses/hinge_loss", hinge_mean.item(), global_step)
                writer.add_scalar(f"charts/lambda", lambda_param.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                sps = int(global_step / (time.time() - start_time))
                writer.add_scalar(f"charts/SPS", sps, global_step)

                progress_bar.set_postfix({
                    "step": global_step,
                    "return": f"{float(latest_return):.2f}" if latest_return is not None else "N/A",
                    "sps": sps
                })

    envs.close()
    writer.close()
