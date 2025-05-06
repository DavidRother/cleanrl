import os
import random
import time
from dataclasses import dataclass
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import tqdm
import math
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak averaging."""
    with torch.no_grad():
        for p_t, p in zip(target.parameters(), source.parameters()):
            p_t.mul_(1.0 - tau).add_(p, alpha=tau)
            
            
def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


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


def u_alpha(y: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Derivative of f‑conjugate for α‑divergence (occupancy weight).
    Handles α ≈ 0 or 1 via limits.
    """
    eps = 1e-8
    # Broadcasting handled automatically.
    if torch.allclose(alpha, torch.tensor(1.0).to(alpha)):  # forward KL
        return torch.exp(y - 1)
    if torch.allclose(alpha, torch.tensor(0.0).to(alpha)):  # reverse KL (α→0)
        return 1.0 / (1.0 - y)
    # General α
    return torch.clamp(1 + (1 - alpha) / alpha * y, min=eps) ** (1 / (alpha - 1))


def meta_update(self, obs):
    """Meta‑gradient on α every cfg.meta_update_interval steps."""
    meta_opt.zero_grad()
    action, log_prob, _ = actor.sample(obs)
    q_val = torch.min(q1(obs, action), q2(obs, action)).detach()
    eta = log_eta.exp().detach()
    z = q_val / eta
    # Objective: -weighted log‑prob (see one‑step improvement surrogate)
    loss = (u_alpha(z, alpha).detach() * log_prob).mean()
    loss.backward()
    meta_opt.step()
    # Clamp α to reasonable range
    alpha.data.clamp_(-1.0, 3.0)


@dataclass
class Args:
    obs_dim: int
    action_dim: int
    device: torch.device = torch.device("cpu")
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    divergence_epsilon: float = 0.1  # ε in trust region
    meta_lr: float = 1e-3            # learning rate for α
    batch_size: int = 256
    target_update_interval: int = 1
    meta_update_interval: int = 1000
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_frequency: int = 2
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    env_id: str = "HalfCheetah-v4"
    """the environment id of the task"""
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1" "gymnasium[atari,accept-rom-license]==0.28.1"  "ale-py==0.8.1" 
"""
        )
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # Create one shared TensorBoard writer that will log all runs into the same folder.
    writer = SummaryWriter(f"runs_old/{args.env_id}__{args.exp_name}")
    # You can also add the hyperparameters text once (they remain common across runs)
    writer.add_text(
        "global_hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        global_step=0,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(1)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Divergence parameters
    log_eta = torch.nn.Parameter(torch.zeros((), device=device))
    alpha = torch.nn.Parameter(torch.tensor(1.0, device=device))  # start at forward KL
    dual_opt = torch.optim.Adam([log_eta], lr=args.lr)
    meta_opt = torch.optim.Adam([alpha], lr=args.meta_lr)

    total_it = 0

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    progress_bar = tqdm.trange(args.total_timesteps, desc=f"Training {args.env_id}", dynamic_ncols=True)
    latest_return = None
    episode_returns = []
    episodic_lengths = []
    avg_return_normalised = alpha

    lowest_return = np.inf

    obs, _ = envs.reset(seed=args.seed)
    for global_step in progress_bar:
        # Action selection
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Log episodic information.
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                episodic_return = info["episode"]["r"]
                episodic_length = info["episode"]["l"]
                latest_return = episodic_return
                writer.add_scalar(f"charts/episodic_return", episodic_return, global_step)
                writer.add_scalar(f"charts/episodic_length", episodic_length, global_step)

                episode_returns.append(episodic_return)
                episodic_lengths.append(episodic_length)
                if len(episode_returns) > 50:
                    episode_returns.pop(0)
                    episodic_lengths.pop(0)
                avg_return = np.mean(episode_returns)
                writer.add_scalar(f"charts/episodic_return_avg", avg_return, global_step)

                if episodic_return < lowest_return:
                    lowest_return = episodic_return
                break

        # Process replay buffer
        real_next_obs = next_obs.copy()
        for idx, (trunc, term) in enumerate(zip(truncations, terminations)):
            if trunc or term:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        # Training updates.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)

            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (
                    min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            pi, log_pi, action_probs = actor.get_action(data.next_observations)
            with torch.no_grad():
                qf1_pi = qf1(data.observations, pi)
                qf2_pi = qf2(data.observations, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
            eta = log_eta.exp().detach().clone()
            eps = args.divergence_epsilon
            for _ in range(10):
                z = min_qf_pi / eta
                u = u_alpha(z, alpha)
                b = f_star_fn(z) - z * u
                grad = divergence_limit - b.mean() / eta
                eta = torch.clamp(eta - lr * grad, min=eta_min, max=eta_max)
            for _ in range(10):
                z = min_qf_pi / eta
                w = u_alpha(z, alpha)
                g = eta * (torch.mean(w) * math.log(torch.mean(torch.exp(z))) + eps)  # dummy deriv.
                # crude gradient, real deriv see paper – this keeps code simple
                eta = torch.clamp(eta - 0.1 * g, min=1e-6, max=100.0)
            eta = eta.detach()
            log_eta.data.copy_(eta.log())
            w = u_alpha(min_qf_pi / eta, alpha).detach()
            actor_loss = (-(w * log_pi) - min_qf_pi).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            if global_step % args.meta_update_interval == 0:
                meta_opt.zero_grad()
                meta_loss = (u_alpha(min_qf_pi / eta, alpha).detach() * log_pi).mean()
                meta_loss.backward()
                meta_opt.step()
                alpha.data.clamp_(-1.0, 3.0)

            # Update the target networks.
            if global_step % args.target_update_interval == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar(f"losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar(f"losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar(f"losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar(f"losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar(f"losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar(f"losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar(f"losses/alpha", alpha.item(), global_step)
                writer.add_scalar(f"losses/eta", eta.item(), global_step)
                writer.add_scalar(f"losses/weight", w.item(), global_step)
                sps = int(global_step / (time.time() - start_time))
                writer.add_scalar(f"charts/SPS", sps, global_step)

                progress_bar.set_postfix({
                    "step": global_step,
                    "return": f"{float(latest_return):.2f}" if latest_return is not None else "N/A",
                    "sps": sps
                })

    # End of training for this run.
    envs.close()

    # Save the final actor model into the TensorBoard folder.
    model_save_path = os.path.join(writer.log_dir, f"final_model.pt")
    torch.save(actor.state_dict(), model_save_path)
    print(f"Saved final model for {args.env_id} to {model_save_path}")

    writer.close()
