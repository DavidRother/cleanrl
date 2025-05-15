# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_ataripy
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
import tqdm
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


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 123456
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
    env_id: str = "MinAtar/Freeway-v1"
    """the id of the environment"""
    total_timesteps: int = 3000000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e5)
    """the replay memory buffer size"""  # smaller than in original paper but evaluation is done only for 100k steps anyway
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """target smoothing coefficient (default: 1)"""
    batch_size: int = 64
    """the batch size of sample from the reply memory"""
    learning_starts: int = 2e4
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    update_frequency: int = 4
    """the frequency of training updates"""
    target_network_frequency: int = 8000
    """the frequency of updates for the target networks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    target_entropy_scale: float = 0.89
    """coefficient for scaling the autotune entropy target"""


class ChannelFirstWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        assert len(obs_shape) == 3, "Expected 3D observation (H, W, C)"
        c, h, w = obs_shape[2], obs_shape[0], obs_shape[1]
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(c, h, w),
            dtype=self.observation_space.dtype
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return np.transpose(observation, (2, 0, 1))


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = ChannelFirstWrapper(env)
        # env = NoopResetEnv(env, noop_max=30)
        # env = MaxAndSkipEnv(env, skip=4)
        # env = EpisodicLifeEnv(env)
        # if "FIRE" in env.unwrapped.get_action_meanings():
        #     env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        # env = gym.wrappers.ResizeObservation(env, (84, 84))
        # env = gym.wrappers.GrayScaleObservation(env)
        # env = gym.wrappers.FrameStack(env, 4)

        env.action_space.seed(seed)
        return env

    return thunk


def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ALGO LOGIC: initialize agent here:
# NOTE: Sharing a CNN encoder between Actor and Critics is not recommended for SAC without stopping actor gradients
# See the SAC+AE paper https://arxiv.org/abs/1910.01741 for more info
# TL;DR The actor's gradients mess up the representation when using a joint encoder
class SoftQNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[0], 16, kernel_size=3, stride=1)),
            nn.Flatten(),
        )

        with torch.inference_mode():
            output_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]

        self.fc1 = layer_init(nn.Linear(output_dim, 128))
        self.fc_q = layer_init(nn.Linear(128, envs.single_action_space.n))

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=device)
        x = F.relu(self.conv(x))
        x = F.relu(self.fc1(x))
        q_vals = self.fc_q(x)
        return q_vals


class Actor(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[0], 16, kernel_size=3, stride=1)),
            nn.Flatten(),
        )

        with torch.inference_mode():
            output_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]

        self.fc1 = layer_init(nn.Linear(output_dim, 128))
        self.fc_logits = layer_init(nn.Linear(128, envs.single_action_space.n))

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=device)
        x = F.relu(self.conv(x))
        x = F.relu(self.fc1(x))
        logits = self.fc_logits(x)

        return logits

    def get_action(self, x):
        logits = self(x)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        return action, log_prob, action_probs


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1" "gymnasium[atari,accept-rom-license]==0.28.1"  "ale-py==0.8.1" 
"""
        )
    args = tyro.cli(Args)
    writer = SummaryWriter(f"runs/{args.env_id}__{args.exp_name}")
    # You can also add the hyperparameters text once (they remain common across runs)
    writer.add_text(
        "global_hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        global_step=0,
    )

    # Device remains the same for all runs.
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Outer loop: run training 5 times with different seeds.
    for run_idx in range(5):

        # Update the seed for the current run
        current_seed = args.seed + run_idx
        run_prefix = f"seed_{current_seed}"
        folder_name = f"{args.env_id}__{args.exp_name}"
        run_name = f"folder_name__{run_prefix}__{int(time.time())}"
        print(f"Starting run: {run_prefix}")

        # (Re)seed randomness for current run
        random.seed(current_seed)
        np.random.seed(current_seed)
        torch.manual_seed(current_seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic

        # Optional: If tracking with wandb, initialize a new run.
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

        # Set up the vectorized environment.
        envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_id, current_seed, 0, args.capture_video, run_name)]
        )
        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

        # Initialize networks and optimizers.
        actor = Actor(envs).to(device)
        qf1 = SoftQNetwork(envs).to(device)
        qf2 = SoftQNetwork(envs).to(device)
        qf1_target = SoftQNetwork(envs).to(device)
        qf2_target = SoftQNetwork(envs).to(device)
        qf1_target.load_state_dict(qf1.state_dict())
        qf2_target.load_state_dict(qf2.state_dict())

        q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr, eps=1e-4)
        actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr, eps=1e-4)

        # Automatic entropy tuning
        if args.autotune:
            target_entropy = -args.target_entropy_scale * torch.log(1 / torch.tensor(envs.single_action_space.n))
            log_alpha = torch.zeros(1, requires_grad=True, device=device)
            alpha = log_alpha.exp().item()
            a_optimizer = optim.Adam([log_alpha], lr=args.q_lr, eps=1e-4)
        else:
            alpha = args.alpha

        rb = ReplayBuffer(args.buffer_size, envs.single_observation_space, envs.single_action_space, device,
                          handle_timeout_termination=False, )
        start_time = time.time()

        progress_bar = tqdm.trange(args.total_timesteps, desc=f"Training {run_prefix}", dynamic_ncols=True)
        latest_return = None
        episode_returns = []
        episodic_lengths = []

        lowest_return = np.inf

        obs, _ = envs.reset(seed=current_seed)
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
                    writer.add_scalar(f"{run_prefix}/charts/episodic_return", episodic_return, global_step)
                    writer.add_scalar(f"{run_prefix}/charts/episodic_length", episodic_length, global_step)

                    episode_returns.append(episodic_return)
                    episodic_lengths.append(episodic_length)
                    if len(episode_returns) > 50:
                        episode_returns.pop(0)
                        episodic_lengths.pop(0)
                    avg_return = np.mean(episode_returns)
                    writer.add_scalar(f"{run_prefix}/charts/episodic_return_avg", avg_return, global_step)
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
                if global_step % args.update_frequency == 0:
                    data = rb.sample(args.batch_size)
                    # CRITIC training
                    with torch.no_grad():
                        _, next_state_log_pi, next_state_action_probs = actor.get_action(data.next_observations)
                        qf1_next_target = qf1_target(data.next_observations)
                        qf2_next_target = qf2_target(data.next_observations)
                        # we can use the action probabilities instead of MC sampling to estimate the expectation
                        min_qf_next_target = next_state_action_probs * (torch.min(qf1_next_target, qf2_next_target))
                        # adapt Q-target for discrete Q-function
                        min_qf_next_target = min_qf_next_target.sum(dim=1)
                        next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target)

                    # use Q-values only for the taken actions
                    qf1_values = qf1(data.observations)
                    qf2_values = qf2(data.observations)
                    qf1_a_values = qf1_values.gather(1, data.actions.long()).view(-1)
                    qf2_a_values = qf2_values.gather(1, data.actions.long()).view(-1)
                    qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                    qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                    qf_loss = qf1_loss + qf2_loss

                    q_optimizer.zero_grad()
                    qf_loss.backward()
                    q_optimizer.step()

                    # ACTOR training
                    _, log_pi, action_probs = actor.get_action(data.observations)
                    policy_dist = Categorical(probs=action_probs)
                    entropy = policy_dist.entropy().mean().item()
                    with torch.no_grad():
                        qf1_values = qf1(data.observations)
                        qf2_values = qf2(data.observations)
                        min_qf_values = torch.min(qf1_values, qf2_values)
                    # no need for reparameterization, the expectation can be calculated for discrete actions
                    actor_loss = (action_probs * ((alpha * log_pi) - min_qf_values)).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        # re-use action probabilities for temperature loss
                        alpha_loss = (action_probs.detach() * (-log_alpha.exp() * (log_pi + target_entropy).detach())).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

                    primal_residual = max(0.0, target_entropy - entropy)

                    # 2) Dual‐feasibility residual: r_d = max(0, -alpha)
                    dual_residual = max(0.0, -alpha)

                    # 3) Stationarity/subgradient residual:
                    #    if alpha>0: |H - H_target|, else: max(0, H_target - H)
                    if alpha > 0.0:
                        stationarity_residual = abs(entropy - target_entropy)
                    else:
                        stationarity_residual = max(0.0, target_entropy - entropy)

                    # 4) Complementary‐slackness residual: r_cs = alpha * (H - H_target)
                    complementary_slackness = alpha * (entropy - target_entropy)

                    probs_with_bonus = torch.softmax(min_qf_values / alpha, dim=1)  # [B,A]

                    entropy_with_bonus = -(probs_with_bonus * probs_with_bonus.log()).sum(dim=1).mean().item()

                    q_var = min_qf_values.var(dim=1, unbiased=False).mean().item()

                # update the target networks
                if global_step % args.target_network_frequency == 0:
                    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                if global_step % 100 == 0:
                    writer.add_scalar(f"{run_prefix}/residuals/primal_feasibility", primal_residual, global_step)
                    writer.add_scalar(f"{run_prefix}/residuals/dual_feasibility", dual_residual, global_step)
                    writer.add_scalar(f"{run_prefix}/residuals/stationarity", stationarity_residual, global_step)
                    writer.add_scalar(f"{run_prefix}/residuals/complementary_slackness", complementary_slackness, global_step)
                    writer.add_scalar(f"{run_prefix}/losses/qf1_values", qf1_a_values.mean().item(), global_step)
                    writer.add_scalar(f"{run_prefix}/losses/qf2_values", qf2_a_values.mean().item(), global_step)
                    writer.add_scalar(f"{run_prefix}/losses/qf1_loss", qf1_loss.item(), global_step)
                    writer.add_scalar(f"{run_prefix}/losses/qf2_loss", qf2_loss.item(), global_step)
                    writer.add_scalar(f"{run_prefix}/losses/qf_loss", qf_loss.item() / 2.0, global_step)
                    writer.add_scalar(f"{run_prefix}/losses/actor_loss", actor_loss.item(), global_step)
                    writer.add_scalar(f"{run_prefix}/losses/q_entropy_with_bonus", entropy_with_bonus, global_step)
                    writer.add_scalar(f"{run_prefix}/losses/q_variance", q_var, global_step)
                    writer.add_scalar(f"{run_prefix}/losses/alpha", alpha, global_step)
                    sps = int(global_step / (time.time() - start_time))
                    writer.add_scalar(f"{run_prefix}/charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                    writer.add_scalar(f"{run_prefix}/charts/mean_policy_entropy", entropy, global_step)
                    if args.autotune:
                        writer.add_scalar(f"{run_prefix}/losses/alpha_loss", alpha_loss.item(), global_step)

                    progress_bar.set_postfix({
                        "step": global_step,
                        "return": f"{float(latest_return):.2f}" if latest_return is not None else "N/A",
                        "sps": sps
                    })

        envs.close()
        model_save_path = os.path.join(writer.log_dir, f"final_model_{run_prefix}.pt")
        torch.save(actor.state_dict(), model_save_path)
        print(f"Saved final model for {run_prefix} to {model_save_path}")

    writer.close()
