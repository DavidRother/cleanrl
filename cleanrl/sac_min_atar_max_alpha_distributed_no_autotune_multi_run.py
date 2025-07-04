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
    exp_name: str = os.path.basename(__file__)[:-len(".py")]
    """the name of this experiment"""
    seed: int = 123456
    """seed of the experiment (base seed; each run will use seed + run_index)"""
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
    env_id: str = "MinAtar/Seaquest-v1"
    """the id of the environment"""
    total_timesteps: int = 3000000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e5)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """target smoothing coefficient (default: 1)"""
    batch_size: int = 64
    """the batch size of sample from the replay memory"""
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
    autotune: bool = False
    """automatic tuning of the entropy coefficient"""
    target_entropy_scale: float = 0.89
    """coefficient for scaling the autotune entropy target"""
    alpha_eps: float = 2e-2
    """a small epsilon added for adjusting metrics"""


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
        # Uncomment the following wrappers if desired:
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

    # Create one shared TensorBoard writer that will log all runs into the same folder.
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
        run_name = f"{args.env_id}__{args.exp_name}__{run_prefix}__{int(time.time())}"
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

        rb = ReplayBuffer(
            args.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            device,
            handle_timeout_termination=False,
        )
        start_time = time.time()

        progress_bar = tqdm.trange(args.total_timesteps, desc=f"Training {run_prefix}", dynamic_ncols=True)
        latest_return = None
        episode_returns = []
        episodic_lengths = []
        avg_return_normalised = alpha

        lowest_return = np.inf

        alpha_eps = args.alpha_eps

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

                    if episodic_return < lowest_return:
                        lowest_return = episodic_return

                    avg_return_normalised = (avg_return - lowest_return) / np.mean(episodic_lengths)
                    adjusted_metric = avg_return_normalised - alpha
                    writer.add_scalar(f"{run_prefix}/charts/episodic_return_adjusted", adjusted_metric, global_step)
                    writer.add_scalar(f"{run_prefix}/charts/alpha_upper_bound", avg_return_normalised + args.alpha_eps, global_step)
                    break

            # Process replay buffer
            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]
            rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
            obs = next_obs

            # Training updates.
            if global_step > args.learning_starts:
                if global_step % args.update_frequency == 0:
                    data = rb.sample(args.batch_size)

                    _, log_pi, action_probs = actor.get_action(data.observations)
                    policy_dist = Categorical(probs=action_probs)
                    entropy = policy_dist.entropy().mean().item()

                    alpha_used = (torch.as_tensor(avg_return_normalised, device=device) + args.alpha_eps) / entropy
                    with torch.no_grad():
                        _, next_state_log_pi, next_state_action_probs = actor.get_action(data.next_observations)
                        qf1_next_target = qf1_target(data.next_observations)
                        qf2_next_target = qf2_target(data.next_observations)
                        q_min = torch.min(qf1_next_target, qf2_next_target)
                        abs_q = q_min.abs()
                        per_state_sum = abs_q.sum(1, keepdim=True) + 1e-8
                        dyn_alpha = per_state_sum * alpha_used * per_state_sum.shape[0] / per_state_sum.sum()
                        # we can use the action probabilities instead of MC sampling to estimate the expectation
                        min_qf_next_target = next_state_action_probs * (
                                q_min - dyn_alpha * next_state_log_pi
                        )
                        min_qf_next_target = min_qf_next_target.sum(dim=1)
                        next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target)

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

                    with torch.no_grad():
                        qf1_values = qf1(data.observations)
                        qf2_values = qf2(data.observations)
                        min_qf_values = torch.min(qf1_values, qf2_values)
                    actor_loss = (action_probs * ((alpha_used * log_pi) - min_qf_values)).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        alpha_loss = (action_probs.detach() * (-log_alpha.exp() * (log_pi + target_entropy).detach())).mean()
                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

                # Update the target networks.
                if global_step % args.target_network_frequency == 0:
                    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                if global_step % 100 == 0:
                    writer.add_scalar(f"{run_prefix}/losses/qf1_values", qf1_a_values.mean().item(), global_step)
                    writer.add_scalar(f"{run_prefix}/losses/qf2_values", qf2_a_values.mean().item(), global_step)
                    writer.add_scalar(f"{run_prefix}/losses/qf1_loss", qf1_loss.item(), global_step)
                    writer.add_scalar(f"{run_prefix}/losses/qf2_loss", qf2_loss.item(), global_step)
                    writer.add_scalar(f"{run_prefix}/losses/qf_loss", qf_loss.item() / 2.0, global_step)
                    writer.add_scalar(f"{run_prefix}/losses/actor_loss", actor_loss.item(), global_step)
                    writer.add_scalar(f"{run_prefix}/losses/alpha", alpha, global_step)
                    writer.add_scalar(f"{run_prefix}/losses/alpha_used", alpha_used, global_step)
                    sps = int(global_step / (time.time() - start_time))
                    writer.add_scalar(f"{run_prefix}/charts/SPS", sps, global_step)
                    writer.add_scalar(f"{run_prefix}/charts/mean_policy_entropy", entropy, global_step)
                    if args.autotune:
                        writer.add_scalar(f"{run_prefix}/losses/alpha_loss", alpha_loss.item(), global_step)

                    progress_bar.set_postfix({
                        "step": global_step,
                        "return": f"{float(latest_return):.2f}" if latest_return is not None else "N/A",
                        "sps": sps
                    })

        # End of training for this run.
        envs.close()

        # Save the final actor model into the TensorBoard folder.
        model_save_path = os.path.join(writer.log_dir, f"final_model_{run_prefix}.pt")
        torch.save(actor.state_dict(), model_save_path)
        print(f"Saved final model for {run_prefix} to {model_save_path}")

    writer.close()
