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
)
from stable_baselines3.common.buffers import ReplayBuffer
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[:-3]
    seed: int = 123456
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False

    # Algorithm specific arguments
    env_id: str = "MinAtar/Freeway-v1"
    total_timesteps: int = 3_000_000
    buffer_size: int = int(1e5)
    gamma: float = 0.99
    tau: float = 1.0
    batch_size: int = 64
    learning_starts: int = int(2e4)
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    update_frequency: int = 4
    target_network_frequency: int = 8000

    # Bound slack for alpha‐max(s)
    alpha_eps: float = 2e-2

    # ----- Bet‐and‐Secure hyperparameters -----
    bet_window: int = 1000       # V: steps between trend evaluations
    recalib_steps: int = 100     # M: batches to recalibrate (alpha=0)
    kappa: float = 0.5           # shrink factor in DECREASE phase


class ChannelFirstWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        c, h, w = env.observation_space.shape[2], env.observation_space.shape[0], env.observation_space.shape[1]
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(c, h, w), dtype=env.observation_space.dtype
        )
    def observation(self, obs):
        return np.transpose(obs, (2, 0, 1))


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            e = gym.make(env_id, render_mode="rgb_array")
            e = gym.wrappers.RecordVideo(e, f"videos/{run_name}")
        else:
            e = gym.make(env_id)
        e = gym.wrappers.RecordEpisodeStatistics(e)
        e = ChannelFirstWrapper(e)
        e = ClipRewardEnv(e)
        e.action_space.seed(seed)
        return e
    return thunk


def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class SoftQNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[0], 16, 3, 1)),
            nn.Flatten(),
        )
        with torch.inference_mode():
            dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]
        self.fc1 = layer_init(nn.Linear(dim, 128))
        self.fc_q = layer_init(nn.Linear(128, envs.single_action_space.n))

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=device)
        x = F.relu(self.conv(x))
        x = F.relu(self.fc1(x))
        return self.fc_q(x)


class Actor(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[0], 16, 3, 1)),
            nn.Flatten(),
        )
        with torch.inference_mode():
            dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]
        self.fc1 = layer_init(nn.Linear(dim, 128))
        self.fc_logits = layer_init(nn.Linear(128, envs.single_action_space.n))

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=device)
        x = F.relu(self.conv(x))
        x = F.relu(self.fc1(x))
        return self.fc_logits(x)

    def get_action(self, x):
        logits = self(x)
        dist = Categorical(logits=logits)
        a = dist.sample()
        logp = F.log_softmax(logits, dim=1)
        return a, logp, dist.probs


if __name__ == "__main__":
    import stable_baselines3 as sb3
    if sb3.__version__ < "2.0":
        raise ValueError("Please install SB3 2.0.0a1 and compatible Gymnasium")

    args = tyro.cli(Args)

    # Create one shared TensorBoard writer that will log all runs into the same folder.
    writer = SummaryWriter(f"runs_old/{args.env_id}__{args.exp_name}")
    # You can also add the hyperparameters text once (they remain common across runs)
    writer.add_text(
        "global_hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        global_step=0,
    )

    # Device remains the same for all runs.
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    for run_idx in range(5):
        # Update the seed for the current run
        current_seed = args.seed + run_idx
        run_prefix = f"seed_{current_seed}"
        run_name = f"{args.env_id}__{args.exp_name}__{run_prefix}__{int(time.time())}"
        print(f"Starting run: {run_prefix}")

        random.seed(current_seed)
        np.random.seed(current_seed)
        torch.manual_seed(current_seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic

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

        envs = gym.vector.SyncVectorEnv([make_env(args.env_id, current_seed, 0, args.capture_video, run_name)])
        assert isinstance(envs.single_action_space, gym.spaces.Discrete)

        actor = Actor(envs).to(device)
        qf1 = SoftQNetwork(envs).to(device)
        qf2 = SoftQNetwork(envs).to(device)
        qf1_target = SoftQNetwork(envs).to(device)
        qf2_target = SoftQNetwork(envs).to(device)
        qf1_target.load_state_dict(qf1.state_dict())
        qf2_target.load_state_dict(qf2.state_dict())

        q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr, eps=1e-4)
        actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_lr, eps=1e-4)

        rb = ReplayBuffer(
            args.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            device,
            handle_timeout_termination=False,
        )

        # === Bet‐and‐Secure state ===
        phase = "INCREASE"
        R_ref = 0.0
        cumulative_reward = 0.0
        episode_returns = []
        episodic_lengths = []
        buffer_size = args.buffer_size

        obs, _ = envs.reset(seed=current_seed)
        start_time = time.time()
        progress_bar = tqdm.trange(args.total_timesteps, desc=f"Training {run_prefix}", dynamic_ncols=True)

        # Tracking for plotting
        episode_returns, episode_lengths = [], []
        lowest_return = float("inf")

        for global_step in progress_bar:
            # 1) interact
            if global_step < args.learning_starts:
                actions = np.array([envs.single_action_space.sample()])
            else:
                a, _, _ = actor.get_action(torch.Tensor(obs).to(device))
                actions = a.detach().cpu().numpy()

            next_obs, rewards, terms, truncs, infos = envs.step(actions)
            cumulative_reward += float(rewards[0])

            # log episode metrics
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if "episode" not in info:
                        continue
                    episodic_return = info["episode"]["r"]
                    episodic_length = info["episode"]["l"]
                    latest_return = episodic_return
                    writer.add_scalar(f"{run_prefix}/charts/episodic_return", episodic_return, global_step)
                    writer.add_scalar(f"{run_prefix}/charts/episodic_length", episodic_length, global_step)
                    if sum(episodic_lengths) > buffer_size:
                        episode_returns.pop(0)
                        episodic_lengths.pop(0)

                    episode_returns.append(episodic_return)
                    episodic_lengths.append(episodic_length)
                    avg_return = np.mean(episode_returns)
                    writer.add_scalar(f"{run_prefix}/charts/episodic_return_avg", avg_return, global_step)

                    if episodic_return < lowest_return:
                        lowest_return = episodic_return

                    avg_return_normalised = (avg_return - lowest_return) / np.mean(episodic_lengths)
                    adjusted_metric = avg_return_normalised - alpha
                    writer.add_scalar(f"{run_prefix}/charts/episodic_return_adjusted", adjusted_metric, global_step)
                    writer.add_scalar(f"{run_prefix}/charts/alpha_upper_bound", avg_return_normalised + args.alpha_eps, global_step)
                    break

            # store
            real_next = next_obs.copy()
            for i, tr in enumerate(truncs):
                if tr:
                    real_next[i] = infos["final_observation"][i]
            rb.add(obs, real_next, actions, rewards, terms, infos)
            obs = next_obs

            # 2) update
            if global_step > args.learning_starts and global_step % args.update_frequency == 0:
                batch = rb.sample(args.batch_size)
                # compute bound
                avg_ret = np.mean(episode_returns) if episode_returns else 0.0
                norm = (avg_ret - lowest_return) / np.mean(episode_lengths) if episode_lengths else 0.0
                bound = norm + args.alpha_eps

                # choose alpha for this phase
                if phase == "INCREASE":
                    alpha_used = bound
                elif phase == "DECREASE":
                    alpha_used = args.kappa * bound
                else:  # recalibrate phase shouldn't get here
                    alpha_used = 0.0

                # critic update
                with torch.no_grad():
                    _, next_logp, next_probs = actor.get_action(batch.next_observations)
                    q1_t = qf1_target(batch.next_observations)
                    q2_t = qf2_target(batch.next_observations)
                    min_q_t = torch.min(q1_t, q2_t)
                    target = batch.rewards.flatten() + (1 - batch.dones.flatten()) * args.gamma * (
                        (next_probs * (min_q_t - alpha_used * next_logp)).sum(dim=1)
                    )

                q1 = qf1(batch.observations)
                q2 = qf2(batch.observations)
                q1_val = q1.gather(1, batch.actions.long()).view(-1)
                q2_val = q2.gather(1, batch.actions.long()).view(-1)
                loss_q = F.mse_loss(q1_val, target) + F.mse_loss(q2_val, target)

                q_optimizer.zero_grad()
                loss_q.backward()
                q_optimizer.step()

                # actor update
                with torch.no_grad():
                    q1 = qf1(batch.observations)
                    q2 = qf2(batch.observations)
                    min_q = torch.min(q1, q2)
                    _, logp, probs = actor.get_action(batch.observations)
                loss_pi = (probs * (alpha_used * logp - min_q)).mean()

                actor_optimizer.zero_grad()
                loss_pi.backward()
                actor_optimizer.step()

                # target network update
                if global_step % args.target_network_frequency == 0:
                    for p, tp in zip(qf1.parameters(), qf1_target.parameters()):
                        tp.data.copy_(args.tau * p.data + (1 - args.tau) * tp.data)
                    for p, tp in zip(qf2.parameters(), qf2_target.parameters()):
                        tp.data.copy_(args.tau * p.data + (1 - args.tau) * tp.data)

            # 3) Bet‐and‐Secure phase transitions
            if global_step > 0 and global_step % args.bet_window == 0:
                R = cumulative_reward
                if R > R_ref:
                    if phase == "DECREASE":
                        phase = "INCREASE"
                else:
                    if phase == "INCREASE":
                        phase = "RECALIBRATE"
                    elif phase == "DECREASE":
                        phase = "INCREASE"
                R_ref = R
                cumulative_reward = 0.0

                if phase == "RECALIBRATE":
                    # wash‐out with alpha=0
                    for _ in range(args.recalib_steps):
                        if len(rb) < args.batch_size:
                            break
                        batch = rb.sample(args.batch_size)
                        with torch.no_grad():
                            _, np_logp, np_probs = actor.get_action(batch.next_observations)
                            q1_t = qf1_target(batch.next_observations)
                            q2_t = qf2_target(batch.next_observations)
                            min_q_t = torch.min(q1_t, q2_t)
                            target = batch.rewards.flatten() + (1 - batch.dones.flatten()) * args.gamma * (
                                (np_probs * min_q_t).sum(dim=1)
                            )
                        q1 = qf1(batch.observations)
                        q2 = qf2(batch.observations)
                        q1_val = q1.gather(1, batch.actions.long()).view(-1)
                        q2_val = q2.gather(1, batch.actions.long()).view(-1)
                        loss_q = F.mse_loss(q1_val, target) + F.mse_loss(q2_val, target)
                        q_optimizer.zero_grad()
                        loss_q.backward()
                        q_optimizer.step()
                        # soft update
                        for p, tp in zip(qf1.parameters(), qf1_target.parameters()):
                            tp.data.copy_(args.tau * p.data + (1 - args.tau) * tp.data)
                        for p, tp in zip(qf2.parameters(), qf2_target.parameters()):
                            tp.data.copy_(args.tau * p.data + (1 - args.tau) * tp.data)
                    phase = "DECREASE"

            # logging
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

        envs.close()
        # save actor
        path = os.path.join(writer.log_dir, f"actor_seed{current_seed}.pt")
        torch.save(actor.state_dict(), path)
        print(f"Saved actor to {path}")

    writer.close()
