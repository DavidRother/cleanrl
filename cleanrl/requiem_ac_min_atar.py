# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_ataripy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import math
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
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "MinAtar/Freeway-v1"
    """the id of the environment"""
    total_timesteps: int = 3000000
    """total timesteps of the experiments"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 100000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 1000
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    learning_starts: int = 20000
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training"""
    alpha: float = 0.02  # softmax temperature
    delta_start: float = 0.75  # very exploratory at the beginning
    delta_end: float = 0.99999  # almost deterministic by the end
    delta_fraction: float = 0.7  # finish annealing after 80 % of training
    lambda_lr: float = 1e-3            # step size for dual variable λ
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


def kl_penalty(q_vals: torch.Tensor, delta: float, alpha: float) -> torch.Tensor:
    """
    q_vals : (B, A) – Q-values from the network (requires_grad=True)
    delta  : float   – target KL(π || uniform)
    alpha  : float   – temperature used in the softmax   π = softmax(Q / α)
    returns a scalar hinge loss (0 if KL already ≤ δ)
    """
    B, A = q_vals.shape
    logA = torch.log(torch.tensor(float(A), device=q_vals.device, dtype=q_vals.dtype))
    probs = F.softmax(q_vals / alpha, dim=1)
    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1)        # (B,)
    kl_batch = logA - entropy                                       # (B,)
    penalty = torch.clamp(kl_batch - delta, min=0.0).mean()         # hinge
    return penalty


def kl_close_enough(
        q_vals: torch.Tensor,
        delta: float,
        alphaa: float,
        tol: float = 1e-4) -> bool:
    """
    Returns True if *all* states in the current batch have
    KL(π || uniform) ≤ delta + tol.

    q_vals : (B, A)  – raw Q values  (requires_grad=False is fine)
    delta  : float   – current KL target  (delta_t)
    alphaa  : float   – temperature used by the policy
    tol    : float   – small slack to avoid numerical issues
    """
    with torch.no_grad():
        B, A = q_vals.shape
        logA = math.log(A)
        probs = F.softmax(q_vals / alphaa, dim=1)
        entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1)     # (B,)
        kl = logA - entropy                                          # (B,)
        return torch.all(kl <= delta + tol).item()


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

poetry run pip install "stable_baselines3==2.0.0a1" "gymnasium[atari,accept-rom-license]==0.28.1"  "ale-py==0.8.1" 
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
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    lambda_param = torch.tensor(args.lambda_init, dtype=torch.float32, device=device)
    # TRY NOT TO MODIFY: eps=1e-4 increases numerical stability
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr, eps=1e-4)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr, eps=1e-4)

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    progress_bar = tqdm.trange(args.total_timesteps, desc="Training", dynamic_ncols=True)
    latest_return = None
    episode_returns = []
    episodic_lengths = []
    buffer_size = args.buffer_size
    num_actions = envs.single_action_space.n

    alpha = args.alpha  # softmax temperature
    delta_start = kl_categorical_vs_uniform(args.delta_start, num_actions)
    delta_end = kl_categorical_vs_uniform(args.delta_end, envs.single_action_space.n)
    delta_fraction = args.delta_fraction  # finish annealing after 80 % of training
    print(f"KL start bound: {delta_start}")
    print(f"KL end bound: {delta_end}")

    num_kl_steps = 0

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
                writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                writer.add_scalar("charts/episodic_length", episodic_length, global_step)

                episode_returns.append(episodic_return)
                episodic_lengths.append(episodic_length)
                if sum(episodic_lengths) > buffer_size:
                    episode_returns.pop(0)
                    episodic_lengths.pop(0)
                avg_return = sum(episode_returns) / len(episode_returns)
                writer.add_scalar("charts/episodic_return_avg", avg_return, global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, (trunc, term) in enumerate(zip(truncations, terminations)):
            if trunc or term:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                delta_t = min(delta_end,
                              delta_start + (delta_end - delta_start)
                              * min(1.0, global_step / (delta_fraction * args.total_timesteps)))
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

                B, A = qf1_values.shape
                logA = math.log(A)
                probs = F.softmax(qf1_values / args.alpha, dim=1)
                entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1)
                kl_batch = logA - entropy
                violation = torch.clamp(kl_batch - delta_t, min=0.0)
                hinge_mean_1 = violation.mean()
                qf1_kl_loss = lambda_param.detach() * hinge_mean_1

                probs = F.softmax(qf2_values / args.alpha, dim=1)
                entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1)
                kl_batch = logA - entropy
                violation = torch.clamp(kl_batch - delta_t, min=0.0)
                hinge_mean_2 = violation.mean()
                qf2_kl_loss = lambda_param.detach() * hinge_mean_2

                final_loss = qf_loss + qf2_kl_loss + qf1_kl_loss

                q_optimizer.zero_grad()
                final_loss.backward()
                q_optimizer.step()

                with torch.no_grad():
                    lambda_param += args.lambda_lr * (hinge_mean_1 + hinge_mean_2)
                    lambda_param.clamp_(min=0.0)

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

                if (global_step % (args.train_frequency * 25)) == 0:
                    writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                    writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                    writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                    writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                    writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                    writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                    writer.add_scalar("losses/alpha", alpha, global_step)
                    sps = int(global_step / (time.time() - start_time))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                    writer.add_scalar("charts/mean_policy_entropy", entropy, global_step)
                    writer.add_scalar("charts/delta", delta_t, global_step)
                    writer.add_scalar("charts/kl_mean", kl_batch.mean().item(), global_step)
                    writer.add_scalar("charts/lambda", lambda_param.item(), global_step)



                    progress_bar.set_postfix({
                        "step": global_step,
                        "return": f"{float(latest_return):.2f}" if latest_return is not None else "N/A",
                        "sps": sps
                    })

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

    envs.close()
    writer.close()
