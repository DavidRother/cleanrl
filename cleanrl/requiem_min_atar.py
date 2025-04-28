# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy
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
import math
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer
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
    env_id: str = "MinAtar/SpaceInvaders-v1"
    """the id of the environment"""
    total_timesteps: int = 5000000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
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
    alpha_start = 0.04
    alpha_end = 0.01  # softmax temperature
    delta_start = 0.6  # very exploratory at the beginning
    delta_end = 0.99999  # almost deterministic by the end
    delta_fraction = 0.8  # finish annealing after 80 % of training


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
class QNetwork(nn.Module):
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


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1" "gymnasium[atari,accept-rom-license]==0.28.1"  "ale-py==0.8.1" 
"""
        )
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
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
    writer = SummaryWriter(f"runs_soft_q/{run_name}")
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
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    start_time = time.time()
    progress_bar = tqdm.trange(args.total_timesteps, desc="Training", dynamic_ncols=True)
    latest_return = None
    episode_returns = []
    episodic_lengths = []
    buffer_size = args.buffer_size
    num_actions = envs.single_action_space.n

    alpha = args.alpha_start  # softmax temperature
    alpha_start = args.alpha_start
    alpha_end = args.alpha_end
    delta_start = kl_categorical_vs_uniform(args.delta_start, num_actions)
    delta_end = kl_categorical_vs_uniform(args.delta_end, envs.single_action_space.n)
    delta_fraction = args.delta_fraction  # finish annealing after 80 % of training
    print(f"KL start bound: {delta_start}")
    print(f"KL end bound: {delta_end}")

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in progress_bar:
        # ALGO LOGIC: put action logic here
        delta_t = min(delta_end,
                      delta_start + (delta_end - delta_start) * min(1.0, global_step /
                                                                    (delta_fraction * args.total_timesteps)))
        alpha = max(alpha_end, delta_start + (alpha_end - alpha_start) * min(1.0, global_step /
                                                                             (delta_fraction * args.total_timesteps)))

        q_values = q_network(torch.Tensor(obs).to(device))
        # Sample from the softmax policy induced by Q/alpha
        probs = torch.softmax(q_values / alpha, dim=1)
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample().cpu().numpy()

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
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        num_kl_steps = 0
        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                q_pred = q_network(data.observations)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_pred.gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                q_pred = q_network(data.observations)
                while not kl_close_enough(q_pred, delta_t, alpha):
                    kl_loss = kl_penalty(q_pred, delta_t, alpha)
                    optimizer.zero_grad()
                    kl_loss.backward()
                    optimizer.step()
                    q_pred = q_network(data.observations)
                    num_kl_steps += 1

                probs = torch.softmax(q_values / alpha, dim=1)
                dist = torch.distributions.Categorical(probs)
                entropy = dist.entropy().mean().item()

                if (global_step % (args.train_frequency * 25)) == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    sps = int(global_step / (time.time() - start_time))
                    writer.add_scalar("charts/SPS", sps, global_step)
                    writer.add_scalar("charts/entropy", entropy, global_step)
                    writer.add_scalar("charts/delta", delta_t, global_step)
                    writer.add_scalar("charts/kl_optimizations", num_kl_steps, global_step)

                    progress_bar.set_postfix({
                        "step": global_step,
                        "return": f"{float(latest_return):.2f}" if latest_return is not None else "N/A",
                        "sps": sps
                    })

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

    if args.save_model:
        model_path = f"runs_old/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.dqn_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=0.05,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "DQN", f"runs_old/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
