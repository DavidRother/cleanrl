import os
import random
import shutil
import time
from collections import deque
from dataclasses import dataclass
import datetime

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import tqdm
import minatar.gym
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer, ReplayBufferSamples
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    exp_notes: str = "additional information"
    """notes of this experiment"""
    seed: int = 1
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
    env_id: str = "MinAtar/Freeway-v1"
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
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    target_entropy_scale: float = 0.89
    """coefficient for scaling the autotune entropy target"""
    alpha_eps: float = 2e-2
    """a small epsilon added for adjusting metrics"""
    n_runs: int = 10

    parent_folder: str = ""
    """parent folder to store the data"""

    use_cpc: bool = True
    """whether to use the cpc loss"""
    temperature: float = 2.0  # low peak, high flat
    """the (inverse) temperature for cpc"""
    action_scale: float = 0.2  #
    """the action projection scaling"""
    use_noise: bool = True
    """whether to use noise in the cpc loss"""
    noise_width: float = 0.2  #
    """the noise width/scale"""
    normalize: bool = True
    """normalize embeddings for similarity"""
    print_steps: int = 1e5
    """print intervall, timesteps"""
    target_entropy_start_exploitation: float = 0.50
    """coefficient for scaling the autotune entropy target"""
    target_entropy_end_exploitation: float = 0.80


class ChannelFirstWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        assert len(obs_shape) == 3, "Expected 3D observation (H, W, C)"
        c, h, w = obs_shape[2], obs_shape[0], obs_shape[1]
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(c, h, w), dtype=self.observation_space.dtype)

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


device_global = None


def masked_logsoftmax(vec, mask, dim=1, epsilon=1e-8, mask_exp=True, inv_temperature=1.0):

    vec = vec * inv_temperature

    if mask is not None and not mask_exp:
        vec = vec * mask.float()

    vec = vec - torch.max(vec, dim=dim, keepdim=True)[0]
    exps = torch.exp(vec)

    # eye = torch.eye(exps.shape[0]).to(device_global)
    # negs = exps * (1.0 - eye)
    # negs_quant = torch.quantile(negs.detach(), q=0.8, dim=1, keepdim=True)
    # mask = (exps.detach() > negs_quant).float().fill_diagonal_(1.0)
    # mask = exps.fill_diagonal_(1.0).detach()
    # mask = torch.log1p(exps).fill_diagonal_(1).detach()

    if mask is not None and mask_exp:
        exps = exps * mask.float()
    exps = torch.where(exps < epsilon, 0, exps)
    masked_sums = exps.sum(dim, keepdim=True)

    out = torch.log(exps + epsilon) - torch.log(masked_sums + epsilon)

    return out


def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().clamp(min=1e-8).mean().log()


last_image = 0


def cpc_loss_fnc(layer, activation, data, fixed_projection, use_noise, normalize, env_step, device, in_dists, layer_id, tb_writer):
    global last_image
    single_flag = False
    full_noise = True

    lower = 1.0 - args.noise_width  # 0.8
    upper = 1.0 + args.noise_width  # 1.2
    # start_low = 0.95
    # end_low = 0.6
    # start_high = 1.05
    # end_high = 1.4
    # frac = env_step / args.total_timesteps
    # lower = start_low + (end_low - start_low) * frac
    # upper = start_high + (end_high - start_high) * frac

    N = data[0].shape[0]
    D = data[0].shape[1]
    # D = fixed_projection.shape[1]

    if layer_id == 0:
        # import torchvision.transforms as transforms

        current_state = torch.as_tensor(data.observations, dtype=torch.float32, device=device)
        next_state = torch.as_tensor(data.next_observations, dtype=torch.float32, device=device)
        actions = F.one_hot(data[1].squeeze(1), fixed_projection.shape[0]).to(device_global).float()

        next_state_noise = next_state
        current_state_noise = current_state
    else:

        if single_flag:
            if full_noise:
                noise1 = np.random.uniform(lower, upper, N).reshape(-1, 1)
                # noise2 = np.random.uniform(lower, upper, N).reshape(-1, 1)
            else:
                noise1 = np.random.uniform(lower, upper, 1).reshape(1, 1)
                # noise2 = np.random.uniform(lower, upper, 1).reshape(1, 1)
        else:
            if full_noise:
                noise1 = np.random.uniform(lower, upper, N * D).reshape(N, -1)
                # noise2 = np.random.uniform(lower, upper, N * D).reshape(N, -1)
                # noise1 = np.random.randn(N, D).reshape(N, -1) + 0.2
            else:
                noise1 = np.random.uniform(lower, upper, D).reshape(1, D)  # dim noise
                # noise1 = np.random.uniform(lower, upper, N).reshape(N, 1)  # batch noise
                # noise2 = np.random.uniform(lower, upper, D).reshape(1, D)

        noise1 = torch.as_tensor(noise1).float().to(device)
        # noise2 = torch.as_tensor(noise2).float().to(device)

        current_state = data[0]
        next_state = data[2]
        # actions = data[1].squeeze(1)
        actions = F.one_hot(data[1].squeeze(1), fixed_projection.shape[0]).to(device_global).float()
        values = data[3]

        if use_noise:
            next_state_noise = next_state * noise1
            current_state_noise = current_state * noise1
        else:
            next_state_noise = next_state
            current_state_noise = current_state

    target_state_raw_pre = layer(next_state_noise)
    predicted_state_raw_pre = layer(current_state_noise)
    # combined = torch.concat([target_state_raw_pre, predicted_state_raw_pre], dim=0)
    # added_min = torch.min(combined, dim=0, keepdim=True)[0].clamp(max=0)
    # added_min = torch.quantile(combined, q=0.1, dim=0, keepdim=True)  # .detach()
    # target_state_raw_pre = target_state_raw_pre + added_min
    # predicted_state_raw_pre = predicted_state_raw_pre + added_min
    # target_state_raw_pre = target_state_raw_pre + torch.quantile(target_state_raw_pre, q=0.8, dim=0, keepdim=True).detach()
    # predicted_state_raw_pre = predicted_state_raw_pre + torch.quantile(predicted_state_raw_pre, q=0.8, dim=0, keepdim=True).detach()
    target_state_raw = activation(target_state_raw_pre)
    predicted_state_raw = activation(predicted_state_raw_pre)

    # target_state_raw = (target_state_raw - torch.mean(target_state_raw, dim=1, keepdim=True).detach()) / (torch.std(target_state_raw, dim=1, keepdim=True).detach() + 1e-12)
    # predicted_state_raw = (predicted_state_raw - torch.mean(predicted_state_raw, dim=1, keepdim=True).detach()) / (
    #     torch.std(predicted_state_raw, dim=1, keepdim=True).detach() + 1e-12
    # )

    if normalize:
        target_state = F.normalize(target_state_raw, dim=1)
        predicted_state = F.normalize(predicted_state_raw, dim=1)
        # predicted_state = predicted_state_raw
    else:
        target_state = target_state_raw
        predicted_state = predicted_state_raw

    layer_out_next = target_state_raw
    layer_out_current = predicted_state_raw

    if use_noise:
        layer_out_current = activation(layer(current_state))
        layer_out_next = activation(layer(next_state))

    tmp = args.temperature

    # start_tmp = 0.5
    # end_tmp = 10
    # cycle_decay = 1.0
    # cycle_length = args.total_timesteps // 1
    # cycle = env_step // cycle_length
    # eff_step = env_step - (cycle_length * cycle)
    # frac = eff_step / cycle_length
    # start_cycle = start_tmp * cycle_decay**cycle
    # tmp = start_cycle + (end_tmp - start_cycle) * frac

    temperature_inv = 1 / tmp

    action_noise = actions @ fixed_projection
    action_noise = action_noise**2
    # action_noise = torch.abs(action_noise)
    # action_noise = F.normalize(action_noise, dim=1)
    # action_noise = F.relu(action_noise)
    predicted_state_action = predicted_state + action_noise

    with torch.no_grad():
        raw_cos = F.cosine_similarity(predicted_state, target_state, dim=1).mean()
        tb_writer.add_scalar(f"metrics/cpc_raw_pos_cos_{layer_id}", raw_cos.item(), env_step)
    # predicted_state_action = F.normalize(predicted_state_action, dim=1)

    # active_mask = ((predicted_state > 0).float() * (target_state > 0).float()).clamp(max=1.0).detach()
    # predicted_state_action = predicted_state_action * active_mask
    # target_state = target_state * active_mask
    all_sims = predicted_state_action @ target_state.mT

    labels = torch.arange(all_sims.shape[0]).to(device_global)
    neg_weights = None
    # neg_weights = in_dists
    sm = masked_logsoftmax(all_sims, neg_weights, mask_exp=True, inv_temperature=temperature_inv)
    loss = F.nll_loss(sm, labels, reduction="none").mean()
    # loss = F.cross_entropy(all_sims, labels, label_smoothing=0.001)

    # sim_w = 0.0001
    # sim_mean = (all_sims * (1.0 - torch.eye(all_sims.shape[0]).to(device_global))).mean()
    # loss = loss - sim_w * sim_mean

    # ent_w = 0.1
    # entropy = torch.distributions.Categorical(logits=all_sims).entropy().mean()
    # loss = loss - entropy * ent_w

    # labels_uni = torch.ones_like(all_sims).to(device_global) / all_sims.shape[1]
    # loss_uni = F.cross_entropy(all_sims * temperature_inv, labels_uni, reduction="none").mean()
    # loss = loss + ent_w * loss_uni

    # uni/align loss
    # al_uni_w = 0.1
    # al_loss = align_loss(target_state, predicted_state_action)
    # uni_loss = uniform_loss(target_state, t=2)
    # loss = loss + (al_loss + uni_loss * 1.5) * al_uni_w

    # avg_dists = torch.cdist(predicted_state_action, target_state)
    # avg_dists = torch.mean(avg_dists)
    # loss = loss + avg_dists * 1e-4

    # L1/L2 reg below target
    # reg_weight = 1e-1
    # target_value = 0.0
    # lambd = 1e-0
    # scale = 3e-1
    # beta = 5.0
    # combined_activities = predicted_state_raw_pre
    # combined_activities = torch.concat([predicted_state_raw_pre, target_state_raw_pre], dim=0)
    # reg_loss = (lambd / beta) * F.softplus(beta * (target_value - combined_activities))
    # reg_loss = lambd * torch.exp(-(combined_activities - target_value) / scale)
    # reg_loss = reg_loss.mean(1).mean()
    # combined_activities = torch.abs(combined_activities)
    # combined_activities = torch.pow(combined_activities, 2)
    # reg_loss = F.relu(target_value - combined_activities).sum(dim=1).mean()
    # reg_loss = combined_activities.mean()
    # if layer_id == 0:
    #     reg_loss = layer[0].weight.pow(2).mean()
    # else:
    #     reg_loss = layer.weight.pow(2).mean()

    # firing rate nudge
    # reg_weight = 1e-2
    # beta = 1.0
    # p_min = 0.1
    # p_max = 0.8
    # pre_acts = F.sigmoid(combined_activities * beta).mean(0)
    # reg_loss = (F.relu(p_min - pre_acts) + F.relu(pre_acts - p_max)).mean()

    # variance floor
    # reg_weight = 1e-0
    # reg_loss = F.relu(1.0 - torch.std(combined_activities, dim=0)).mean()

    # loss = loss + reg_loss * reg_weight

    zero_acts = (layer_out_current <= 0).float().mean(1).mean()
    tb_writer.add_scalar(f"charts/zeroact_{layer_id}", zero_acts, global_step=env_step)

    # img_cycle = env_step // 25000
    # if img_cycle >= last_image:
    #     last_image = img_cycle
    #     fig, ax = plt.subplots(figsize=(4, 3), dpi=200)
    #     ax.matshow(layer_out_current.detach().cpu())
    #     fig.tight_layout()
    #     tb_writer.add_figure(f"rep_{layer_id}", fig, global_step=env_step, close=True)

    # zero_penalty = torch.where(predicted_state_raw <= 0, predicted_state_raw_pre, 0).pow(2).mean(1).mean()
    # zero_w = 1e-3
    # zero_target = 0.6 if layer_id == 0 else 0.9
    # zero_loss = (zero_target - zero_penalty).pow(2)
    # loss = loss + zero_w * zero_loss
    # loss = loss - zero_w * zero_penalty

    with torch.no_grad():
        # cosine of positive pairs
        pos_cos = F.cosine_similarity(predicted_state_action, target_state, dim=1).mean()

        # margin: positive logit minus hardest negative logit
        B = all_sims.size(0)
        diag = torch.arange(B, device=all_sims.device)
        pos_logits = all_sims[diag, diag]
        neg_mask = ~torch.eye(B, dtype=torch.bool, device=all_sims.device)
        max_neg_logits, _ = all_sims.masked_fill(~neg_mask, float("-inf")).max(dim=1)
        margin = (pos_logits - max_neg_logits).mean()

        tb_writer.add_scalar(f"metrics/cpc_pos_cos_{layer_id}", pos_cos.item(), global_step=env_step)
        tb_writer.add_scalar(f"metrics/cpc_pos_minus_maxneg_{layer_id}", margin.item(), global_step=env_step)

    return loss, layer_out_current, layer_out_next


def cpc_wrapper(model, data, use_noise, normalize, env_step, device, tb_writer):
    in_dists = None

    # next_state = torch.as_tensor(data.next_observations, dtype=torch.float32, device=device).flatten(start_dim=1)
    # in_dists = torch.cdist(next_state, next_state)
    # in_dists = in_dists / (torch.max(in_dists, dim=1, keepdim=True)[0] + 1e-8)
    # margin = 0.2
    # margin_low = torch.quantile(in_dists, q=0.2, dim=1, keepdim=True)
    # margin_high = torch.quantile(in_dists, q=0.95, dim=1, keepdim=True)
    # in_dists = torch.where(in_dists < margin_low, 0.0, 1.0) * torch.where(in_dists > 1.0 - margin_high, 0.0, 1.0)
    # in_dists = in_dists.fill_diagonal_(1.0).detach()
    # in_dists = 1.0 - in_dists
    # in_dists = torch.exp(in_dists)
    # in_dists = 1 / (1 + in_dists)
    # in_dists = torch.exp(-in_dists / 0.1).detach()

    loss1, current_state, next_state = cpc_loss_fnc(model.conv, model.act_fnc, data, model.fixed_projection_conv, use_noise, normalize, env_step, device, in_dists, 0, tb_writer)

    # current_state = torch.as_tensor(data.observations, dtype=torch.float32, device=device)
    # next_state = torch.as_tensor(data.next_observations, dtype=torch.float32, device=device)
    # current_state = model.act_fnc(model.conv(current_state))
    # next_state = model.act_fnc(model.conv(next_state))

    new_data = ReplayBufferSamples(
        observations=current_state,
        next_observations=next_state,
        actions=data.actions,
        dones=data.dones,
        rewards=data.rewards,
    )
    loss2, _, _ = cpc_loss_fnc(model.fc1, model.act_fnc, new_data, model.fixed_projection1, use_noise, normalize, env_step, device, in_dists, 1, tb_writer)

    schedule = False
    if schedule:
        start_tmp = 0.01
        end_tmp = 0.5
        cycle_decay = 1.0
        cycle_length = args.total_timesteps // 1
        cycle = env_step // cycle_length
        eff_step = env_step - (cycle_length * cycle)
        frac = eff_step / cycle_length
        start_cycle = start_tmp * cycle_decay**cycle
        weight = start_cycle + (end_tmp - start_cycle) * frac
    else:
        weight = 0.5
        # cycle_length = args.total_timesteps // 3
        # cycle = env_step // cycle_length
        # if cycle == 0:
        # if cycle % 2 == 0:
        #     weight = 0.5  # * 0.9**cycle
        # else:
        #     weight = 0.0
    loss = (loss1 + loss2) * weight
    # loss = weight * loss1 + (1.0 - weight) * loss2

    return loss


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


class L2NormalizationLayer(nn.Module):
    def __init__(self, dim=1, eps=1e-12):
        super(L2NormalizationLayer, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim, eps=self.eps)


class Actor(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[0], 16, kernel_size=3, stride=1)),
            nn.Flatten(),
        )
        self.norm1 = L2NormalizationLayer(dim=1, eps=1e-12)
        self.norm2 = L2NormalizationLayer(dim=1, eps=1e-12)

        with torch.inference_mode():
            output_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]

        fc_dim = 128

        self.fc1 = layer_init(nn.Linear(output_dim, fc_dim))
        self.fc_logits = layer_init(nn.Linear(fc_dim, envs.single_action_space.n))

        scale = args.action_scale
        self.fixed_projection1 = torch.randn(np.prod(envs.single_action_space.n), fc_dim).to(device_global) * scale
        self.fixed_projection_conv = torch.randn(np.prod(envs.single_action_space.n), output_dim).to(device_global) * scale

        # act_prob = 0.5
        # self.fixed_projection1 = (torch.rand(np.prod(envs.single_action_space.n), fc_dim) > act_prob).float().to(device_global)
        # self.fixed_projection_conv = (torch.rand(np.prod(envs.single_action_space.n), output_dim) > output_dim).float().to(device_global)

        # min_s = -5.0
        # max_s = 5.0
        # self.fixed_projection1 = torch.FloatTensor(np.prod(envs.single_action_space.n), fc_dim).uniform_(min_s, max_s).to(device_global)
        # self.fixed_projection_conv = torch.FloatTensor(np.prod(envs.single_action_space.n), output_dim).uniform_(min_s, max_s).to(device_global)

        self.act_fnc = F.relu

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=device)
        x = self.act_fnc(self.conv(x))
        # x = self.norm1(x)
        x = self.act_fnc(self.fc1(x))
        # x = self.norm2(x)
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


def target_entropy_from_exploitation_prob(exploitation_prob, num_actions):
    ent = -(exploitation_prob * np.log(exploitation_prob) + (1.0 - exploitation_prob) * np.log((1.0 - exploitation_prob) / (num_actions - 1.0)))
    return ent


def target_entropy_from_exploitation_probability(p, n):
    if p <= 0 or p >= 1:
        raise ValueError("Exploitation probability p must be in the open interval (0, 1).")

    # Compute the entropy of the distribution.
    ent = - (p * math.log(p) + (1 - p) * math.log((1 - p) / (n - 1)))

    # Return the SAC-style target entropy (i.e., negative of the computed entropy).
    return ent


def log_svals(tag, Z, topk=10):
    Zc = Z - Z.mean(dim=0, keepdim=True)
    # small epsilon to avoid NaNs if batch is small/degenerate
    if Zc.shape[0] >= 2:
        S = torch.linalg.svdvals(Zc)
        top = min(topk, S.numel())
        for i in range(top):
            writer.add_scalar(f"rep/{tag}_sv{i + 1}", S[i].item(), global_step)
    writer.add_scalar(f"rep/{tag}_feature_var_mean", Zc.var(dim=0).mean().item(), global_step)


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1" "gymnasium[atari,accept-rom-license]==0.28.1"  "ale-py==0.8.1" 
"""
        )
    args = tyro.cli(Args)

    current_time = datetime.datetime.now()
    current_time = current_time.strftime("%Y-%m-%d_%H-%M")
    setting_name = "sac"
    if args.use_cpc:
        setting_name += "-cpc"
    env_name = args.env_id
    env_name = env_name.split("/")[-1]
    run_name = f"{env_name}__{setting_name}__{os.uname()[1]}__{current_time}__{args.seed}"
    print(f"*******\n{run_name}\n*******")

    writer = SummaryWriter(f"runs_dt_method/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    minatar.gym.register_envs()
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    device_global = device

    current_seed = args.seed
    run_prefix = f"seed_{current_seed}"
    # folder_name = f"{env_name}__{args.exp_name}"
    # run_name = f"folder_name__{run_prefix}__{int(time.time())}"
    print(f"Starting run: {run_name}")

    # (Re)seed randomness for current run
    random.seed(current_seed)
    np.random.seed(current_seed)
    torch.manual_seed(current_seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Set up the vectorized environment.
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, current_seed, 0, args.capture_video, run_name)])
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
        # target_entropy = -args.target_entropy_scale * torch.log(1 / torch.tensor(envs.single_action_space.n))
        target_entropy_start = target_entropy_from_exploitation_prob(0.5, envs.single_action_space.n)
        target_entropy_end = target_entropy_from_exploitation_prob(0.8, envs.single_action_space.n)
        # target_entropy_start2 = target_entropy_from_exploitation_probability(args.target_entropy_start_exploitation,
        #                                                                      envs.single_action_space.n)
        # target_entropy_end2 = target_entropy_from_exploitation_probability(args.target_entropy_end_exploitation,
        #                                                                    envs.single_action_space.n)
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

    # progress_bar = tqdm.trange(args.total_timesteps, desc=f"Training {run_prefix}", dynamic_ncols=True)
    latest_return = None
    episode_returns = []
    episodic_lengths = []
    avg_return_normalised = alpha
    num_actions = envs.single_action_space.n
    print(f"num_actions: {num_actions}")

    action_counts = np.zeros(num_actions, dtype=np.int64)

    lowest_return = np.inf

    alpha_eps = args.alpha_eps

    scores_window = deque(maxlen=50)
    best_avg_score = -np.Inf
    best_avg_score_step = 0
    last_print = 0
    obs, _ = envs.reset(seed=current_seed)
    mean_cpc = None
    # for global_step in progress_bar:
    for global_step in range(args.total_timesteps):
        # Action selection
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        episode_start = np.logical_or(terminations, truncations)

        # writer.add_scalar(f"{run_prefix}/charts/reward", rewards[0], global_step)
        # writer.add_scalar(f"{run_prefix}/charts/terminations", terminations[0], global_step)
        # writer.add_scalar(f"{run_prefix}/charts/truncations", truncations[0], global_step)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "episode" in infos:
            # episodic_return = infos["episode"]["r"]
            # episodic_length = infos["episode"]["l"]
            # latest_return = episodic_return
            # writer.add_scalar(f"{run_prefix}/charts/episodic_return", episodic_return, global_step)
            # writer.add_scalar(f"{run_prefix}/charts/episodic_length", episodic_length, global_step)
            writer.add_scalar("charts/episodic_return", infos["episode"]["r"], global_step)
            writer.add_scalar("charts/episodic_length", infos["episode"]["l"], global_step)
            scores_window.append(infos["episode"]["r"])

            # episode_returns.append(episodic_return)
            # episodic_lengths.append(episodic_length)

            if len(scores_window) > 0:
                mean_score = np.mean(scores_window)
                if mean_score > best_avg_score:
                    best_avg_score = mean_score
                    best_avg_score_step = global_step
                print_step = global_step // args.print_steps
                if print_step > last_print:
                    last_print = print_step
                    print(
                        f"{global_step:1.1e}/{args.total_timesteps:g}"
                        f"\tAvg.Score: {mean_score:.2f} +- {np.std(scores_window):.2f} (max: {np.max(scores_window):.2f} best: {best_avg_score:.2f} {best_avg_score_step:1.2e})"
                    )

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        rb.add(obs, next_obs, actions, rewards, terminations, infos)

        if episode_start:
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # Training updates.
        if global_step > args.learning_starts:
            if global_step % args.update_frequency == 0:
                data = rb.sample(args.batch_size)

                with torch.no_grad():
                    x = data.observations.float()
                    eps = 1e-3
                    delta = torch.randn_like(x) * eps

                    # Conv features
                    conv_x = actor.act_fnc(actor.conv(x))
                    conv_x_eps = actor.act_fnc(actor.conv(x + delta))
                    conv_x_n = F.normalize(conv_x, dim=1)
                    conv_x_epsn = F.normalize(conv_x_eps, dim=1)
                    local_cos_conv = F.cosine_similarity(conv_x_n, conv_x_epsn, dim=1).mean().item()
                    writer.add_scalar("metrics/local_cos_eps1e-3_conv", local_cos_conv, global_step)

                    # fc1 features (pass through conv first)
                    fc_x = actor.act_fnc(actor.fc1(conv_x))
                    fc_x_eps = actor.act_fnc(actor.fc1(conv_x_eps))
                    fc_x_n = F.normalize(fc_x, dim=1)
                    fc_x_epsn = F.normalize(fc_x_eps, dim=1)
                    local_cos_fc1 = F.cosine_similarity(fc_x_n, fc_x_epsn, dim=1).mean().item()
                    writer.add_scalar("metrics/local_cos_eps1e-3_fc1", local_cos_fc1, global_step)

                if global_step % 1000 == 0:
                    with torch.no_grad():
                        log_svals("conv", conv_x)
                        log_svals("fc1", fc_x)

                _, log_pi, action_probs = actor.get_action(data.observations)
                policy_dist = Categorical(probs=action_probs)
                entropy = policy_dist.entropy().mean().item()

                alpha_used = alpha
                with torch.no_grad():
                    _, next_state_log_pi, next_state_action_probs = actor.get_action(data.next_observations)
                    qf1_next_target = qf1_target(data.next_observations)
                    qf2_next_target = qf2_target(data.next_observations)
                    min_qf_next_target = next_state_action_probs * (torch.min(qf1_next_target, qf2_next_target) - alpha_used * next_state_log_pi)
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

                #################################################################
                # ADD CPC LOSS
                #################################################################
                if args.use_cpc:
                    cpc_loss = cpc_wrapper(
                        model=actor,
                        data=data,
                        use_noise=args.use_noise,
                        normalize=args.normalize,
                        env_step=global_step,
                        device=device,
                        tb_writer=writer,
                    )

                    # if mean_cpc is None:
                    #     mean_cpc = cpc_loss.detach() * 0.9
                    # else:
                    #     mean_tau = 0.5
                    #     mean_cpc = mean_tau * cpc_loss.detach() + (1 - mean_tau) * mean_cpc
                    # mean_cpc = torch.maximum(mean_cpc, cpc_loss.detach() * 0.9)
                    # cpc_loss_scaled = cpc_loss / mean_cpc.detach()
                    # cpc_loss_scaled = cpc_loss * (torch.abs(actor_loss) / cpc_loss).detach()

                    # cycle_length = args.total_timesteps // 8
                    # cycle = global_step // cycle_length

                    # if cycle % 2 == 0:
                    #     combined_loss = actor_loss + cpc_loss
                    # else:
                    #     combined_loss = actor_loss

                    combined_loss = actor_loss + cpc_loss  # _scaled
                else:
                    combined_loss = actor_loss
                #################################################################

                actor_optimizer.zero_grad()
                combined_loss.backward()
                actor_optimizer.step()

                progress_ratio = min(global_step / args.total_timesteps, 1.0)
                target_entropy = target_entropy_start + progress_ratio * (target_entropy_end - target_entropy_start)
                writer.add_scalar("losses/current_target_entropy", target_entropy, global_step)

                if args.autotune:
                    with torch.no_grad():
                        log_alpha.copy_(torch.log(torch.as_tensor(alpha_used, device=device)))
                    alpha_loss = (action_probs.detach() * (-log_alpha.exp() * (log_pi + target_entropy).detach())).mean()
                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

                # writer.add_scalar(f"{run_prefix}/residuals/primal_feasibility", primal_residual, global_step)
                # writer.add_scalar(f"{run_prefix}/residuals/dual_feasibility", dual_residual, global_step)
                # writer.add_scalar(f"{run_prefix}/residuals/stationarity", stationarity_residual, global_step)
                # writer.add_scalar(f"{run_prefix}/residuals/complementary_slackness", complementary_slackness,
                #                   global_step)
                # writer.add_scalar(f"{run_prefix}/losses/qf1_values", qf1_a_values.mean().item(), global_step)
                # writer.add_scalar(f"{run_prefix}/losses/qf2_values", qf2_a_values.mean().item(), global_step)
                # writer.add_scalar(f"{run_prefix}/losses/qf1_loss", qf1_loss.item(), global_step)
                # writer.add_scalar(f"{run_prefix}/losses/qf2_loss", qf2_loss.item(), global_step)
                # writer.add_scalar(f"{run_prefix}/losses/qf_loss", qf_loss.item() / 2.0, global_step)
                if global_step % 1000 == 0:
                    writer.add_scalar(f"losses/actor_loss", actor_loss.item(), global_step)
                    if args.use_cpc:
                        writer.add_scalar("losses/cpc_loss", cpc_loss.item(), global_step)
                    # writer.add_scalar(f"{run_prefix}/losses/q_variance", q_var, global_step)
                    # writer.add_scalar(f"{run_prefix}/losses/alpha", alpha, global_step)
                    # writer.add_scalar(f"{run_prefix}/losses/alpha_used", alpha_used, global_step)
                    sps = int(global_step / (time.time() - start_time))
                    writer.add_scalar(f"charts/SPS", sps, global_step)
                    writer.add_scalar(f"charts/mean_policy_entropy", entropy, global_step)
                    # if args.autotune:
                    #     writer.add_scalar(f"{run_prefix}/losses/alpha_loss", alpha_loss.item(), global_step)
                    # actions_in_window = action_counts.sum()
                    # if actions_in_window:  # avoid division by zero
                    #     freq_window = action_counts / actions_in_window
                    #     for idx, freq in enumerate(freq_window):
                    #         writer.add_scalar(f"{run_prefix}/metrics/a{idx}", freq, global_step)

            # Update the target networks.
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            # if global_step % 100 == 0:
            #     progress_bar.set_postfix({
            #         "step": global_step,
            #         "return": f"{float(latest_return):.2f}" if latest_return is not None else "N/A",
            #         "sps": sps
            #     })

    # End of training for this run.

    mean_score = np.mean(scores_window)
    print(
        f"{global_step:1.1e}/{args.total_timesteps:g}"
        f"\tAvg.Score: {mean_score:.2f} +- {np.std(scores_window):.2f} (max: {np.max(scores_window):.2f} best: {best_avg_score:.2f} {best_avg_score_step:1.2e})\n"
    )

    envs.close()

    # Save the final actor model into the TensorBoard folder.
    # model_save_path = os.path.join(writer.log_dir, f"final_model_{run_prefix}.pt")
    # torch.save(actor.state_dict(), model_save_path)
    # print(f"Saved final model for {run_prefix} to {model_save_path}")

    writer.close()
