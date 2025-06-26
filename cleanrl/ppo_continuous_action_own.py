# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import datetime
import os
import random
import shutil
import time
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from sympy.physics.control import bilinear
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from ema_pytorch import EMA
import matplotlib.pyplot as plt


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    exp_notes: str = "additional information"
    """notes of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
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

    parent_folder: str = ""
    """parent folder to store the data"""

    use_cpc: bool = True
    """whether to use the cpc loss"""
    temperature: float = 0.5  # low peak, high flat
    """the (inverse) temperature for cpc"""
    use_noise: bool = True
    """whether to use noise in the cpc loss"""
    normalize: bool = True
    """normalize embeddings for similarity"""
    print_steps: int = 1e5
    """print intervall, timesteps"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v5"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), env.observation_space)
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


device_global = None

# rnd_generator = torch.Generator()


def masked_logsoftmax(vec, mask, dim=1, epsilon=1e-8, mask_exp=True):
    if not mask_exp:
        vec = vec * mask.float()
        # vec = torch.where(vec > 0.0, vec * mask.float(), vec * (2 - mask.float()))
    vec = vec - torch.max(vec, dim=dim, keepdim=True)[0]  # .detach()
    exps = torch.exp(vec)
    if mask_exp:
        exps = exps * mask.float()
    exps = torch.where(exps < epsilon, 0, exps)
    masked_sums = exps.sum(dim, keepdim=True)

    # w = exps / masked_sums
    # w = w.detach().fill_diagonal_(1)
    # masked_sums = masked_sums * w

    # out = vec - torch.log(masked_sums + epsilon)
    out = torch.log(exps + epsilon) - torch.log(masked_sums + epsilon)

    return out


def masked_logstablemax(vec, mask, dim=1, epsilon=1e-8, mask_exp=True):
    if not mask_exp:
        vec = vec * mask.float()
    # vec = vec - torch.min(vec, dim=dim, keepdim=True)[0]
    exps = torch.where(vec >= 0, torch.log1p(vec), -torch.log1p(-vec))
    # exps = (1 + vec + vec.pow(2)) / 2
    if mask_exp:
        exps = exps * mask.float()
    # exps = torch.where(exps < epsilon, 0, exps)
    masked_sums = exps.sum(dim, keepdim=True)

    out = torch.log(exps + epsilon) - torch.log(masked_sums + epsilon)
    out = torch.clamp(out, min=epsilon)
    return out


print_counter = 0


def cpc_loss_fnc(layer, ema_layer, bilinear, activation, data, fixed_projection, use_noise, normalize, env_step, device, in_dists, layer_id):
    single_flag = False
    full_noise = True

    lower = 0.8
    upper = 1.2
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
            # noise1 = np.random.randn(N, D).reshape(N, -1)
        else:
            noise1 = np.random.uniform(lower, upper, D).reshape(1, D)  # dim noise
            # noise1 = np.random.uniform(lower, upper, N).reshape(N, 1)  # batch noise
            # noise2 = np.random.uniform(lower, upper, D).reshape(1, D)

    noise1 = torch.as_tensor(noise1).float().to(device)
    # noise2 = torch.as_tensor(noise2).float().to(device)

    current_state = data[0]
    next_state = data[2]
    actions = data[1].squeeze(1)
    values = data[3]

    if use_noise:
        next_state_noise = next_state * noise1
        current_state_noise = current_state * noise1
    else:
        next_state_noise = next_state
        current_state_noise = current_state

    target_state_raw = activation(layer(next_state_noise))
    predicted_state_raw = activation(layer(current_state_noise))

    if normalize:
        target_state = F.normalize(target_state_raw, dim=1)
        predicted_state = F.normalize(predicted_state_raw, dim=1)
    else:
        target_state = target_state_raw
        predicted_state = predicted_state_raw

    layer_out_next = target_state_raw
    layer_out_current = predicted_state_raw

    if use_noise:
        layer_out_current = activation(layer(current_state))
        layer_out_next = activation(layer(next_state))

    tmp = args.temperature
    # start_tmp = 1.0
    # end_tmp = 0.1
    # frac = env_step / args.total_timesteps
    # tmp = start_tmp + (end_tmp - start_tmp) * frac
    temperature_inv = 1 / tmp

    action_noise = actions @ fixed_projection
    action_noise = action_noise**2
    predicted_state_action = predicted_state + action_noise

    all_sims = predicted_state_action @ target_state.mT
    # all_sims = predicted_state_action @ torch.concat([target_state, predicted_state_action], dim=0)
    # all_sims = predicted_state_action @ (torch.bmm(torch.diag_embed(action_noise), target_state.unsqueeze(2)).squeeze(2)).mT
    # all_sims = predicted_state_action @ (torch.diag(bilinear) @ target_state.mT)
    # all_sims = all_sims - torch.max(all_sims, dim=-1, keepdim=True)[0]
    # all_sims_det = all_sims.detach()
    #
    # inv_eye = 1.0 - torch.eye(*all_sims.shape).to(device)
    # negs = all_sims * inv_eye
    # pos = torch.diagonal(all_sims, offset=0).unsqueeze(1)
    # neg_weights = (pos - negs).abs()  # .detach()
    # negs_max = torch.max(neg_weights, dim=1, keepdim=True)[0].detach()
    # neg_weights = neg_weights / (negs_max + 1e-8)
    # # neg_weights = 1.0 - neg_weights
    # negs = negs * neg_weights
    # all_sims = torch.diag(pos.squeeze(1)) + inv_eye * negs
    # class_weights = torch.mean(neg_weights, dim=1)
    # class_weights = class_weights / class_weights.max()
    # class_weights = 1.0 - class_weights

    # det_mask = torch.where(torch.rand_like(all_sims_det) > neg_weights, 1.0, 0.0)
    # all_sims = (1.0 - det_mask) * all_sims + det_mask * all_sims_det

    logits = all_sims * temperature_inv
    labels = torch.arange(logits.shape[0]).to(device_global)
    # labels = torch.zeros(logits.shape[0]).long().to(device_global)
    neg_weights = torch.as_tensor([1.0])
    sm = masked_logsoftmax(logits, neg_weights)
    loss = F.nll_loss(sm, labels, reduction="none")
    # loss = F.kl_div(torch.log(torch.eye(*sm.shape) + 1e-8), sm, log_target=True, reduction="batchmean")
    # logits = logits - torch.max(logits, dim=-1, keepdim=True)[0]
    # label_smoothing = 0.01 if layer_id == 0 else 0.1
    # start_tmp = 0.1
    # end_tmp = 0.01
    # frac = env_step / args.total_timesteps
    # label_smoothing = start_tmp + (end_tmp - start_tmp) * frac
    # label_smoothing = 0
    # loss = F.cross_entropy(logits, labels, reduction="none", label_smoothing=label_smoothing)

    # smooth_labels = torch.ones_like(logits) / logits.shape[1]
    # smooth_loss = F.cross_entropy(logits, smooth_labels)
    # smooth_w = 0.05
    # loss = torch.where(loss < 1e-4, torch.nan, loss)
    loss = torch.nanmean(loss)  # * (1.0 - smooth_w) + smooth_w * smooth_loss
    # if torch.isnan(loss):
    #     loss = torch.zeros([1])

    # start_tmp = 0.2
    # end_tmp = 0.005
    # frac = env_step / args.total_timesteps
    # ent_w = start_tmp + (end_tmp - start_tmp) * frac
    # # ent_w = 0.1
    # # log_sm = F.log_softmax(logits, dim=1)
    # # sm = torch.exp(log_sm)
    # # p_log_p = log_sm * sm
    # # entropy = -torch.sum(p_log_p, dim=1).mean()
    # entropy = torch.distributions.Categorical(logits=logits).entropy().mean()
    # loss = loss - entropy * ent_w

    # dis_loss = torch.cdist(predicted_state_action, predicted_state_action).pow(2).sum(dim=1).mean()
    # loss = loss + dis_loss * 0.1

    # negs = logits * (1.0 - torch.eye(logits.shape[0]).to(device_global))
    # zero_target = torch.zeros_like(negs)
    # ortho_loss = F.huber_loss(negs, zero_target)
    # loss = loss + ortho_loss * 0.01
    # loss = loss - torch.sum(negs, dim=1).mean() * 0.001

    # all_sims = predicted_state.detach() @ predicted_state2.mT
    # logits = all_sims * temperature_inv
    # labels = torch.arange(logits.shape[0]).to(device_global)
    # loss_curl = F.cross_entropy(logits, labels, reduction="none")
    #
    # loss = loss + torch.nanmean(loss_curl)

    # l2_reg = torch.pow(target_state_raw, 2).sum(1).mean()
    # l2_reg += torch.pow(predicted_state_raw, 2).sum(1).mean()
    # loss = loss + l2_reg * 0.001

    return loss, layer_out_current, layer_out_next, logits


def cpc_wrapper(model, data, use_noise, normalize, env_step, device):
    in_dists = None
    # in_dists = torch.cdist(data[2], data[2])
    # in_dists = in_dists / (torch.max(in_dists, dim=1, keepdim=True)[0] + 1e-8)
    # in_dists = 1.0 - in_dists
    # in_dists = 1 / (1 + in_dists)
    # in_dists = torch.exp(-in_dists.pow(2) / 0.05)
    in_dists = data[2] @ data[2].mT
    in_dists = in_dists / (torch.max(in_dists, dim=1, keepdim=True)[0] + 1e-8)
    loss1, current_state, next_state, logits1 = cpc_loss_fnc(
        model.actor_mean[0], model.actor_emas[0], model.bilinear1, F.tanh, data, model.fixed_projection1, use_noise, normalize, env_step, device, in_dists, 0
    )
    new_data = (current_state, data[1], next_state, data[3])
    loss2, _, _, logits2 = cpc_loss_fnc(
        model.actor_mean[2], model.actor_emas[1], model.bilinear2, F.tanh, new_data, model.fixed_projection2, use_noise, normalize, env_step, device, in_dists, 1
    )

    schedule = False
    if schedule:
        start_tmp = 1.0
        end_tmp = 0.5
        cycle_decay = 0.9
        cycle_length = args.total_timesteps // 3
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
        # weight = 1.0  # * 0.9**cycle
        # else:
        #     weight = 0.1

    loss = (loss1 + loss2) * weight
    # loss = weight * loss1 + (1.0 - weight) * loss2

    return loss


class L2NormalizationLayer(nn.Module):
    def __init__(self, dim=1, eps=1e-12):
        super(L2NormalizationLayer, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim, eps=self.eps)


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            # torch.nn.utils.parametrizations.spectral_norm(layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64))),
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            # L2NormalizationLayer(),
            # torch.nn.utils.parametrizations.spectral_norm(layer_init(nn.Linear(64, 64))),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            # L2NormalizationLayer(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        # self.actor_hidden = nn.Sequential(
        #     self.actor_mean[0],
        #     nn.Tanh(),
        #     self.actor_mean[2],
        #     nn.Tanh(),
        # )
        # self.actor_head = self.actor_mean[-1]
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

        ema_params = {"beta": 0.9999, "update_after_step": 5, "update_every": 5, "inv_gamma": 1, "power": 2 / 3}
        self.actor_emas = []
        self.actor_emas += [EMA(self.actor_mean[0], **ema_params)]
        self.actor_emas += [EMA(self.actor_mean[2], **ema_params)]

        self.bilinear1 = None  # nn.Parameter(torch.eye(64).to(device_global))
        self.bilinear2 = None  # nn.Parameter(torch.eye(64).to(device_global))
        # self.bilinear1 = layer_init(nn.Linear(64, 64))  # nn.Parameter(torch.ones(64).to(device_global))
        # self.bilinear2 = layer_init(nn.Linear(64, 64))  # nn.Parameter(torch.ones(64).to(device_global))
        # self.fixed_projection1 = layer_init(nn.Linear(np.prod(envs.single_action_space.shape) + 64, 64))
        # self.fixed_projection2 = layer_init(nn.Linear(np.prod(envs.single_action_space.shape) + 64, 64))
        scale = 0.2
        # self.fixed_projection1 = torch.randn(np.prod(envs.single_action_space.shape), np.array(envs.single_observation_space.shape).prod()).to(device_global) * scale
        self.fixed_projection1 = torch.randn(np.prod(envs.single_action_space.shape), 64).to(device_global) * scale
        self.fixed_projection2 = torch.randn(np.prod(envs.single_action_space.shape), 64).to(device_global) * scale
        # drop_ratio = 0.3
        # self.fixed_projection1 = (torch.rand_like(self.fixed_projection1) > drop_ratio).float()
        # self.fixed_projection2 = (torch.rand_like(self.fixed_projection2) > drop_ratio).float()
        # self.fixed_projection1[:64, : -(64 - np.prod(envs.single_action_space.shape))] = 0
        # self.fixed_projection2[:64, : -(64 - np.prod(envs.single_action_space.shape))] = 0
        # self.fixed_projection1 = torch.nn.init.orthogonal_(self.fixed_projection1)
        # self.fixed_projection2 = torch.nn.init.orthogonal_(self.fixed_projection2)
        # self.fixed_projection1 = F.normalize(self.fixed_projection1, dim=1)
        # self.fixed_projection2 = F.normalize(self.fixed_projection2, dim=1)
        # self.fixed_projection1 = torch.full((np.prod(envs.single_action_space.shape), 64), fill_value=1.0) / 64  # np.prod(envs.single_action_space.shape)
        # self.fixed_projection2 = torch.full((np.prod(envs.single_action_space.shape), 64), fill_value=1.0) / 64  # np.prod(envs.single_action_space.shape)
        # self.fixed_projection1 = torch.FloatTensor(np.prod(envs.single_action_space.shape), 64).uniform_(0.1, 0.5)
        # self.fixed_projection2 = torch.FloatTensor(np.prod(envs.single_action_space.shape), 64).uniform_(0.1, 0.5)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )

    def get_action_and_value_detach_last(self, x, action=None):
        action_mean = self.actor_hidden(x).detach()
        action_mean = self.actor_head(action_mean)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    # run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    current_time = datetime.datetime.now()
    current_time = current_time.strftime("%Y-%m-%d_%H-%M")
    setting_name = "ppo"
    if args.use_cpc:
        setting_name += "-cpc"
    run_name = f"{args.env_id}__{setting_name}__{os.uname()[1]}__{current_time}__{args.seed}"
    print(f"*******\n{run_name}\n*******")
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
    writer = SummaryWriter(f"runs/{args.parent_folder}{run_name}")
    param_text = "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]))
    writer.add_text(
        "hyperparameters",
        param_text,
    )

    with open(os.path.join(f"runs", args.parent_folder, run_name, "parameters.txt"), "w") as file:
        file.write(param_text)

    filename = os.path.basename(__file__)
    shutil.copy(__file__, os.path.join(f"runs", args.parent_folder, run_name, filename))

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.use_deterministic_algorithms(args.torch_deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    device_global = device

    # env setup
    envs_list = [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    envs = gym.vector.SyncVectorEnv(envs_list, observation_mode="same")
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    # optimizer = optim.RMSprop(agent.parameters(), lr=args.learning_rate)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    obs_next = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    terminated = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    next_terminated = torch.zeros(args.num_envs).to(device)

    scores_window = deque(maxlen=50)
    best_avg_score = -np.Inf
    best_avg_score_step = 0
    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs

            obs[step] = next_obs
            dones[step] = next_done
            terminated[step] = next_terminated

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            obs_next[step] = torch.Tensor(next_obs).to(device)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            next_terminated = torch.Tensor(terminations).to(device)

            if "episode" in infos:
                scores_window.append(infos["episode"]["r"])
                writer.add_scalar("charts/episodic_return", infos["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", infos["episode"]["l"], global_step)

            if len(scores_window) > 0:
                mean_score = np.mean(scores_window)
                if mean_score > best_avg_score:
                    best_avg_score = mean_score
                    best_avg_score_step = global_step
                if global_step % args.print_steps == 0:
                    print(
                        f"{global_step:1.1e}/{args.total_timesteps:g} ({iteration})"
                        f"\tAvg.Score: {mean_score:.2f} +- {np.std(scores_window):.2f} (max: {np.max(scores_window):.2f} best: {best_avg_score:.2f} {best_avg_score_step:1.2e})"
                    )

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    # nextnonterminal = 1.0 - next_done
                    nextnonterminal = 1.0 - next_terminated
                    nextvalues = next_value
                else:
                    # nextnonterminal = 1.0 - dones[t + 1]
                    nextnonterminal = 1.0 - terminated[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_obs_next = obs_next.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_dones = dones.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # N = args.minibatch_size
                # D = envs.single_observation_space.shape[0]
                # noise1 = np.random.uniform(0.8, 1.2, N * D).reshape(N, -1)
                # noise1 = torch.as_tensor(noise1).float().to(device)
                # used_obs = b_obs[mb_inds] * noise1

                # cycle_length = args.total_timesteps // 4
                # cycle = global_step // cycle_length
                # if cycle == 0:
                #     _, newlogprob, entropy, newvalue = agent.get_action_and_value_detach_last(b_obs[mb_inds], b_actions[mb_inds])
                # else:
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                #################################################################
                # ADD CPC LOSS
                #################################################################
                if args.use_cpc:
                    valids = (1.0 - b_dones[mb_inds]).bool()
                    valid_current_state = b_obs[mb_inds][valids]  # * noise1[valids]
                    valid_actions = b_actions[mb_inds][valids]
                    valid_next_state = b_obs_next[mb_inds][valids]
                    valid_value = newvalue[valids]  # direct critic output
                    # valid_return = b_returns[mb_inds][valids] #critic target: advantage (GAE) + values
                    # valid_return = (valid_return - torch.mean(valid_return)) / (torch.std(valid_return) + 1e8)
                    # valid_advantage = mb_advantages[valids]

                    cpc_loss = cpc_wrapper(
                        model=agent,
                        data=(valid_current_state, valid_actions, valid_next_state, valid_value.detach()),
                        use_noise=args.use_noise,
                        normalize=args.normalize,
                        env_step=global_step,
                        device=device,
                    )
                    # start_tmp = 0.8
                    # end_tmp = 0.01
                    # cycle_decay = 0.9
                    # cycle_length = args.total_timesteps // 4
                    # cycle = global_step // cycle_length
                    # eff_step = global_step - (cycle_length * cycle)
                    # frac = eff_step / cycle_length
                    # start_cycle = start_tmp * cycle_decay**cycle
                    # weight = start_cycle + (end_tmp - start_cycle) * frac

                    # cycle_length = args.total_timesteps // 4
                    # cycle = global_step // cycle_length
                    # if cycle == 0:
                    #     if cycle % 2 == 0:
                    # weight = 0.8  # * 0.9**cycle
                    # loss = loss + cpc_loss
                    # else:
                    # weight = 0.2
                    # loss = loss * (1.0 - weight) + cpc_loss * weight
                    loss = loss + cpc_loss  # * 0.01
                #################################################################
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                # for ema in agent.actor_emas:
                #     ema.update()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        if args.use_cpc:
            writer.add_scalar("losses/cpc_loss", cpc_loss.item(), global_step)

            # metrics1 = metrics[0]
            # metrics2 = metrics[1]
            # writer.add_scalar("losses/volume_1", metrics1[0].item(), global_step)
            # writer.add_scalar("losses/std_1", metrics1[1].item(), global_step)
            # writer.add_scalar("losses/cov_det_1", metrics1[2].item(), global_step)
            # writer.add_scalar("losses/volume_2", metrics2[0].item(), global_step)
            # writer.add_scalar("losses/std_2", metrics2[1].item(), global_step)
            # writer.add_scalar("losses/cov_det_2", metrics2[2].item(), global_step)

        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        # print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    print(
        f"{global_step:1.1e}/{args.total_timesteps:g} ({iteration})"
        f"\tAvg.Score: {mean_score:.2f} +- {np.std(scores_window):.2f} (max: {np.max(scores_window):.2f} best: {best_avg_score:.2f} {best_avg_score_step:1.2e})\n"
    )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from ..cleanrl_utils.evals.ppo_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=Agent,
            device=device,
            gamma=args.gamma,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from ..cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(
                args,
                episodic_returns,
                repo_id,
                "PPO",
                f"runs/{run_name}",
                f"videos/{run_name}-eval",
            )

    envs.close()
    writer.close()
