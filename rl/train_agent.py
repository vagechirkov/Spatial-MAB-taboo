import os
import time
from collections import deque
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical as CategoricalDist

from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.envs import EnvBase
from torchrl.data import Bounded, Categorical as CategoricalSpec, Composite, UnboundedContinuous
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from torchrl.collectors import Collector
from torchrl.record import WandbLogger

from rl.utils import (
    generate_correlated_dog_bank_split, 
    evaluate_models, WandbEvalCallback
)

# Disable overly verbose TorchRL warnings for cleaner output
os.environ["RL_WARNINGS"] = "False"


class BatchedSpatialBanditEnv(EnvBase):
    """
    A batched, GPU-native spatial bandit environment built on TorchRL's EnvBase.
    """
    def __init__(self, gp_bank, dog_bank, turbulence_bank=None, is_dog_bank=None, num_envs=128, grid_size=33, 
                 budgets=[25, 50, 100, 200], noise_std=0.01, turbulence_scale=0.0,
                 device='cuda', dog_max_range=[1.0, 3.0], fixed_eval_grid=False,
                 hide_dog_max=False, hide_time_budget=False, memory_size=None):
        
        super().__init__(device=device, batch_size=torch.Size([num_envs]))
        
        self.gp_bank = gp_bank
        self.dog_bank = dog_bank
        self.turbulence_bank = turbulence_bank
        self.is_dog_bank = is_dog_bank
        self.grid_size = grid_size
        self.num_envs = num_envs
        self.budgets_pool = torch.tensor(budgets, device=device, dtype=torch.float32)
        self.noise_std = noise_std
        self.turbulence_scale = turbulence_scale
        self.dog_max_range = dog_max_range
        self.fixed_eval_grid = fixed_eval_grid
        self.hide_dog_max = hide_dog_max
        self.hide_time_budget = hide_time_budget
        self.memory_size = memory_size
        self.reward_scale = 10.0
        
        if self.memory_size:
            self.history_coords = torch.zeros((num_envs, memory_size, 2), dtype=torch.long, device=device)
            self.history_rewards = torch.zeros((num_envs, memory_size), device=device)

        self.observation_spec = Composite({
            "grid": Bounded(0.0, 1.0, shape=(num_envs, 2, grid_size, grid_size), dtype=torch.float32, device=device),
            "step_fraction": Bounded(0.0, 1.0, shape=(num_envs, 1), dtype=torch.float32, device=device),
            "total_budget": Bounded(0.0, 1.0, shape=(num_envs, 1), dtype=torch.float32, device=device),
            "dog_max": Bounded(float(dog_max_range[0]), float(dog_max_range[1]), shape=(num_envs, 1), dtype=torch.float32, device=device),
            "avg_reward": UnboundedContinuous(shape=(num_envs, 1), device=device)
        }, shape=torch.Size([num_envs]))
        
        self.action_spec = CategoricalSpec(grid_size * grid_size, shape=torch.Size([num_envs]), device=device)
        self.reward_spec = UnboundedContinuous(shape=torch.Size([num_envs, 1]), device=device)
        self.done_spec = Composite({
            "done": CategoricalSpec(2, torch.Size([num_envs, 1]), device=device, dtype=torch.bool),
            "terminated": CategoricalSpec(2, torch.Size([num_envs, 1]), device=device, dtype=torch.bool)
        }, shape=torch.Size([num_envs]))

        self.rng = torch.Generator(device=device)
        self.steps_taken = torch.zeros(num_envs, device=device)
        self.budget = torch.zeros(num_envs, device=device)
        self.current_dog_max = torch.zeros(num_envs, device=device)
        
        self.true_rewards = torch.zeros(num_envs, grid_size, grid_size, device=device)
        self.current_turbulence_mask = torch.zeros(num_envs, grid_size, grid_size, device=device)
        self.revealed_rewards = torch.zeros(num_envs, grid_size, grid_size, device=device)
        self.visited_mask = torch.zeros(num_envs, grid_size, grid_size, device=device)
        self.cumulative_reward = torch.zeros(num_envs, device=device)
        
        if self.fixed_eval_grid: 
            self._init_fixed_grid()

    def _init_fixed_grid(self):
        """Pre-computes a fixed grid of dog_max and budget combinations for evaluation."""
        lo, hi = self.dog_max_range
        rs = torch.linspace(lo, hi, 4, device=self.device)
        
        # Ensure we have at least 4 budgets for the grid
        if len(self.budgets_pool) < 4:
            ts = self.budgets_pool.repeat(4)[:4]
        else:
            ts = self.budgets_pool[:4]
            
        grid_r, grid_t = torch.meshgrid(rs, ts, indexing='ij')
        
        # Flatten and repeat to fill the number of environments
        repeat_factor = (self.num_envs // 16) + 1
        self.fixed_dog_max = grid_r.flatten().repeat(repeat_factor)[:self.num_envs]
        self.fixed_budget = grid_t.flatten().repeat(repeat_factor)[:self.num_envs]

    def _reset(self, tensordict=None):
        """Resets specific environments triggered by the SyncDataCollector."""
        if tensordict is not None and "_reset" in tensordict.keys():
            idx = torch.where(tensordict["_reset"].view(-1))[0]
        else:
            idx = torch.arange(self.num_envs, device=self.device)
            
        n = len(idx)
        if n == 0:
            return self._get_obs()
            
        # Sample base maps from the bank
        bank_idx = torch.randint(0, len(self.gp_bank), (n,), device=self.device)
        gp_maps = self.gp_bank[bank_idx].clone()
        dog_maps = self.dog_bank[bank_idx].clone()
        if self.turbulence_bank is not None:
            turb_maps = self.turbulence_bank[bank_idx].clone()
        else:
            turb_maps = torch.zeros_like(gp_maps)
        
        # Random spatial augmentations
        k_rot = torch.randint(0, 4, (1,), device=self.device).item()
        flip = torch.randint(0, 2, (1,), device=self.device).item()
        
        gp_maps = torch.rot90(gp_maps, k=k_rot, dims=[1, 2])
        dog_maps = torch.rot90(dog_maps, k=k_rot, dims=[1, 2])
        turb_maps = torch.rot90(turb_maps, k=k_rot, dims=[1, 2])
        
        if flip == 1: 
            gp_maps = torch.flip(gp_maps, dims=[2])
            dog_maps = torch.flip(dog_maps, dims=[2])
            turb_maps = torch.flip(turb_maps, dims=[2])
            
        # Assign budgets and dog_max multiplier
        if self.fixed_eval_grid: 
            dog_max_vals = self.fixed_dog_max[idx]
            budgets = self.fixed_budget[idx]
        else: 
            lo, hi = self.dog_max_range
            dog_max_vals = torch.rand(n, device=self.device) * (hi - lo) + lo
            
            # If we have a bank mask, force dog_max to 1.0 for non-dog samples
            if self.is_dog_bank is not None:
                is_dog_sample = self.is_dog_bank[bank_idx]
                dog_max_vals = torch.where(is_dog_sample, dog_max_vals, torch.ones_like(dog_max_vals))
            
            budget_idx = torch.randint(0, len(self.budgets_pool), (n,), device=self.device)
            budgets = self.budgets_pool[budget_idx]
            
        self.current_dog_max[idx] = dog_max_vals
        self.budget[idx] = budgets
        
        # Compute true ground truth landscape
        combined_landscape = gp_maps + dog_maps * dog_max_vals.view(n, 1, 1)
        self.true_rewards[idx] = torch.clamp(combined_landscape, min=0.0)
        self.current_turbulence_mask[idx] = turb_maps
        
        # Reset tracking variables
        self.revealed_rewards[idx] = 0.0
        self.visited_mask[idx] = 0.0
        self.steps_taken[idx] = 0.0
        self.cumulative_reward[idx] = 0.0
        
        if self.memory_size:
            self.history_coords[idx] = 0
            self.history_rewards[idx] = 0.0
        
        return self._get_obs()

    def _get_obs(self):
        """Constructs the observation TensorDict."""
        if self.memory_size:
            revealed = torch.zeros_like(self.revealed_rewards)
            visited = torch.zeros_like(self.visited_mask)
            idx = torch.arange(self.num_envs, device=self.device)
            
            # Reconstruct from oldest to newest so newest takes precedence on the same pixel
            for i in reversed(range(self.memory_size)):
                target_step = self.steps_taken - 1 - i
                valid_mask = target_step >= 0
                if not valid_mask.any(): continue
                
                v_idx = idx[valid_mask]
                h_idx = (target_step[valid_mask] % self.memory_size).long()
                
                hx = self.history_coords[v_idx, h_idx, 0]
                hy = self.history_coords[v_idx, h_idx, 1]
                hr = self.history_rewards[v_idx, h_idx]
                
                visited[v_idx, hx, hy] = 1.0
                revealed[v_idx, hx, hy] = hr
            grid_obs = torch.stack([revealed, visited], dim=1)
        else:
            grid_obs = torch.stack([self.revealed_rewards, self.visited_mask], dim=1)
            
        step_fraction_obs = (1.0 - (self.steps_taken / self.budget)).unsqueeze(1)
        
        max_budget = torch.max(self.budgets_pool).item()
        total_budget_obs = (self.budget / max_budget).unsqueeze(1)
        
        if self.hide_dog_max:
            dog_max_obs = torch.zeros_like(self.current_dog_max).unsqueeze(1)
        else:
            dog_max_obs = self.current_dog_max.unsqueeze(1)
            
        if self.hide_time_budget:
            step_fraction_obs = torch.zeros_like(step_fraction_obs)
            total_budget_obs = torch.zeros_like(total_budget_obs)
            
        avg_reward_obs = (self.cumulative_reward / torch.clamp(self.budget, min=1.0)).unsqueeze(1)
        
        return TensorDict({
            "grid": grid_obs, 
            "step_fraction": step_fraction_obs,
            "total_budget": total_budget_obs, 
            "dog_max": dog_max_obs, 
            "avg_reward": avg_reward_obs
        }, batch_size=torch.Size([self.num_envs]), device=self.device)

    def _step(self, tensordict):
        """Executes a batched action step."""
        idx = torch.arange(self.num_envs, device=self.device)
        actions = tensordict["action"].view(-1)
        
        # Decode spatial action
        x = actions // self.grid_size
        y = actions % self.grid_size
        
        # Sample noisy reward from ground truth and clamp to local max
        true_r = self.true_rewards[idx, x, y]
        local_turb = self.current_turbulence_mask[idx, x, y]
        
        dynamic_noise_std = self.noise_std + (local_turb * self.turbulence_scale)
        noisy_r = torch.normal(true_r, dynamic_noise_std, generator=self.rng)
        clamped_r = torch.min(torch.clamp(noisy_r, min=0.0), self.current_dog_max)
        
        # Update state
        if self.memory_size:
            h_idx = (self.steps_taken % self.memory_size).long()
            self.history_coords[idx, h_idx, 0] = x
            self.history_coords[idx, h_idx, 1] = y
            self.history_rewards[idx, h_idx] = clamped_r
            
        self.visited_mask[idx, x, y] = 1.0
        self.revealed_rewards[idx, x, y] = clamped_r
        self.steps_taken += 1
        self.cumulative_reward += clamped_r
        
        # Check termination
        done = (self.steps_taken >= self.budget).unsqueeze(1)
        
        # Build return dictionary
        obs = self._get_obs()
        obs.update({
            "reward": (clamped_r / self.reward_scale).unsqueeze(1), 
            "done": done, 
            "terminated": done
        })
        
        return obs

    def _set_seed(self, seed): 
        if seed is not None: 
            self.rng.manual_seed(seed)


def orthogonal_init(module, gain=1.0):
    """Applies orthogonal initialization to linear and conv layers."""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None: 
            nn.init.constant_(module.bias, 0)

class SpatialTrunk(nn.Module):
    """Shared convolutional feature extractor for both Actor and Critic."""
    def __init__(self, grid_size=33):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.ReLU(), 
            nn.Conv2d(32, 32, 3, padding=2, dilation=2), nn.GroupNorm(4, 32), nn.ReLU(), 
            nn.Conv2d(32, 32, 3, padding=4, dilation=4), nn.GroupNorm(4, 32), nn.ReLU(), 
            nn.Conv2d(32, 32, 3, padding=12, dilation=12), nn.GroupNorm(4, 32), nn.ReLU()
        )
        self.apply(lambda m: orthogonal_init(m, gain=nn.init.calculate_gain('relu')))
        
    def forward(self, grid, step_fraction, total_budget, dog_max):
        batch_dims = grid.shape[:-3]
        C, H, W = grid.shape[-3:]
        
        # Expand scalar inputs into spatial channels
        grid_flat = grid.reshape(-1, C, H, W)
        step_frac_spatial = step_fraction.reshape(-1, 1, 1, 1).expand(-1, 1, H, W)
        budget_spatial = total_budget.reshape(-1, 1, 1, 1).expand(-1, 1, H, W)
        dog_max_spatial = dog_max.reshape(-1, 1, 1, 1).expand(-1, 1, H, W)
        
        x = torch.cat([grid_flat, step_frac_spatial, budget_spatial, dog_max_spatial], dim=1)
        features = self.net(x)
        
        return features.reshape(*batch_dims, 32, H, W)


class EfficientSpatialTrunk(nn.Module):
    def __init__(self, grid_size=33):
        super().__init__()
        # Only takes the 2D grid: 2 channels (revealed, visited)
        self.conv_net = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=4, dilation=4), nn.ReLU()
        )
        
        # MLPs to generate FiLM parameters from the 3 scalars
        self.film_scale = nn.Linear(3, 32)
        self.film_shift = nn.Linear(3, 32)
        
    def forward(self, grid, step_fraction, total_budget, dog_max):
        batch_dims = grid.shape[:-3]
        C, H, W = grid.shape[-3:]

        # 1. Process spatial data (flatten batch dims for conv layer)
        grid_flat = grid.view(-1, C, H, W)
        x_flat = self.conv_net(grid_flat) 
        
        # 2. Process scalars (flatten batch dims for linear layers)
        scalars_flat = torch.cat([
            step_fraction.view(-1, 1), 
            total_budget.view(-1, 1), 
            dog_max.view(-1, 1)
        ], dim=-1)
        
        scale = self.film_scale(scalars_flat).view(-1, 32, 1, 1)
        shift = self.film_shift(scalars_flat).view(-1, 32, 1, 1)
        
        # 3. Modulate (FiLM)
        # out_flat = (x_flat * scale) + shift
        # Memory-efficient. Creates the scaled tensor, then adds shift in-place
        out_flat = x_flat.mul(scale).add_(shift)
        
        # 4. Unflatten to original batch dims
        return out_flat.view(*batch_dims, 32, H, W)

class DecoupledNet(nn.Module):
    """Joins a spatial trunk with a head, handling multi-argument forwarding."""
    def __init__(self, trunk, head):
        super().__init__()
        self.trunk = trunk
        self.head = head
        
    def forward(self, grid, step_fraction, total_budget, dog_max): 
        return self.head(self.trunk(grid, step_fraction, total_budget, dog_max))

# Combiner class removed in favor of TensorDictSequential

class ActorHead(nn.Module):
    """Outputs action logits per pixel."""
    def __init__(self):
        super().__init__()
        self.head = nn.Conv2d(32, 1, kernel_size=1)
        orthogonal_init(self.head, gain=0.01) # Low gain for high initial entropy
        
    def forward(self, hidden_features): 
        batch_dims = hidden_features.shape[:-3]
        x = hidden_features.view(-1, *hidden_features.shape[-3:])
        logits = self.head(x).flatten(-3)
        return logits.view(*batch_dims, -1)

class CriticHead(nn.Module):
    """Outputs a single state value, preserving spatial hierarchies via downsampling."""
    def __init__(self, grid_size=33):
        super().__init__()
        
        self.conv_part = nn.Sequential(
            # Input is [Batch, 32, grid_size, grid_size]
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        )
        
        # Calculate the flattened size after convolutions
        with torch.no_grad():
            dummy_input = torch.zeros(1, 32, grid_size, grid_size)
            dummy_output = self.conv_part(dummy_input)
            flattened_size = dummy_output.numel()

        self.net = nn.Sequential(
            self.conv_part,
            nn.Flatten(start_dim=-3), 
            nn.Linear(flattened_size, 128), nn.ReLU(), 
            nn.Linear(128, 1)
        )
        
        self.apply(lambda m: orthogonal_init(
            m, gain=nn.init.calculate_gain('relu') if isinstance(m, nn.Linear) and m.out_features > 1 else 1.0
        ))
        
    def forward(self, hidden_features): 
        batch_dims = hidden_features.shape[:-3]
        x = hidden_features.view(-1, *hidden_features.shape[-3:])
        values = self.net(x)
        return values.view(*batch_dims, 1)


def run_training(environment="correlated_dog", grid_size=33, budgets=[25, 50, 100, 150, 200], 
                 length_scale=4.0, dog_max_range=[1.2, 1.8], total_timesteps=100_000_000,
                 turbulence_scale=0.0, valley_gradient_mag=0.0, noise_std=0.01,
                 non_dog_fraction=0.0, hide_dog_max=False, hide_time_budget=False,
                 memory_size=None):
    
    # 0. Hyperparameter Configuration
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_envs": 256 if grid_size <= 22 else 128,
        "frames_per_batch": 256 * 200 if grid_size <= 22 else 128 * 200,
        "mini_batch_size": 2048,
        "ppo_epochs": 4,
        "learning_rate": 5e-4,
        "environment": environment,
        "grid_size": grid_size,
        "budgets": budgets,
        "length_scale": length_scale,
        "dog_max_range": dog_max_range,
        "total_timesteps": total_timesteps,
        "clip_epsilon": 0.2,
        "entropy_bonus": True,
        "entropy_coeff": 0.015,
        "critic_coeff": 0.5,
        "loss_critic_type": "smooth_l1",
        "gamma": 0.99,
        "lmbda": 0.95,
        "trunk_type": "EfficientSpatialTrunk",
        "reward_scale": 10.0,
        "noise_std": noise_std,
        "turbulence_scale": turbulence_scale,
        "valley_gradient_mag": valley_gradient_mag,
        "regen_freq": 250_000,
        "hide_time_budget": hide_time_budget,
        "memory_size": memory_size,
    }

    # 1. Environment Initialization
    train_gp_bank, train_dog_bank, train_turb_bank, train_is_dog_bank = generate_correlated_dog_bank_split(
        10_000, config["grid_size"], length_scale=config["length_scale"], 
        valley_gradient_mag=config["valley_gradient_mag"], 
        non_dog_fraction=non_dog_fraction, device=config["device"]
    )
    train_env = BatchedSpatialBanditEnv(
        train_gp_bank, train_dog_bank, turbulence_bank=train_turb_bank, is_dog_bank=train_is_dog_bank,
        num_envs=config["num_envs"], 
        grid_size=config["grid_size"], budgets=config["budgets"], 
        dog_max_range=config["dog_max_range"], 
        noise_std=config["noise_std"], 
        turbulence_scale=config["turbulence_scale"],
        device=config["device"],
        hide_dog_max=hide_dog_max,
        hide_time_budget=hide_time_budget,
        memory_size=memory_size
    )
    
    # Evaluation Bank: ALWAYS 100% DoG (non_dog_fraction=0.0)
    eval_gp_bank, eval_dog_bank, eval_turb_bank, eval_is_dog_bank = generate_correlated_dog_bank_split(
        300, config["grid_size"], length_scale=config["length_scale"], 
        valley_gradient_mag=config["valley_gradient_mag"], 
        non_dog_fraction=0.0, device=config["device"]
    )
    eval_env = BatchedSpatialBanditEnv(
        eval_gp_bank, eval_dog_bank, turbulence_bank=eval_turb_bank, is_dog_bank=eval_is_dog_bank,
        num_envs=8192, 
        grid_size=config["grid_size"], budgets=config["budgets"], 
        dog_max_range=config["dog_max_range"], 
        turbulence_scale=config["turbulence_scale"],
        device=config["device"], fixed_eval_grid=True,
        hide_dog_max=hide_dog_max,
        hide_time_budget=hide_time_budget,
        memory_size=memory_size
    )

    # 2. Network Instantiation
    # actor_trunk = SpatialTrunk(grid_size=config["grid_size"]).to(config["device"])
    actor_trunk = EfficientSpatialTrunk(grid_size=config["grid_size"]).to(config["device"])
    actor_head = ActorHead().to(config["device"])
    policy_module = TensorDictModule(
        DecoupledNet(actor_trunk, actor_head), 
        in_keys=["grid", "step_fraction", "total_budget", "dog_max"], 
        out_keys=["logits"]
    )

    # critic_trunk = SpatialTrunk(grid_size=config["grid_size"]).to(config["device"])
    critic_trunk = EfficientSpatialTrunk(grid_size=config["grid_size"]).to(config["device"])
    critic_head = CriticHead(grid_size=config["grid_size"]).to(config["device"])
    value_module = TensorDictModule(
        DecoupledNet(critic_trunk, critic_head), 
        in_keys=["grid", "step_fraction", "total_budget", "dog_max"], 
        out_keys=["state_value"]
    )

    # End-to-end models for Data Collection (Environment Rollouts)
    policy_forward = ProbabilisticActor(
        module=policy_module, 
        spec=train_env.action_spec, in_keys=["logits"], out_keys=["action"], 
        distribution_class=CategoricalDist, return_log_prob=True
    ).to(config["device"])
    
    value_forward = ValueOperator(
        module=value_module, 
        in_keys=["grid", "step_fraction", "total_budget", "dog_max"]
    ).to(config["device"])

    policy_loss = policy_forward
    value_loss = value_forward

    # 3. TorchRL Pipeline Objects
    collector = Collector(
        train_env, policy_forward, 
        frames_per_batch=config["frames_per_batch"], total_frames=config["total_timesteps"], 
        device=config["device"], storing_device=config["device"]
    )
    
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(config["frames_per_batch"], device=config["device"]), 
        sampler=SamplerWithoutReplacement(),
        batch_size=config["mini_batch_size"]
    )
    
    loss_module = ClipPPOLoss(
        policy_loss, value_loss, 
        clip_epsilon=config["clip_epsilon"], 
        entropy_bonus=config["entropy_bonus"], 
        entropy_coeff=config["entropy_coeff"], 
        critic_coeff=config["critic_coeff"], 
        loss_critic_type=config["loss_critic_type"]
    )
    loss_module.set_keys(reward=("next", "reward"))
    
    adv_module = GAE(
        gamma=config["gamma"], 
        lmbda=config["lmbda"], 
        value_network=value_forward, 
        average_gae=True
    )
    
    optimizer = torch.optim.Adam(
        list(policy_module.parameters()) + list(value_module.parameters()), 
        lr=config["learning_rate"]
    )
    
    # 4. Logging & Tracking
    logger = WandbLogger(
        exp_name=f"UniversalNN_{int(time.time())}", 
        project="Spatial-MAB-taboo",
        config=config
    )
    eval_callback = WandbEvalCallback(eval_env, eval_freq=1_000_000)

    pbar = tqdm(total=total_timesteps)
    ep_rewards = deque(maxlen=100)
    total_frames = 0
    last_regen = 0

    # 5. Training Loop
    for data in collector:
        frames_collected = data.numel()
        total_frames += frames_collected
        pbar.update(frames_collected)
        
        # Track Rewards
        done_mask = data["next", "done"].squeeze(-1)
        if done_mask.any():
            finished_rewards = data["next", "avg_reward"][done_mask]
            for val in finished_rewards: 
                ep_rewards.append(val.item())
                
        # Compute Advantages
        with torch.no_grad(): 
            # Use float32 for GAE stability and to avoid mixed-type bias errors
            adv_module(data)
            
        # Push flat data into buffer
        replay_buffer.extend(data.reshape(-1))
        
        # PPO Optimization Epochs
        for _ in range(config["ppo_epochs"]):
            # Calculate total mini-batches needed to cover the rollout
            num_mini_batches = config["frames_per_batch"] // config["mini_batch_size"]
            
            for _ in range(num_mini_batches):
                # Pulls exactly 1 mini-batch of size [2048]
                batch = replay_buffer.sample() 

                loss_dict = loss_module(batch)
                loss_val = loss_dict["loss_objective"] + loss_dict["loss_critic"] + loss_dict["loss_entropy"]
                
                optimizer.zero_grad()
                loss_val.backward()
                
                # Best Practice: Clip gradients to prevent policy collapse in spatial envs
                torch.nn.utils.clip_grad_norm_(
                    list(policy_module.parameters()) + list(value_module.parameters()), 
                    max_norm=1.0
                )
                
                optimizer.step()
                
        # Logging & Bank Regeneration
        if len(ep_rewards) > 0: 
            logger.log_scalar("train/ep_reward", np.mean(ep_rewards), step=total_frames)
            
        logger.log_scalar("train/loss_critic", loss_dict["loss_critic"].item(), step=total_frames)
        logger.log_scalar("train/loss_entropy", loss_dict["loss_entropy"].item(), step=total_frames)
        logger.log_scalar("train/loss_actor", loss_dict["loss_objective"].item(), step=total_frames)
        eval_callback.on_step(total_frames, policy_forward, logger)

        replay_buffer.empty()
        
        if total_frames - last_regen >= config["regen_freq"]:
            train_gp_bank, train_dog_bank, train_turb_bank, train_is_dog_bank = generate_correlated_dog_bank_split(
                10_000, config["grid_size"], length_scale=config["length_scale"], 
                valley_gradient_mag=config["valley_gradient_mag"], 
                non_dog_fraction=non_dog_fraction, device=config["device"]
            )
            train_env.gp_bank = train_gp_bank
            train_env.dog_bank = train_dog_bank
            train_env.turbulence_bank = train_turb_bank
            train_env.is_dog_bank = train_is_dog_bank
            last_regen = total_frames

            torch.cuda.empty_cache()
            
    torch.save(policy_forward.state_dict(), "final_fcn.pt")
    if logger is not None:
        logger.experiment.finish()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Spatial MAB with TorchRL")
    parser.add_argument(
        "--environment", type=str, default="correlated_dog", 
        choices=["simple_gp", "mexican_hat", "correlated_dog"]
    )
    parser.add_argument("--grid_size", type=int, default=22)
    parser.add_argument(
        "--budgets", type=int, nargs="+", default=[15, 30, 45, 60],
        help="Time budget(s) per episode"
    )
    parser.add_argument("--length_scale", type=float, default=4.0)
    parser.add_argument(
        "--dog_max_range", type=float, nargs=2, default=[1.2, 1.8],
        help="Range [low, high] for DoG peak multiplier"
    )
    parser.add_argument("--total_timesteps", type=int, default=100_000_000)
    parser.add_argument("--turbulence_scale", type=float, default=0.0)
    parser.add_argument("--valley_gradient_mag", type=float, default=0.0)
    parser.add_argument("--noise_std", type=float, default=0.01)
    parser.add_argument("--non_dog_fraction", type=float, default=0.0,
                        help="Fraction of training bank with pure GP maps (no DoG component)")
    parser.add_argument("--hide_dog_max", action="store_true",
                        help="If set, dog_max is hidden from agent observations (set to 0.0)")
    parser.add_argument("--hide_time_budget", action="store_true",
                        help="If set, time budget info is hidden from agent observations (set to 0.0)")
    parser.add_argument("--memory_size", type=int, default=None,
                        help="If set, only the last N choices are shown in the grid observation")
    args = parser.parse_args()
    
    run_training(
        environment=args.environment,
        grid_size=args.grid_size,
        budgets=args.budgets,
        length_scale=args.length_scale,
        dog_max_range=args.dog_max_range,
        total_timesteps=args.total_timesteps,
        turbulence_scale=args.turbulence_scale,
        valley_gradient_mag=args.valley_gradient_mag,
        noise_std=args.noise_std,
        non_dog_fraction=args.non_dog_fraction,
        hide_dog_max=args.hide_dog_max,
        hide_time_budget=args.hide_time_budget,
        memory_size=args.memory_size,
    )
