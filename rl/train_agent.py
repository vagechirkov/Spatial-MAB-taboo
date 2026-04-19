import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import getpass

import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import wandb
from wandb.integration.sb3 import WandbCallback

from rl.utils import (
    generate_cholesky_bank, generate_mexican_hat_bank, generate_correlated_dog_bank,
    generate_correlated_dog_bank_split,
    WandbEvalCallback, RegenerateEnvBankCallback,
)
from stable_baselines3.common.vec_env import VecEnv

class BatchedSpatialBanditEnv(VecEnv):
    def __init__(self, reward_bank=None, num_envs=64, grid_size=11, budgets=[15],
                 noise_std=0.01, device='cuda', dog_max_range=[1.0, 3.0],
                 gp_bank=None, dog_bank=None, fixed_eval_grid=False):
        self.device = device
        # Support both legacy single bank and split bank modes
        self.reward_bank = reward_bank  # legacy: combined (num_maps, H, W)
        self.gp_bank = gp_bank          # split mode: GP component (num_maps, H, W)
        self.dog_bank = dog_bank         # split mode: DoG component (num_maps, H, W)
        self.use_split_bank = (gp_bank is not None and dog_bank is not None)
        
        observation_space = gym.spaces.Dict({
            "grid": gym.spaces.Box(low=0.0, high=1.0, shape=(2, grid_size, grid_size), dtype=np.float32),
            "budget": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "dog_max": gym.spaces.Box(low=float(dog_max_range[0]), high=float(dog_max_range[1]), shape=(1,), dtype=np.float32),
        })
        action_space = gym.spaces.Discrete(grid_size * grid_size)
        super(BatchedSpatialBanditEnv, self).__init__(num_envs, observation_space, action_space)
        
        self.grid_size = grid_size
        self.budgets_pool = torch.tensor(budgets, device=device, dtype=torch.float32)
        self.noise_std = noise_std
        self.dog_max_range = dog_max_range
        
        self.rng = torch.Generator(device=device)
        self.steps_taken = torch.zeros(num_envs, device=device, dtype=torch.float32)
        self.budget = torch.zeros(num_envs, device=device, dtype=torch.float32)
        self.current_dog_max = torch.zeros(num_envs, device=device, dtype=torch.float32)
        self.true_rewards = torch.zeros(num_envs, grid_size, grid_size, device=device)
        self.revealed_rewards = torch.zeros(num_envs, grid_size, grid_size, device=device)
        self.visited_mask = torch.zeros(num_envs, grid_size, grid_size, device=device)
        self.cumulative_reward = torch.zeros(num_envs, device=device, dtype=torch.float32)
        self.fixed_eval_grid = fixed_eval_grid
        if self.fixed_eval_grid:
            # Generate a structured visualization grid based on simulation parameters
            lo, hi = dog_max_range
            viz_rs = torch.linspace(lo, hi, 4, device=device)
            
            # Pick up to 4 budgets from the pool for the visualization columns
            pool_len = len(self.budgets_pool)
            if pool_len >= 4:
                # Pick 4 roughly evenly spaced budgets from the pool
                indices = torch.linspace(0, pool_len - 1, 4, device=device).long()
                viz_ts = self.budgets_pool[indices]
            else:
                # Repeat the pool if there are fewer than 4 budgets
                viz_ts = self.budgets_pool.repeat(4)[:4]
            
            grid_r, grid_t = torch.meshgrid(viz_rs, viz_ts, indexing='ij')
            self.fixed_dog_max = grid_r.flatten()
            self.fixed_budgets = grid_t.flatten()
            
            # Fill remaining environments with the standard linspace distribution
            remaining = num_envs - len(self.fixed_dog_max)
            if remaining > 0:
                # Use ceil division to ensure we have enough samples
                n_per_budget = (remaining + len(self.budgets_pool) - 1) // len(self.budgets_pool)
                rem_budgets = self.budgets_pool.repeat(n_per_budget)[:remaining]
                lo, hi = dog_max_range
                r_linspace = torch.linspace(lo, hi, remaining, device=device)
                rem_dog_max = r_linspace # Already has correct length 'remaining'
                
                self.fixed_budgets = torch.cat([self.fixed_budgets, rem_budgets])
                self.fixed_dog_max = torch.cat([self.fixed_dog_max, rem_dog_max])
            else:
                self.fixed_budgets = self.fixed_budgets[:num_envs]
                self.fixed_dog_max = self.fixed_dog_max[:num_envs]
        
        self._reset_idx(torch.arange(num_envs, device=device))

    def _reset_idx(self, indices):
        if len(indices) == 0:
            return
        
        n = len(indices)
        
        if self.use_split_bank:
            # Split bank mode: sample GP and DoG components, then combine with per-env dog_max
            bank_size = len(self.gp_bank)
            bank_idx = torch.randint(0, bank_size, (n,), device=self.device)
            gp_maps = self.gp_bank[bank_idx].clone()
            dog_maps = self.dog_bank[bank_idx].clone()
            
            # Apply random augmentation (same transform to both components)
            k_rot = torch.randint(0, 4, (1,), device=self.device).item()
            flip = torch.randint(0, 2, (1,), device=self.device).item()
            gp_maps = torch.rot90(gp_maps, k=k_rot, dims=[1, 2])
            dog_maps = torch.rot90(dog_maps, k=k_rot, dims=[1, 2])
            if flip == 1:
                gp_maps = torch.flip(gp_maps, dims=[2])
                dog_maps = torch.flip(dog_maps, dims=[2])
            
            # Sample per-env dog_max uniformly from range
            if self.fixed_eval_grid:
                dog_max_vals = self.fixed_dog_max[indices]
            else:
                lo, hi = self.dog_max_range
                dog_max_vals = torch.rand(n, device=self.device) * (hi - lo) + lo
            self.current_dog_max[indices] = dog_max_vals
            
            # Combine: GP (peak=1.0) + DoG (peak=dog_max_val) 
            combined = gp_maps + dog_maps * dog_max_vals.view(n, 1, 1)
            self.true_rewards[indices] = torch.clamp(combined, min=0.0)
        else:
            # Legacy single bank mode
            bank_idx = torch.randint(0, len(self.reward_bank), (n,), device=self.device)
            sampled_envs = self.reward_bank[bank_idx].clone()
            
            k_rot = torch.randint(0, 4, (1,), device=self.device).item()
            flip = torch.randint(0, 2, (1,), device=self.device).item()
            
            sampled_envs = torch.rot90(sampled_envs, k=k_rot, dims=[1, 2])
            if flip == 1:
                sampled_envs = torch.flip(sampled_envs, dims=[2])
                
            self.true_rewards[indices] = sampled_envs
            # For legacy mode, set dog_max to 1.0 (no DoG scaling)
            self.current_dog_max[indices] = 1.0
            
        self.revealed_rewards[indices] = 0.0
        self.visited_mask[indices] = 0.0
        self.steps_taken[indices] = 0.0
        self.cumulative_reward[indices] = 0.0
        
        if self.fixed_eval_grid:
            self.budget[indices] = self.fixed_budgets[indices]
        else:
            b_idx = torch.randint(0, len(self.budgets_pool), (n,), device=self.device)
            self.budget[indices] = self.budgets_pool[b_idx]
        
    def reset(self):
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        return self._get_obs()

    def _get_obs(self):
        grid_obs = torch.stack([self.revealed_rewards, self.visited_mask], dim=1) 
        budget_obs = (1.0 - (self.steps_taken / self.budget)).unsqueeze(1)
        dog_max_obs = self.current_dog_max.unsqueeze(1)
        
        return {
            "grid": grid_obs.cpu().numpy(),
            "budget": budget_obs.cpu().numpy(),
            "dog_max": dog_max_obs.cpu().numpy(),
        }

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        actions = torch.tensor(self.actions, device=self.device)
        
        x = actions // self.grid_size
        y = actions % self.grid_size
        
        batch_idx = torch.arange(self.num_envs, device=self.device)
        
        true_reward = self.true_rewards[batch_idx, x, y]
        # Clip reward to [0, per-env dog_max] to handle DoG peaks correctly
        max_clip = self.current_dog_max
        noisy_reward = torch.clip(
            torch.normal(true_reward, self.noise_std, generator=self.rng),
            torch.zeros_like(max_clip), max_clip
        )
        
        self.visited_mask[batch_idx, x, y] = 1.0
        self.revealed_rewards[batch_idx, x, y] = noisy_reward
        self.steps_taken += 1
        self.cumulative_reward += noisy_reward
        
        rewards = noisy_reward.cpu().numpy()
        dones = (self.steps_taken >= self.budget)
        dones_np = dones.cpu().numpy()
        
        infos = [{} for _ in range(self.num_envs)]
            
        terminal_indices = torch.nonzero(dones).squeeze(1)
        if len(terminal_indices) > 0:
            terminal_obs = self._get_obs()
            grid_cpu = terminal_obs['grid']
            budget_cpu = terminal_obs['budget']
            dog_max_cpu = terminal_obs['dog_max']
            term_idx_cpu = terminal_indices.cpu().numpy()

            for idx_int in term_idx_cpu:
                idx_int = int(idx_int)
                infos[idx_int]['terminal_observation'] = {
                    "grid": grid_cpu[idx_int],
                    "budget": budget_cpu[idx_int],
                    "dog_max": dog_max_cpu[idx_int],
                }
                infos[idx_int]['avg_reward'] = (self.cumulative_reward[idx_int] / self.budget[idx_int]).item()
            self._reset_idx(terminal_indices)
            
        return self._get_obs(), rewards, dones_np, infos
    
    def close(self): pass
    def get_attr(self, attr_name, indices=None): return [None]*self.num_envs
    def set_attr(self, attr_name, value, indices=None): pass
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs): pass
    def env_is_wrapped(self, wrapper_class, indices=None): return [False]*self.num_envs

class SpatialFCNNetwork(nn.Module):
    def __init__(self, observation_space, action_space_n):
        super().__init__()
        grid_shape = observation_space.spaces["grid"].shape
        # grid channels + budget (spatial broadcast) + dog_max (spatial broadcast)
        in_channels = grid_shape[0] + 2
        
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=8, dilation=8),
            nn.ReLU()
        )
        
        # ACTOR: 1x1 Conv compresses the 64 channels into exactly 1 action logit per pixel
        self.actor_conv = nn.Conv2d(64, 1, kernel_size=1)
        
        # CRITIC
        self.critic_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 33 * 33, 256), # Flatten preserves spatial locations
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, obs):
        grid = obs["grid"]
        budget = obs["budget"]
        dog_max = obs["dog_max"]
        B, C, H, W = grid.shape
        
        # Broadcast scalar observations spatially and concatenate
        budget_spatial = budget.view(B, 1, 1, 1).expand(B, 1, H, W)
        dog_max_spatial = dog_max.view(B, 1, 1, 1).expand(B, 1, H, W)
        x = torch.cat([grid, budget_spatial, dog_max_spatial], dim=1)
        
        shared_features = self.shared_conv(x)
        action_logits_2d = self.actor_conv(shared_features)
        action_logits = action_logits_2d.view(B, -1)
        value = self.critic_net(shared_features)
        
        return action_logits, value

class FullyConvPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        
        self.fcn = SpatialFCNNetwork(self.observation_space, self.action_space.n)
        
        self.optimizer = self.optimizer_class(
            self.parameters(), 
            lr=lr_schedule(1), 
            **self.optimizer_kwargs
        )

    def forward(self, obs, deterministic=False):
        action_logits, values = self.fcn(obs)
        distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions):
        action_logits, values = self.fcn(obs)
        distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs):
        action_logits, _ = self.fcn(obs)
        return self.action_dist.proba_distribution(action_logits=action_logits)

    def predict_values(self, obs):
        _, values = self.fcn(obs)
        return values

class TrainingRewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_avg_rewards = []

    def _on_step(self) -> bool:
        # Check if any environment finished an episode
        for info in self.locals.get("infos", []):
            if "avg_reward" in info:
                self.episode_avg_rewards.append(info["avg_reward"])
        
        # Log mean avg_reward periodically
        if len(self.episode_avg_rewards) >= 100:
            mean_avg = np.mean(self.episode_avg_rewards)
            if wandb.run is not None:
                wandb.log({"train/ep_avg_reward_mean": mean_avg}, step=self.num_timesteps)
            self.episode_avg_rewards = []
            
        return True

from stable_baselines3.common.logger import KVWriter

class WandbWriter(KVWriter):
    def write(self, key_values, key_excluded, step=0):
        wandb.log(key_values, step=step)
    def close(self): pass

def run_training_and_eval(environment="correlated_dog", grid_size=11,
                          budgets=None, length_scale=2.0, dog_max_range=None,
                          total_timesteps=15_000_000):
    if budgets is None:
        budgets = [25, 50, 100, 150, 200]
    if dog_max_range is None:
        dog_max_range = [1.0, 3.0]
    
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    VALID_ENVS = {"simple_gp", "mexican_hat", "correlated_dog"}
    if environment not in VALID_ENVS:
        raise ValueError(f"environment must be one of {VALID_ENVS}, got {environment!r}")

    train_bank_size = 5000
    
    # For correlated_dog, use split bank for per-env dog_max scaling
    use_split = (environment == "correlated_dog")
    
    if environment == "simple_gp":
        train_bank = generate_cholesky_bank(num_envs=train_bank_size, grid_size=grid_size, length_scale=length_scale, device=device)
        eval_bank  = generate_cholesky_bank(num_envs=300,   grid_size=grid_size, length_scale=length_scale, device=device)
    elif environment == "mexican_hat":
        train_bank = generate_mexican_hat_bank(num_envs=train_bank_size, grid_size=grid_size, device=device)
        eval_bank  = generate_mexican_hat_bank(num_envs=300,   grid_size=grid_size, device=device)
    elif environment == "correlated_dog":
        train_gp_bank, train_dog_bank = generate_correlated_dog_bank_split(
            num_envs=train_bank_size, grid_size=grid_size, length_scale=length_scale, device=device
        )
        eval_gp_bank, eval_dog_bank = generate_correlated_dog_bank_split(
            num_envs=300, grid_size=grid_size, length_scale=length_scale, device=device
        )

    if device == "cuda":
        n_envs = 256
    else:
        n_envs = min(os.cpu_count() or 4, 64)

    # Use more eval envs for better (T, R) coverage over the sparse grid
    n_eval_envs = max(10_000, len(budgets) * 100)

    if use_split:
        train_env = BatchedSpatialBanditEnv(
            gp_bank=train_gp_bank, dog_bank=train_dog_bank,
            num_envs=n_envs, grid_size=grid_size, budgets=budgets,
            device=device, dog_max_range=dog_max_range
        )
    else:
        train_env = BatchedSpatialBanditEnv(
            reward_bank=train_bank, num_envs=n_envs, grid_size=grid_size,
            budgets=budgets, device=device, dog_max_range=dog_max_range
        )
    train_env = VecMonitor(train_env)
    # Disable reward normalization to preserve absolute R ratio
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=False)

    # Bank regeneration
    if environment == "simple_gp":
        def make_train_bank():
            return generate_cholesky_bank(num_envs=train_bank_size, grid_size=grid_size, length_scale=length_scale, device=device)
    elif environment == "mexican_hat":
        def make_train_bank():
            return generate_mexican_hat_bank(num_envs=train_bank_size, grid_size=grid_size, device=device)
    elif environment == "correlated_dog":
        def make_train_bank():
            return generate_correlated_dog_bank_split(
                num_envs=train_bank_size, grid_size=grid_size, length_scale=length_scale, device=device
            )

    if use_split:
        eval_env = BatchedSpatialBanditEnv(
            gp_bank=eval_gp_bank, dog_bank=eval_dog_bank,
            num_envs=n_eval_envs, grid_size=grid_size, budgets=budgets,
            device=device, dog_max_range=dog_max_range, fixed_eval_grid=True
        )
    else:
        eval_env = BatchedSpatialBanditEnv(
            reward_bank=eval_bank, num_envs=n_eval_envs, grid_size=grid_size,
            budgets=budgets, device=device, dog_max_range=dog_max_range, fixed_eval_grid=True
        )
    eval_env = VecMonitor(eval_env)

    # Finite-horizon: gamma=1.0 so long searches are not artificially punished
    model = PPO(
        FullyConvPolicy, 
        train_env, 
        verbose=1,
        learning_rate=1e-4,
        ent_coef=0.005,
        n_steps=128,
        batch_size=1024,
        n_epochs=10,
        gamma=1.0,
        device=device
    )

    run = wandb.init(
        project="Spatial-MAB-taboo",
        sync_tensorboard=False,
        monitor_gym=False,
        save_code=True,
        config={
            "environment": environment,
            "grid_size": grid_size,
            "length_scale": length_scale,
            "dog_max_range": dog_max_range,
            "budgets": budgets,
            "n_envs": n_envs,
            "n_eval_envs": n_eval_envs,
            "total_timesteps": total_timesteps,
            "learning_rate": 1e-4,
            "ent_coef": 0.005,
            "n_steps": 128,
            "batch_size": 1024,
            "n_epochs": 10,
            "gamma": 1.0,
            "norm_reward": False,
            "dilated_conv": True,
        },
    )
    
    log_dir = "./mab_logs"
    os.makedirs(log_dir, exist_ok=True)
    new_logger = configure(log_dir, ["stdout", "csv"])
    new_logger.output_formats.append(WandbWriter())
    model.set_logger(new_logger)
    
    # Callbacks
    wandb_callback = WandbCallback(
        gradient_save_freq=1000,
        model_save_path=f"models/{run.id}",
        verbose=2,
    )
    regen_callback = RegenerateEnvBankCallback(
        train_env=train_env,
        generator_fn=make_train_bank,
        regen_freq=250_000,
        verbose=1,
    )
    periodic_eval_callback = WandbEvalCallback(eval_env=eval_env, eval_freq=500_000 // n_envs)
    train_reward_callback = TrainingRewardLoggerCallback()
    
    model.learn(
        total_timesteps=total_timesteps, 
        progress_bar=True, 
        callback=[wandb_callback, periodic_eval_callback, regen_callback, train_reward_callback]
    )

    model_path = f"models/{run.id}/final_model"
    model.save(model_path)
    wandb.save(model_path + ".zip")
    wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train RL agent on Spatial MAB")
    parser.add_argument(
        "--environment",
        type=str,
        default="correlated_dog",
        choices=["simple_gp", "mexican_hat", "correlated_dog"],
        help="Reward landscape type (default: correlated_dog)",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=33,
        help="Square grid side length (default: 33)",
    )

    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[25, 50, 100, 150, 200],
        help="Time budget(s) per episode; multiple values are sampled uniformly each reset (default: 25 50 100 150 200)",
    )
    parser.add_argument(
        "--length_scale",
        type=float,
        default=4.0
    )
    parser.add_argument(
        "--dog_max_range",
        type=float,
        nargs=2,
        default=[1.0, 3.0],
        help="Range [low, high] for DoG peak multiplier, sampled uniformly per env (default: 1.0 3.0)",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=100_000_000,
        help="Total environment steps for PPO training (default: 100_000_000)",
    )
    args = parser.parse_args()
    run_training_and_eval(
        environment=args.environment,
        grid_size=args.grid_size,
        budgets=args.budgets,
        length_scale=args.length_scale,
        dog_max_range=args.dog_max_range,
        total_timesteps=args.total_timesteps,
    )
