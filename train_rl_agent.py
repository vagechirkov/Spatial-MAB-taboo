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
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import wandb
from wandb.integration.sb3 import WandbCallback

from rl_utils import generate_cholesky_bank, generate_mexican_hat_bank, generate_correlated_dog_bank, WandbEvalCallback, RegenerateEnvBankCallback
from stable_baselines3.common.vec_env import VecEnv

class BatchedSpatialBanditEnv(VecEnv):
    def __init__(self, reward_bank: torch.Tensor, num_envs=64, grid_size=11, budgets=[15], noise_std=0.01, device='cuda'):
        self.device = device
        self.reward_bank = reward_bank
        
        observation_space = gym.spaces.Dict({
            "grid": gym.spaces.Box(low=0.0, high=1.0, shape=(2, grid_size, grid_size), dtype=np.float32),
            "budget": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        })
        action_space = gym.spaces.Discrete(grid_size * grid_size)
        super(BatchedSpatialBanditEnv, self).__init__(num_envs, observation_space, action_space)
        
        self.grid_size = grid_size
        self.budgets_pool = torch.tensor(budgets, device=device, dtype=torch.float32)
        self.noise_std = noise_std
        
        self.rng = torch.Generator(device=device)
        self.steps_taken = torch.zeros(num_envs, device=device, dtype=torch.float32)
        self.budget = torch.zeros(num_envs, device=device, dtype=torch.float32)
        self.true_rewards = torch.zeros(num_envs, grid_size, grid_size, device=device)
        self.revealed_rewards = torch.zeros(num_envs, grid_size, grid_size, device=device)
        self.visited_mask = torch.zeros(num_envs, grid_size, grid_size, device=device)
        
        self._reset_idx(torch.arange(num_envs, device=device))

    def _reset_idx(self, indices):
        if len(indices) == 0:
            return
        
        bank_idx = torch.randint(0, len(self.reward_bank), (len(indices),), device=self.device)
        sampled_envs = self.reward_bank[bank_idx].clone()
        
        k_rot = torch.randint(0, 4, (1,), device=self.device).item()
        flip = torch.randint(0, 2, (1,), device=self.device).item()
        
        sampled_envs = torch.rot90(sampled_envs, k=k_rot, dims=[1, 2])
        if flip == 1:
            sampled_envs = torch.flip(sampled_envs, dims=[2])
            
        self.true_rewards[indices] = sampled_envs
        self.revealed_rewards[indices] = 0.0
        self.visited_mask[indices] = 0.0
        self.steps_taken[indices] = 0.0
        
        b_idx = torch.randint(0, len(self.budgets_pool), (len(indices),), device=self.device)
        self.budget[indices] = self.budgets_pool[b_idx]
        
    def reset(self):
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        return self._get_obs()

    def _get_obs(self):
        grid_obs = torch.stack([self.revealed_rewards, self.visited_mask], dim=1) 
        budget_obs = (1.0 - (self.steps_taken / self.budget)).unsqueeze(1)
        
        return {
            "grid": grid_obs.cpu().numpy(),
            "budget": budget_obs.cpu().numpy()
        }

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        actions = torch.tensor(self.actions, device=self.device)
        
        x = actions // self.grid_size
        y = actions % self.grid_size
        
        batch_idx = torch.arange(self.num_envs, device=self.device)
        
        true_reward = self.true_rewards[batch_idx, x, y]
        noisy_reward = torch.clip(torch.normal(true_reward, self.noise_std, generator=self.rng), 0.0, 1.0)
        
        self.visited_mask[batch_idx, x, y] = 1.0
        self.revealed_rewards[batch_idx, x, y] = noisy_reward
        self.steps_taken += 1
        
        rewards = noisy_reward.cpu().numpy()
        true_rewards_cpu = true_reward.cpu().numpy()
        noisy_rewards_cpu = noisy_reward.cpu().numpy()
        x_cpu = x.cpu().numpy()
        y_cpu = y.cpu().numpy()
        
        dones = (self.steps_taken >= self.budget)
        dones_np = dones.cpu().numpy()
        
        infos = []
        for i in range(self.num_envs):
            infos.append({
                'true_reward': float(true_rewards_cpu[i]),
                'noisy_reward': float(noisy_rewards_cpu[i]),
                'position': (int(x_cpu[i]), int(y_cpu[i]))
            })
            
        terminal_indices = torch.nonzero(dones).squeeze(1)
        if len(terminal_indices) > 0:
            terminal_obs = self._get_obs()
            grid_cpu = terminal_obs['grid']
            budget_cpu = terminal_obs['budget']
            term_idx_cpu = terminal_indices.cpu().numpy()
            
            for idx_int in term_idx_cpu:
                idx_int = int(idx_int)
                infos[idx_int]['terminal_observation'] = {
                    "grid": grid_cpu[idx_int],
                    "budget": budget_cpu[idx_int]
                }
            self._reset_idx(terminal_indices)
            
        return self._get_obs(), rewards, dones_np, infos
    
    def close(self): pass
    def get_attr(self, attr_name, indices=None): return [None]*self.num_envs
    def set_attr(self, attr_name, value, indices=None): pass
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs): pass
    def env_is_wrapped(self, wrapper_class, indices=None): return [False]*self.num_envs

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128, use_attention: bool = True):
        super().__init__(observation_space, features_dim=features_dim)

        self.use_attention = use_attention
        grid_shape = observation_space.spaces["grid"].shape
        in_channels = grid_shape[0]

        if self.use_attention:
            stem_channels = 128
            self.cnn_stem = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, stem_channels, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, *grid_shape)
                stem_out = self.cnn_stem(dummy)
                _, C_stem, H_stem, W_stem = stem_out.shape
                seq_len = H_stem * W_stem

            d_model = 128
            self.token_proj = nn.Linear(C_stem, d_model)
            self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=8, dim_feedforward=256,
                dropout=0.0, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

            n_flatten = d_model
        else:
            self.cnn = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            with torch.no_grad():
                dummy_grid = torch.zeros(1, *grid_shape)
                n_flatten = self.cnn(dummy_grid).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten + 1, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        if self.use_attention:
            stem_out = self.cnn_stem(observations["grid"])
            B, C_stem, H_stem, W_stem = stem_out.shape
            grid_seq = stem_out.view(B, C_stem, -1).permute(0, 2, 1)
            x = self.token_proj(grid_seq) + self.pos_embedding
            x = self.transformer(x)
            grid_feats = x.mean(dim=1)
        else:
            grid_feats = self.cnn(observations["grid"])

        budget_feat = observations["budget"]
        combined = torch.cat([grid_feats, budget_feat], dim=1)
        return self.linear(combined)

from stable_baselines3.common.logger import KVWriter

class WandbWriter(KVWriter):
    def write(self, key_values, key_excluded, step=0):
        wandb.log(key_values, step=step)
    def close(self): pass

def run_training_and_eval(use_attention=False, environment="correlated_dog", grid_size=11, budgets=None, length_scale=2.0, dog_max=1.2, total_timesteps=15_000_000):
    if budgets is None:
        budgets = [15]
    
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    VALID_ENVS = {"simple_gp", "mexican_hat", "correlated_dog"}
    if environment not in VALID_ENVS:
        raise ValueError(f"environment must be one of {VALID_ENVS}, got {environment!r}")

    train_bank_size = 5000
    if environment == "simple_gp":
        train_bank = generate_cholesky_bank(num_envs=train_bank_size, grid_size=grid_size, length_scale=length_scale, device=device)
        eval_bank  = generate_cholesky_bank(num_envs=300,   grid_size=grid_size, length_scale=length_scale, device=device)
    elif environment == "mexican_hat":
        train_bank = generate_mexican_hat_bank(num_envs=train_bank_size, grid_size=grid_size, device=device)
        eval_bank  = generate_mexican_hat_bank(num_envs=300,   grid_size=grid_size, device=device)
    elif environment == "correlated_dog":
        train_bank = generate_correlated_dog_bank(num_envs=train_bank_size, grid_size=grid_size, length_scale=length_scale, dog_max=dog_max, device=device)
        eval_bank  = generate_correlated_dog_bank(num_envs=300,   grid_size=grid_size, length_scale=length_scale, dog_max=dog_max, device=device)

    if device == "cuda":
        n_envs = 256
    else:
        n_envs = min(os.cpu_count() or 4, 64)

    n_eval_envs = 256
    train_env = BatchedSpatialBanditEnv(reward_bank=train_bank, num_envs=n_envs, grid_size=grid_size, budgets=budgets, device=device)
    train_env = VecMonitor(train_env)
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    if environment == "simple_gp":
        def make_train_bank():
            return generate_cholesky_bank(num_envs=train_bank_size, grid_size=grid_size, length_scale=length_scale, device=device)
    elif environment == "mexican_hat":
        def make_train_bank():
            return generate_mexican_hat_bank(num_envs=train_bank_size, grid_size=grid_size, device=device)
    elif environment == "correlated_dog":
        def make_train_bank():
            return generate_correlated_dog_bank(num_envs=train_bank_size, grid_size=grid_size, length_scale=length_scale, dog_max=dog_max, device=device)

    eval_env = BatchedSpatialBanditEnv(reward_bank=eval_bank, num_envs=n_eval_envs, grid_size=grid_size, budgets=budgets, device=device)
    eval_env = VecMonitor(eval_env)

    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(features_dim=128, use_attention=use_attention),
        net_arch=dict(pi=[64, 64], vf=[64, 64])
    )
    
    print(f"Initializing PPO Component on {train_env.num_envs} environments...")
    print(f"Hardware Acceleration automatically selected: {device.upper()}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    model = PPO(
        "MultiInputPolicy", 
        train_env, 
        policy_kwargs=policy_kwargs, 
        verbose=1,
        learning_rate=1e-4,
        ent_coef=0.005,
        n_steps=128,
        batch_size=1024,
        n_epochs=10,
        device=device
    )

    run = wandb.init(
        project="Spatial-MAB-taboo",
        sync_tensorboard=False,
        monitor_gym=False,
        save_code=True,
        config={
            "environment": environment,
            "use_attention": use_attention,
            "grid_size": grid_size,
            "length_scale": length_scale,
            "dog_max": dog_max,
            "budgets": budgets,
            "n_envs": n_envs,
            "n_eval_envs": n_eval_envs,
            "total_timesteps": total_timesteps,
            "learning_rate": 1e-4,
            "ent_coef": 0.005,
            "n_steps": 128,
            "batch_size": 1024,
            "n_epochs": 10,
            "features_dim": 128,
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
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=[wandb_callback, periodic_eval_callback, regen_callback])

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
        help="Reward landscape type (default: simple_gp)",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=33,
        help="Square grid side length (default: 11)",
    )
    parser.add_argument(
        "--use_attention",
        action="store_true",
        default=False,
        help="Use Transformer-based feature extractor",
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[15],
        help="Time budget(s) per episode; multiple values are sampled uniformly each reset (default: 15)",
    )
    parser.add_argument(
        "--length_scale",
        type=float,
        default=2.0,
        help="RBF kernel length scale for GP / correlated-DoG environments (default: 2.0)",
    )
    parser.add_argument(
        "--dog_max",
        type=float,
        default=1.2,
        help="Scale factor for the DoG peak in correlated-DoG landscapes (default: 1.2)",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=100_000_000,
        help="Total environment steps for PPO training (default: 100_000_000)",
    )
    args = parser.parse_args()
    run_training_and_eval(
        use_attention=args.use_attention,
        environment=args.environment,
        grid_size=args.grid_size,
        budgets=args.budgets,
        length_scale=args.length_scale,
        dog_max=args.dog_max,
        total_timesteps=args.total_timesteps,
    )
