import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import wandb
from concurrent.futures import ProcessPoolExecutor

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv


def generate_cholesky_bank(num_envs, grid_size=11, length_scale=2.0, device='cuda'):
    print(f"Generating memory bank of {num_envs} GP maps on {device}...")
    x, y = torch.meshgrid(torch.arange(grid_size, device=device), torch.arange(grid_size, device=device), indexing='xy')
    X = torch.stack([x.ravel(), y.ravel()], dim=1).double()
    dist_sq = torch.cdist(X, X, p=2.0) ** 2
    K = torch.exp(-dist_sq / (2.0 * length_scale ** 2))

    jitter = 1e-6
    L = torch.linalg.cholesky(K + jitter * torch.eye(K.size(0), device=device))
    z = torch.randn(num_envs, K.size(0), 1, dtype=torch.float64, device=device)
    samples = torch.matmul(L, z).squeeze(-1)
    
    grids = samples.view(num_envs, grid_size, grid_size)
    mins = grids.view(num_envs, -1).min(dim=1)[0].view(-1, 1, 1)
    maxs = grids.view(num_envs, -1).max(dim=1)[0].view(-1, 1, 1)
    
    grids = (grids - mins) / (maxs - mins + 1e-8)
    return grids.float()


def generate_mexican_hat_bank(
    num_envs,
    grid_size=11,
    frequency=2.0,
    sigma_inner=None,
    sigma_outer=None,
    center_margin=2,
    sigma_jitter=0.20,
    seed=None,
    device='cuda',
):
    if seed is not None:
        torch.manual_seed(seed)
        
    print(f"Generating memory bank of {num_envs} Mexican-hat maps on {device}...")

    if sigma_inner is None:
        wavelength = grid_size / frequency
        base_sigma_inner = wavelength / 4.0
    else:
        base_sigma_inner = float(sigma_inner)
    
    if sigma_outer is None:
        base_sigma_outer = 2.0 * base_sigma_inner
    else:
        base_sigma_outer = float(sigma_outer)

    lo = float(center_margin)
    hi = float(grid_size - 1 - center_margin)

    # Random centers per environment
    centers = torch.rand(num_envs, 2, device=device) * (hi - lo) + lo
    rows_c = centers[:, 0].view(num_envs, 1, 1)
    cols_c = centers[:, 1].view(num_envs, 1, 1)

    # Random sigmas with log-normal jitter
    si_jitter = torch.exp(torch.randn(num_envs, device=device) * sigma_jitter)
    so_jitter = torch.exp(torch.randn(num_envs, device=device) * sigma_jitter)
    
    si = base_sigma_inner * si_jitter
    so = base_sigma_outer * so_jitter
    so = torch.max(so, si * 1.05)
    
    si = si.view(num_envs, 1, 1)
    so = so.view(num_envs, 1, 1)

    # Grid coordinates
    rows_g, cols_g = torch.meshgrid(
        torch.arange(grid_size, device=device, dtype=torch.float32), 
        torch.arange(grid_size, device=device, dtype=torch.float32), 
        indexing='ij'
    )
    rows_g = rows_g.unsqueeze(0)
    cols_g = cols_g.unsqueeze(0)

    r2 = (rows_g - rows_c)**2 + (cols_g - cols_c)**2

    # DoG formula matching abm.rewards._dog_filter_2d
    g_inner = torch.exp(-r2 / (2.0 * si**2))
    g_outer = torch.exp(-r2 / (2.0 * so**2)) * (si / so) * 1.5
    dog = g_inner - g_outer
    
    # Normalize per map to [0, 1]
    dog_flat = dog.view(num_envs, -1)
    d_min = dog_flat.min(dim=1, keepdim=True)[0].view(num_envs, 1, 1)
    d_max = dog_flat.max(dim=1, keepdim=True)[0].view(num_envs, 1, 1)
    grids = (dog - d_min) / (d_max - d_min + 1e-8)
    
    return grids.float()


def generate_correlated_dog_bank(
    num_envs,
    grid_size=33,
    length_scale=10.0,
    sigma_inner=None,
    sigma_outer=None,
    dog_max=1.2,
    seed=None,
    n_workers=None,
    device='cuda',
):
    print(f"Generating memory bank of {num_envs} correlated-DoG maps on {device} (Torch-vectorized)...")
    if seed is not None:
        torch.manual_seed(seed)

    # 1. GP part: generate GP samples on double precision for stability
    x, y = torch.meshgrid(
        torch.arange(grid_size, device=device, dtype=torch.float64), 
        torch.arange(grid_size, device=device, dtype=torch.float64), 
        indexing='xy'
    )
    coords = torch.stack([x.ravel(), y.ravel()], dim=1)
    dist_sq = torch.cdist(coords, coords, p=2.0) ** 2
    K = torch.exp(-dist_sq / (2.0 * length_scale ** 2))
    jitter = 1e-6
    L = torch.linalg.cholesky(K + jitter * torch.eye(K.size(0), device=device, dtype=torch.float64))
    z = torch.randn(num_envs, K.size(0), 1, dtype=torch.float64, device=device)
    gp_samples = torch.matmul(L, z).squeeze(-1).view(num_envs, grid_size, grid_size)
    
    # 2. Find min indices for each map
    min_indices = torch.argmin(gp_samples.view(num_envs, -1), dim=1)

    gp_flat = gp_samples.view(num_envs, -1)
    gp_min = gp_flat.min(dim=1, keepdim=True)[0].view(num_envs, 1, 1)
    gp_max_val = gp_flat.max(dim=1, keepdim=True)[0].view(num_envs, 1, 1)
    gp_samples = (gp_samples - gp_min) / (gp_max_val - gp_min + 1e-8)

    rows_min = (min_indices // grid_size).view(num_envs, 1, 1).float()
    cols_min = (min_indices % grid_size).view(num_envs, 1, 1).float()

    # 3. DoG part
    if sigma_inner is None and sigma_outer is None:
        sigma_outer = length_scale
        sigma_inner = sigma_outer / 2.0
    
    # Grid coordinates for DoG (ij indexing works for distance)
    rows_g, cols_g = torch.meshgrid(
        torch.arange(grid_size, device=device, dtype=torch.float32), 
        torch.arange(grid_size, device=device, dtype=torch.float32), 
        indexing='ij'
    )
    rows_g = rows_g.unsqueeze(0)
    cols_g = cols_g.unsqueeze(0)
    r2 = (rows_g - rows_min)**2 + (cols_g - cols_min)**2
    
    # Matching dog_rbf_landscape from abm.rewards
    dog_inner = torch.exp(-r2 / (2.0 * float(sigma_inner)**2))
    dog_outer = torch.exp(-r2 / (2.0 * float(sigma_outer)**2)) * (float(sigma_inner) / float(sigma_outer))
    dog = dog_inner - dog_outer
    
    # Scale dog peak to dog_max and center mean
    dog_flat = dog.view(num_envs, -1)
    current_dog_max = torch.abs(dog_flat).max(dim=1, keepdim=True)[0].view(num_envs, 1, 1)
    dog = dog / (current_dog_max + 1e-8) * dog_max
    dog = dog - dog.view(num_envs, -1).mean(dim=1).view(num_envs, 1, 1)

    # 4. Result = GP + DoG, then normalize to [0, 1]
    combined = gp_samples.float() + dog
    c_flat = combined.view(num_envs, -1)
    c_min = c_flat.min(dim=1, keepdim=True)[0].view(num_envs, 1, 1)
    c_max = c_flat.max(dim=1, keepdim=True)[0].view(num_envs, 1, 1)
    grids = (combined - c_min) / (c_max - c_min + 1e-8)
    
    return grids.float()

class RegenerateEnvBankCallback(BaseCallback):
    def __init__(self, train_env: VecEnv, generator_fn, regen_freq: int = 1_000_000, verbose: int = 1):
        super().__init__(verbose)
        self.train_env = train_env
        self.generator_fn = generator_fn
        self.regen_freq = regen_freq
        self._last_regen = 0

    @staticmethod
    def _unwrap(env):
        while hasattr(env, 'venv'):
            env = env.venv
        return env

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_regen >= self.regen_freq:
            if self.verbose >= 1:
                print(f"[RegenerateEnvBankCallback] Regenerating reward bank at {self.num_timesteps:,} timesteps...")
            new_bank = self.generator_fn()
            raw_env = self._unwrap(self.train_env)
            raw_env.reward_bank = new_bank
            self._last_regen = self.num_timesteps
            if self.verbose >= 1:
                print(f"[RegenerateEnvBankCallback] New bank shape: {new_bank.shape}")
        return True


class WandbEvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq: int, verbose=0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.eval_env = eval_env
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            print(f"Running periodic evaluation at {self.num_timesteps} timesteps...")
            results, true_surface, path = evaluate_models(self.model, self.eval_env)
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            ax1 = axes[0]
            for budget, rewards_list in results.items():
                if len(rewards_list) == 0: continue
                max_len = max(len(r) for r in rewards_list)
                padded = np.array([r + [r[-1]]*(max_len - len(r)) for r in rewards_list])
                steps_array = np.arange(1, max_len + 1)
                avg_rewards = padded / steps_array
                mean_rewards = np.mean(avg_rewards, axis=0)
                std_rewards = np.std(avg_rewards, axis=0)
                steps = np.arange(1, max_len + 1)
                ax1.plot(steps, mean_rewards, label=f"Budget: {budget}")
                ax1.fill_between(steps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)
                
            ax1.set_title(f"Average Step Reward (Eval Environments)")
            ax1.set_xlabel("Steps")
            ax1.set_ylabel("Average Reward per step")
            ax1.legend()
            ax1.grid(True)
            
            ax2 = axes[1]
            if true_surface is not None and len(path) > 0:
                cax = ax2.imshow(true_surface, origin='lower', cmap='viridis')
                fig.colorbar(cax, ax=ax2, label="True Reward")
                path = np.array(path)
                xs = path[:, 0]; ys = path[:, 1]
                ax2.plot(ys, xs, color='white', linewidth=1.5, alpha=0.7)
                ax2.scatter(ys[0], xs[0], color='blue', s=50, label='Start', zorder=5)
                ax2.scatter(ys[-1], xs[-1], color='red', s=50, label='End', marker='X', zorder=5)
                for i in range(len(ys) - 1):
                    if ys[i] != ys[i+1] or xs[i] != xs[i+1]:
                        dy = ys[i+1] - ys[i]; dx = xs[i+1] - xs[i]
                        ax2.arrow(ys[i], xs[i], dy*0.5, dx*0.5, head_width=0.3, head_length=0.3, fc='white', ec='white', alpha=0.9, length_includes_head=True)
                ax2.set_title("Agent Trajectory Layout")
                ax2.legend()
            plt.tight_layout()
            
            if wandb.run is not None:
                wandb.log({"eval/curves_and_paths": wandb.Image(fig), "global_step": self.num_timesteps}, step=self.num_timesteps)
            plt.close(fig)
            
        return True

def evaluate_models(model, eval_env):
    obs = eval_env.reset()
    n_envs = eval_env.num_envs
    
    cumulative_rewards = np.zeros(n_envs)
    dones = np.zeros(n_envs, dtype=bool)
    
    true_surface_example = eval_env.true_rewards[0].cpu().numpy()
    path_example = []
    
    all_rewards = [[] for _ in range(n_envs)]
    
    while not np.all(dones):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, info = eval_env.step(action)
        
        for i in range(n_envs):
            if not dones[i]:
                cumulative_rewards[i] += reward[i]
                all_rewards[i].append(cumulative_rewards[i])
                if i == 0:
                    path_example.append(info[i]['position'])
            
            if terminated[i]:
                dones[i] = True
                
    results_by_budget = {}
    for i in range(n_envs):
        b = int(eval_env.budget[i].item())
        if b not in results_by_budget:
            results_by_budget[b] = []
        results_by_budget[b].append(all_rewards[i])
        
    return results_by_budget, true_surface_example, path_example
