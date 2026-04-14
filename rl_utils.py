import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import wandb

from stable_baselines3.common.callbacks import BaseCallback

def generate_cholesky_bank(num_envs, grid_size=11, length_scale=2.0, device='cuda'):
    print(f"Generating memory bank of {num_envs} GP maps on {device}...")
    x, y = torch.meshgrid(torch.arange(grid_size, device=device), torch.arange(grid_size, device=device), indexing='xy')
    X = torch.stack([x.ravel(), y.ravel()], dim=1).float()
    dist_sq = torch.cdist(X, X, p=2.0) ** 2
    K = torch.exp(-dist_sq / (2.0 * length_scale ** 2))

    jitter = 1e-6
    L = torch.linalg.cholesky(K + jitter * torch.eye(K.size(0), device=device))
    z = torch.randn(num_envs, K.size(0), 1, device=device)
    samples = torch.matmul(L, z).squeeze(-1)
    
    grids = samples.view(num_envs, grid_size, grid_size)
    mins = grids.view(num_envs, -1).min(dim=1)[0].view(-1, 1, 1)
    maxs = grids.view(num_envs, -1).max(dim=1)[0].view(-1, 1, 1)
    
    grids = (grids - mins) / (maxs - mins + 1e-8)
    return grids

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
