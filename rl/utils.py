import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import wandb
from concurrent.futures import ProcessPoolExecutor
import scipy.stats as stats

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
    
    # Generate raw DoG (naturally approaches 0 at distance)
    dog_inner = torch.exp(-r2 / (2.0 * float(sigma_inner)**2))
    dog_outer = torch.exp(-r2 / (2.0 * float(sigma_outer)**2)) * (float(sigma_inner) / float(sigma_outer))
    raw_dog = dog_inner - dog_outer
    
    # Find the max of the raw DoG for scaling
    raw_dog_flat = raw_dog.view(num_envs, -1)
    dog_max_raw = raw_dog_flat.max(dim=1, keepdim=True)[0].view(num_envs, 1, 1)
    
    # PIECEWISE SCALING: 
    # Scale positive peak to `dog_max`. Keep negative values exactly as they are.
    # This ensures the physical width of the base at 0.0 never changes.
    dog = torch.where(
        raw_dog > 0, 
        (raw_dog / (dog_max_raw + 1e-8)) * dog_max, 
        raw_dog
    )

    # 4. Result = GP + DoG
    # GP is already exactly [0, 1]. GP peak is 1.0. 
    # DoG is inserted at GP min (0.0), so DoG peak becomes `dog_max`.
    combined = gp_samples.float() + dog
    
    # Clamp at 0 to keep the lowest values strictly fixed and prevent negative valleys.
    grids = torch.clamp(combined, min=0.0)
    
    return grids.float()


def generate_correlated_dog_bank_split(
    num_envs,
    grid_size=33,
    length_scale=10.0,
    sigma_inner=None,
    sigma_outer=None,
    seed=None,
    device='cuda',
):
    """Generate separate GP and DoG components for per-env dog_max scaling.
    """
    print(f"Generating split memory bank of {num_envs} correlated-DoG maps on {device}...")
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

    # 2. Find min indices for each map (before normalization)
    min_indices = torch.argmin(gp_samples.view(num_envs, -1), dim=1)

    # Normalize GP to [0, 1]
    gp_flat = gp_samples.view(num_envs, -1)
    gp_min = gp_flat.min(dim=1, keepdim=True)[0].view(num_envs, 1, 1)
    gp_max_val = gp_flat.max(dim=1, keepdim=True)[0].view(num_envs, 1, 1)
    gp_bank = ((gp_samples - gp_min) / (gp_max_val - gp_min + 1e-8)).float()

    rows_min = (min_indices // grid_size).view(num_envs, 1, 1).float()
    cols_min = (min_indices % grid_size).view(num_envs, 1, 1).float()

    # 3. DoG part — centered at GP minimum, peak normalized to 1.0
    if sigma_inner is None and sigma_outer is None:
        sigma_outer = length_scale
        sigma_inner = sigma_outer / 2.0

    rows_g, cols_g = torch.meshgrid(
        torch.arange(grid_size, device=device, dtype=torch.float32),
        torch.arange(grid_size, device=device, dtype=torch.float32),
        indexing='ij'
    )
    rows_g = rows_g.unsqueeze(0)
    cols_g = cols_g.unsqueeze(0)
    r2 = (rows_g - rows_min)**2 + (cols_g - cols_min)**2

    dog_inner = torch.exp(-r2 / (2.0 * float(sigma_inner)**2))
    dog_outer = torch.exp(-r2 / (2.0 * float(sigma_outer)**2)) * (float(sigma_inner) / float(sigma_outer))
    raw_dog = dog_inner - dog_outer

    # Normalize positive peak to 1.0 (per-map), keep negatives as-is
    raw_dog_flat = raw_dog.view(num_envs, -1)
    dog_max_raw = raw_dog_flat.max(dim=1, keepdim=True)[0].view(num_envs, 1, 1)
    dog_bank = torch.where(
        raw_dog > 0,
        raw_dog / (dog_max_raw + 1e-8),
        raw_dog
    ).float()

    return gp_bank, dog_bank


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
            result = self.generator_fn()
            raw_env = self._unwrap(self.train_env)
            # Support both split banks (gp_bank, dog_bank) and legacy single bank
            if isinstance(result, tuple):
                raw_env.gp_bank, raw_env.dog_bank = result
                if self.verbose >= 1:
                    print(f"[RegenerateEnvBankCallback] New split bank shapes: GP={raw_env.gp_bank.shape}, DoG={raw_env.dog_bank.shape}")
            else:
                raw_env.reward_bank = result
                if self.verbose >= 1:
                    print(f"[RegenerateEnvBankCallback] New bank shape: {result.shape}")
            self._last_regen = self.num_timesteps
        return True


class WandbEvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq: int, verbose=0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.eval_env = eval_env
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            print(f"Running periodic evaluation at {self.num_timesteps} timesteps...")
            eval_results = evaluate_models(self.model, self.eval_env)
            budgets = eval_results["budgets"]
            dog_maxes = eval_results["dog_maxes"]
            found_dog = eval_results["found_dog"]
            t_gp_list = eval_results["t_gp_list"]
            t_dog_list = eval_results["t_dog_list"]
            paths = eval_results["paths"]
            true_surfaces = eval_results["true_surfaces"]

            # --- Scalar metrics ---
            valid_t_gp = [t for t in t_gp_list if t is not None]
            valid_t_dog = [t for t in t_dog_list if t is not None]
            empirical_t_gp = float(np.mean(valid_t_gp)) if len(valid_t_gp) > 0 else float('nan')
            empirical_t_dog = float(np.mean(valid_t_dog)) if len(valid_t_dog) > 0 else float('nan')
            
            avg_reward_per_step = float(np.mean(eval_results["avg_rewards"]))

            log_dict = {
                "eval/empirical_t_gp": empirical_t_gp,
                "eval/empirical_t_dog": empirical_t_dog,
                "eval/found_dog_rate": float(np.mean(found_dog)),
                "eval/avg_reward_per_step": avg_reward_per_step,
                "global_step": self.num_timesteps,
            }

            # --- Plot 1: Trajectories (4x4 grid) ---
            n_plots = min(16, len(paths))
            if n_plots > 0:
                fig_traj, axes_traj = plt.subplots(4, 4, figsize=(20, 20))
                axes_traj = axes_traj.flatten()
                for idx in range(16):
                    ax = axes_traj[idx]
                    if idx < n_plots and true_surfaces[idx] is not None and len(paths[idx]) > 0:
                        surface = true_surfaces[idx]
                        cax = ax.imshow(surface, origin='lower', cmap='viridis')
                        fig_traj.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
                        path = np.array(paths[idx])
                        xs = path[:, 0]
                        ys = path[:, 1]
                        ax.plot(ys, xs, color='white', linewidth=1.5, alpha=0.8)
                        ax.scatter(ys[0], xs[0], color='cyan', s=80, label='Start', zorder=5, edgecolors='black')
                        ax.scatter(ys[-1], xs[-1], color='red', s=80, label='End', marker='X', zorder=5, edgecolors='black')
                        T_val = budgets[idx]
                        R_val = dog_maxes[idx]
                        ax.set_title(f"T={T_val:.0f}, R={R_val:.2f}", fontsize=12)
                        ax.legend(loc='upper right', fontsize=8)
                    else:
                        ax.set_visible(False)
                fig_traj.suptitle("Agent Trajectories", fontsize=14)
                plt.tight_layout()
                log_dict["eval/trajectories"] = wandb.Image(fig_traj)
                plt.close(fig_traj)

            # --- Plot 2: Policy Heatmap ---
            budgets_arr = np.array(budgets, dtype=np.float64)
            dog_maxes_arr = np.array(dog_maxes, dtype=np.float64)
            found_dog_arr = np.array(found_dog, dtype=np.float64)

            if len(budgets_arr) > 0:
                # Define bins for heatmap
                unique_budgets = np.unique(budgets_arr)
                budget_edges = np.concatenate([
                    unique_budgets - 0.5 * np.diff(np.concatenate([[unique_budgets[0]], unique_budgets])[:len(unique_budgets)+1])[:len(unique_budgets)],
                    [unique_budgets[-1] + 0.5 * (unique_budgets[-1] - unique_budgets[-2]) if len(unique_budgets) > 1 else unique_budgets[-1] + 1]
                ])
                # Simpler: use unique budget values to make edges
                if len(unique_budgets) > 1:
                    half_gaps = np.diff(unique_budgets) / 2.0
                    budget_edges = np.concatenate([
                        [unique_budgets[0] - half_gaps[0]],
                        unique_budgets[:-1] + half_gaps,
                        [unique_budgets[-1] + half_gaps[-1]]
                    ])
                else:
                    budget_edges = np.array([unique_budgets[0] - 1, unique_budgets[0] + 1])
                
                n_r_bins = 10
                r_edges = np.linspace(dog_maxes_arr.min() - 0.01, dog_maxes_arr.max() + 0.01, n_r_bins + 1)

                try:
                    stat_result = stats.binned_statistic_2d(
                        budgets_arr, dog_maxes_arr, found_dog_arr,
                        statistic='mean',
                        bins=[budget_edges, r_edges]
                    )
                    heatmap = stat_result.statistic.T  # shape: (n_r_bins, n_budget_bins)

                    fig_hm, ax_hm = plt.subplots(1, 1, figsize=(10, 7))
                    extent = [budget_edges[0], budget_edges[-1], r_edges[0], r_edges[-1]]
                    im = ax_hm.imshow(
                        heatmap, origin='lower', aspect='auto', extent=extent,
                        cmap='RdBu_r', vmin=0.0, vmax=1.0
                    )
                    fig_hm.colorbar(im, ax=ax_hm, label="P(Found DoG)")

                    # Overlay theoretical T_critical line
                    if not (np.isnan(empirical_t_gp) or np.isnan(empirical_t_dog)):
                        R_vals = np.linspace(max(r_edges[0], 1.01), r_edges[-1], 200)
                        T_crit = (R_vals * empirical_t_dog - empirical_t_gp) / (R_vals - 1.0)
                        ax_hm.plot(T_crit, R_vals, color='lime', linewidth=2.5, linestyle='--',
                                   label=f'T_crit (t_gp={empirical_t_gp:.1f}, t_dog={empirical_t_dog:.1f})')
                        ax_hm.legend(loc='upper right', fontsize=9)

                    ax_hm.set_xlabel("Time Budget (T)", fontsize=12)
                    ax_hm.set_ylabel("Dog Max (R)", fontsize=12)
                    ax_hm.set_title("Policy Heatmap: Explore vs Exploit", fontsize=14)
                    ax_hm.set_xlim(budget_edges[0], budget_edges[-1])
                    ax_hm.set_ylim(r_edges[0], r_edges[-1])
                    plt.tight_layout()
                    log_dict["eval/policy_heatmap"] = wandb.Image(fig_hm)
                    plt.close(fig_hm)
                except Exception as e:
                    print(f"[WandbEvalCallback] Policy heatmap failed: {e}")

            if wandb.run is not None:
                wandb.log(log_dict, step=self.num_timesteps)

        return True


def evaluate_models(model, eval_env):
    # Unwrap environment to get metadata and true rewards
    curr = eval_env
    while hasattr(curr, 'venv'):
        curr = curr.venv
    raw_env = curr

    true_rewards = raw_env.true_rewards  # (n_envs, grid_size, grid_size)
    obs = eval_env.reset()
    n_envs = eval_env.num_envs
    grid_size = true_rewards.shape[1]

    # Per-env tracking
    dones = np.zeros(n_envs, dtype=bool)
    t_gp = [None] * n_envs       # first step with reward > 0.95
    t_dog = [None] * n_envs      # first step with reward > 1.05
    max_reward = np.zeros(n_envs) # max reward seen per env
    total_reward = np.zeros(n_envs)
    step_count = np.zeros(n_envs, dtype=int)

    # Track paths for first 16 envs (visualization grid)
    n_path_envs = min(16, n_envs)
    paths = [[] for _ in range(n_path_envs)]
    true_surfaces = []
    for i in range(n_path_envs):
        true_surfaces.append(true_rewards[i].cpu().numpy())

    # Get budgets and dog_maxes at the start (they were set during reset)
    budgets_tensor = raw_env.budget.cpu().numpy()
    dog_maxes_tensor = raw_env.current_dog_max.cpu().numpy()

    while not np.all(dones):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, info = eval_env.step(action)

        for i in range(n_envs):
            if not dones[i]:
                step_count[i] += 1
                r = reward[i]
                total_reward[i] += r
                max_reward[i] = max(max_reward[i], r)

                # Track first GP hit
                if t_gp[i] is None and r > 0.95:
                    t_gp[i] = int(step_count[i])

                # Track first DoG hit
                if t_dog[i] is None and r > 1.05:
                    t_dog[i] = int(step_count[i])

                # Track path for first 4 envs
                if i < n_path_envs:
                    ax_i = action[i] // grid_size
                    ay_i = action[i] % grid_size
                    paths[i].append((int(ax_i), int(ay_i)))

            if terminated[i]:
                dones[i] = True

    found_dog = [bool(max_reward[i] > 1.05) for i in range(n_envs)]

    return {
        "budgets": budgets_tensor.tolist(),
        "dog_maxes": dog_maxes_tensor.tolist(),
        "found_dog": found_dog,
        "t_gp_list": t_gp,
        "t_dog_list": t_dog,
        "avg_rewards": (total_reward / budgets_tensor).tolist(),
        "paths": paths,
        "true_surfaces": true_surfaces,
    }


def visualize_dog_max_scaling(
    dog_max_values=[1.2, 1.5, 1.8, 2.0], 
    grid_size=33, 
    length_scale=4.0, 
    seed=42, 
    device='cpu'
):
    print(f"Generating examples for dog_max comparison (Seed: {seed}, Length Scale: {length_scale})...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, dog_max in enumerate(dog_max_values):
        grids = generate_correlated_dog_bank(
            num_envs=1,
            grid_size=grid_size,
            length_scale=length_scale,
            dog_max=dog_max,
            seed=seed,
            device=device
        )
        grid = grids[0].numpy()
        
        im = axes[i].imshow(grid, origin='lower', cmap='viridis')
        axes[i].set_title(f"dog_max = {dog_max}")
        axes[i].axis('off')
        fig.colorbar(im, ax=axes[i])
        
        # Print stats to verify gradients/peaks
        print(f"dog_max: {dog_max} | Min: {grid.min():.4f} | Max: {grid.max():.4f}")

    plt.suptitle(f"Correlated DoG Comparison (Seed: {seed}, LS: {length_scale})")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = "dog_examples_comparison.png"
    plt.savefig(save_path)
    print(f"\nComparison plot saved to: {os.path.abspath(save_path)}")


if __name__ == "__main__":
    visualize_dog_max_scaling(length_scale=4.0)
