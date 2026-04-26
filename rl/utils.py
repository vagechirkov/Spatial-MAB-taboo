import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import wandb
from concurrent.futures import ProcessPoolExecutor
import scipy.stats as stats

from tensordict import TensorDict
from torchrl.envs.utils import step_mdp


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


def apply_valley_gradient(raw_dog, dog_inner, dog_outer, rows_g, cols_g, rows_min, cols_min, gradient_magnitude):
    """Applies a purely positive, randomized linear gradient across the entire landscape 
        to naturally lift flat zero-clamped regions."""
    num_envs = raw_dog.shape[0]
    device = raw_dog.device
    
    # 1. Create a linear plane in a random direction across the whole grid
    theta = torch.rand(num_envs, 1, 1, device=device) * 2.0 * math.pi
    plane = cols_g * torch.cos(theta) + rows_g * torch.sin(theta)
    
    # 2. Shift the plane so it is strictly >= 0 across the board, then scale
    global_min = plane.view(num_envs, -1).min(dim=1, keepdim=True)[0].view(num_envs, 1, 1)
    positive_plane = (plane - global_min) * gradient_magnitude
    
    # 3. Add the sloped floor to the ENTIRE landscape
    return raw_dog + positive_plane


def calculate_spatial_turbulence_mask(r2, peak_safe_radius=1.5, moat_width=6.0, fade_width=3.0):
    """Generates a spatial turbulence ring based on exact physical grid cell distance.
    Guarantees 0.0 at the exact peak, a broad plateau of 1.0 for 'moat_width' cells, 
    and a smooth fade out to avoid detectable spatial artifacts."""
    
    # Convert squared distance to raw cell distance
    r = torch.sqrt(r2 + 1e-8)
    
    # 1. Protect the exact peak: Smooth rise from 0 to 1
    rise = torch.clamp(r / peak_safe_radius, 0.0, 1.0)
    
    # 2. The Broad Moat & Fade Out
    outer_start = peak_safe_radius + moat_width
    fall = 1.0 - torch.clamp((r - outer_start) / fade_width, 0.0, 1.0)
    
    return rise * fall


def generate_correlated_dog_bank_split(
    num_envs,
    grid_size=33,
    length_scale=10.0,
    sigma_inner=None,
    sigma_outer=None,
    seed=None,
    valley_gradient_mag=0.0,
    non_dog_fraction=0.0,
    device='cuda',
):
    """Generate separate GP, DoG, and Turbulence components for per-env dog_max scaling.
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

# 4. Spatial Turbulence Mask (Controls width purely by grid cells)
    # Using 6.0 for the moat_width gives you the ~6 cell broad zone you requested.
    turbulence_mask_bank = calculate_spatial_turbulence_mask(
        r2, peak_safe_radius=2, moat_width=6.0, fade_width=4.0
    )

    # 5. Randomized Valley Gradient (Optional) to hide flat zero-regions
    if valley_gradient_mag > 0:
        raw_dog = apply_valley_gradient(
            raw_dog, dog_inner, dog_outer, 
            rows_g, cols_g, rows_min, cols_min, 
            valley_gradient_mag
        )

    # 6. Final Normalization for the actual environment dog_bank
    raw_dog_flat = raw_dog.view(num_envs, -1)
    dog_max_raw = raw_dog_flat.max(dim=1, keepdim=True)[0].view(num_envs, 1, 1)
    dog_bank = torch.where(
        raw_dog > 0,
        raw_dog / (dog_max_raw + 1e-8),
        raw_dog
    ).float()

    # 7. Non-DoG Sample Mixing
    is_dog_bank = torch.ones(num_envs, dtype=torch.bool, device=device)
    if non_dog_fraction > 0:
        num_non_dog = int(num_envs * non_dog_fraction)
        if num_non_dog > 0:
            is_dog_bank[:num_non_dog] = False
            dog_bank[:num_non_dog] = 0.0

    return gp_bank, dog_bank, turbulence_mask_bank, is_dog_bank



class WandbEvalCallback:
    def __init__(self, eval_env, eval_freq: int, verbose=0):
        self.eval_freq = eval_freq
        self.eval_env = eval_env
        self._last_eval = 0
        
    def log_to_wandb(self, eval_results, current_frames, logger):
        budgets = eval_results["budgets"]
        dog_maxes = eval_results["dog_maxes"]
        found_dog = eval_results["found_dog"]
        t_gp_list = eval_results["t_gp_list"]
        t_dog_list = eval_results["t_dog_list"]
        paths = eval_results["paths"]
        true_surfaces = eval_results["true_surfaces"]

        valid_t_gp_exploit = []
        valid_t_dog_explore = []
        
        for i in range(len(found_dog)):
            # What was the max time budget for this specific environment?
            T_max = budgets[i]
            
            # Expected Time to Exploit (GP)
            # If it failed to find GP, it wasted the whole budget
            t_gp = t_gp_list[i] if t_gp_list[i] is not None else T_max
            valid_t_gp_exploit.append(t_gp)
            
            # Expected Time to Explore (DoG)
            # If it failed to find DoG, it wasted the whole budget
            t_dog = t_dog_list[i] if t_dog_list[i] is not None else T_max
            valid_t_dog_explore.append(t_dog)
        
        empirical_t_gp = float(np.mean(valid_t_gp_exploit)) if len(valid_t_gp_exploit) > 0 else float('nan')
        empirical_t_dog = float(np.mean(valid_t_dog_explore)) if len(valid_t_dog_explore) > 0 else float('nan')
        
        avg_reward_per_step = float(np.mean(eval_results["avg_rewards"]))

        log_dict = {
            "eval/empirical_t_gp": empirical_t_gp,
            "eval/empirical_t_dog": empirical_t_dog,
            "eval/found_dog_rate": float(np.mean(found_dog)),
            "eval/avg_reward_per_step": avg_reward_per_step,
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
                    cax = ax.imshow(surface, origin='lower', cmap='viridis', extent=[0, surface.shape[1], 0, surface.shape[0]])
                    fig_traj.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
                    path = np.array(paths[idx])
                    xs = path[:, 0]
                    ys = path[:, 1]
                    # In our env: x is row, y is col. axes.plot(X, Y) where X is horizontal (col), Y is vertical (row).
                    ax.plot(ys + 0.5, xs + 0.5, color='white', linewidth=1.5, alpha=0.8)
                    ax.scatter(ys[0] + 0.5, xs[0] + 0.5, color='cyan', s=80, label='Start', zorder=5, edgecolors='black')
                    ax.scatter(ys[-1] + 0.5, xs[-1] + 0.5, color='red', s=80, label='End', marker='X', zorder=5, edgecolors='black')
                    T_val = budgets[idx]
                    R_val = dog_maxes[idx]
                    ax.set_title(f"T={T_val:.0f}, R={R_val:.2f}", fontsize=12)
                    ax.legend(loc='upper right', fontsize=8)
                else:
                    ax.set_visible(False)
            fig_traj.suptitle("Agent Trajectories", fontsize=14)
            plt.tight_layout()
            logger.experiment.log({"eval/trajectories": wandb.Image(fig_traj)}, step=current_frames)
            plt.close(fig_traj)

        # --- Plot 2: Policy Heatmap ---
        budgets_arr = np.array(budgets, dtype=np.float64)
        dog_maxes_arr = np.array(dog_maxes, dtype=np.float64)
        found_dog_arr = np.array(found_dog, dtype=np.float64)

        if len(budgets_arr) > 0:
            unique_budgets = np.sort(np.unique(np.round(budgets_arr, 4)))
            if len(unique_budgets) > 1:
                half_gaps = np.diff(unique_budgets) / 2.0
                budget_edges = np.concatenate([
                    [unique_budgets[0] - half_gaps[0]],
                    unique_budgets[:-1] + half_gaps,
                    [unique_budgets[-1] + half_gaps[-1]]
                ])
            else:
                budget_edges = np.array([unique_budgets[0] - 1, unique_budgets[0] + 1])
            
            unique_rs = np.sort(np.unique(np.round(dog_maxes_arr, 4)))
            if len(unique_rs) > 1:
                half_gaps_r = np.diff(unique_rs) / 2.0
                r_edges = np.concatenate([
                    [unique_rs[0] - half_gaps_r[0]],
                    unique_rs[:-1] + half_gaps_r,
                    [unique_rs[-1] + half_gaps_r[-1]]
                ])
            else:
                r_edges = np.array([unique_rs[0] - 0.1, unique_rs[0] + 0.1])

            try:
                stat_result = stats.binned_statistic_2d(
                    budgets_arr, dog_maxes_arr, found_dog_arr,
                    statistic='mean',
                    bins=[budget_edges, r_edges]
                )
                heatmap = stat_result.statistic.T 

                fig_hm, ax_hm = plt.subplots(1, 1, figsize=(10, 7))
                extent = [budget_edges[0], budget_edges[-1], r_edges[0], r_edges[-1]]
                im = ax_hm.imshow(
                    heatmap, origin='lower', aspect='auto', extent=extent,
                    cmap='RdBu_r', vmin=0.0, vmax=1.0
                )
                fig_hm.colorbar(im, ax=ax_hm, label="P(Found DoG)")

                if not (np.isnan(empirical_t_gp) or np.isnan(empirical_t_dog)):
                    # Start R slightly above 1.0 to avoid division by zero explosions
                    R_vals = np.linspace(max(r_edges[0], 1.05), r_edges[-1], 200)
                    T_crit = (R_vals * empirical_t_dog - empirical_t_gp) / (R_vals - 1.0)
                    
                    # --- NEW: Filter out T_crit values that shoot way off the chart ---
                    valid_idx = (T_crit >= budget_edges[0]) & (T_crit <= budget_edges[-1])
                    
                    # Only plot the valid segments of the line
                    ax_hm.plot(T_crit[valid_idx], R_vals[valid_idx], color='lime', linewidth=2.5, linestyle='--',
                               label=f'T_crit (t_gp={empirical_t_gp:.1f}, t_dog={empirical_t_dog:.1f})')
                    ax_hm.legend(loc='lower right', fontsize=9)

                ax_hm.set_xlabel("Time Budget (T)", fontsize=12)
                ax_hm.set_ylabel("Dog Max (R)", fontsize=12)
                ax_hm.set_title("Policy Heatmap: Explore vs Exploit", fontsize=14)
                ax_hm.set_xlim(budget_edges[0], budget_edges[-1])
                ax_hm.set_ylim(r_edges[0], r_edges[-1])
                plt.tight_layout()
                logger.experiment.log({"eval/policy_heatmap": wandb.Image(fig_hm)}, step=current_frames)
                plt.close(fig_hm)
            except Exception as e:
                print(f"[WandbEvalCallback] Policy heatmap failed: {e}")

        for k, v in log_dict.items():
            logger.log_scalar(k, v, step=current_frames)

    def on_step(self, current_frames, policy, logger):
        if current_frames - self._last_eval >= self.eval_freq:
            print(f"Running evaluation at step {current_frames}...")
            eval_results = evaluate_models(policy, self.eval_env)
            self.log_to_wandb(eval_results, current_frames, logger)
            self._last_eval = current_frames


def evaluate_models(policy, eval_env):
    n_envs = eval_env.batch_size[0]
    grid_size = eval_env.grid_size
    device = eval_env.device
    
    td = eval_env.reset()
    
    true_rewards = eval_env.true_rewards
    budgets_tensor = eval_env.budget.cpu().numpy()
    dog_maxes_tensor = eval_env.current_dog_max.cpu().numpy()
    dog_maxes_torch = eval_env.current_dog_max
    
    dones = torch.zeros(n_envs, dtype=torch.bool, device=device)
    t_gp = [None] * n_envs
    t_dog = [None] * n_envs
    max_reward = torch.zeros(n_envs, device=device)
    total_reward = torch.zeros(n_envs, device=device)
    step_count = torch.zeros(n_envs, dtype=torch.int, device=device)
    
    n_path_envs = min(16, n_envs)
    paths = [[] for _ in range(n_path_envs)]
    true_surfaces = [true_rewards[i].cpu().numpy() for i in range(n_path_envs)]
    
    while not torch.all(dones):
        with torch.no_grad():
            td = policy(td)
            
        td = eval_env.step(td)
        
        # 1. Pull the scaled reward
        scaled_reward = td["next", "reward"].squeeze(-1)
        done = td["next", "done"].squeeze(-1)
        action = td["action"]
        
        # 2. Unscale it back to original landscape values
        true_reward = scaled_reward * eval_env.reward_scale
        
        active = ~dones
        if not active.any():
            break
            
        step_count[active] += 1
        # Track unscaled reward for accurate metrics
        total_reward[active] += true_reward[active]
        max_reward[active] = torch.max(max_reward[active], true_reward[active])
        
        # 3. Use true_reward for your thresholds
        mask_gp = torch.tensor([x is None for x in t_gp], device=device) & (true_reward > 0.95) & active
        if mask_gp.any():
            for idx in torch.where(mask_gp)[0]:
                t_gp[idx.item()] = int(step_count[idx].item())
                
        mask_dog = torch.tensor([x is None for x in t_dog], device=device) & (action == eval_env.dog_peak_indices) & active
        if mask_dog.any():
            for idx in torch.where(mask_dog)[0]:
                t_dog[idx.item()] = int(step_count[idx].item())
                
        for i in range(n_path_envs):
            if not dones[i]:
                row = action[i] // grid_size
                col = action[i] % grid_size
                paths[i].append((int(row.item()), int(col.item())))
                
        dones = dones | done
        td = step_mdp(td)
        
    found_dog = [bool(t_dog[i] is not None) for i in range(n_envs)]
    
    return {
        "budgets": budgets_tensor.tolist(),
        "dog_maxes": dog_maxes_tensor.tolist(),
        "found_dog": found_dog,
        "t_gp_list": t_gp,
        "t_dog_list": t_dog,
        "avg_rewards": (total_reward.cpu().numpy() / budgets_tensor).tolist(), 
        "paths": paths,
        "true_surfaces": true_surfaces,
    }


def visualize_dog_max_scaling(
    dog_max_values=[0.5, 1.0, 1.5, 2.0], 
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


def visualize_features(
    grid_size=33, 
    valley_gradient_mag=0.05, 
    turbulence_scale=0.1, 
    dog_max=2.0,
    seed=42,
    device='cpu'
):
    print(f"Generating full environment visualization (Seed: {seed}, Valley Grad: {valley_gradient_mag}, Turb: {turbulence_scale})...")
    
    # 1. Generate banks
    gp_bank, dog_bank, turb_bank = generate_correlated_dog_bank_split(
        num_envs=1,
        grid_size=grid_size,
        length_scale=4.0, 
        valley_gradient_mag=valley_gradient_mag,
        seed=seed,
        device=device
    )
    
    gp = gp_bank[0].cpu()
    dog = dog_bank[0].cpu()
    turb = turb_bank[0].cpu()
    
    # 2. Combine to get true reward surface
    true_reward = torch.clamp(gp + dog * dog_max, min=0.0)
    
    # 3. Calculate noise std dev (Variance Heatmap)
    base_noise = 0.01
    noise_std = base_noise + turb * turbulence_scale
    
    # Convert to numpy for plotting
    gp_np = gp.numpy()
    dog_np = dog.numpy()
    true_reward_np = true_reward.numpy()
    turb_np = turb.numpy()
    noise_std_np = noise_std.numpy()
    
    # 4. Plotting
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: The Components
    # A. GP Bank
    im0 = axes[0, 0].imshow(gp_np, origin='lower', cmap='viridis')
    axes[0, 0].set_title("1. GP Bank Map", fontsize=14)
    fig.colorbar(im0, ax=axes[0, 0])
    
    # B. DoG Bank (includes Valley Gradient)
    im1 = axes[0, 1].imshow(dog_np, origin='lower', cmap='viridis')
    axes[0, 1].set_title(f"2. DoG Bank Map (Grad={valley_gradient_mag})", fontsize=14)
    fig.colorbar(im1, ax=axes[0, 1])
    
    # C. Final True Reward
    im2 = axes[0, 2].imshow(true_reward_np, origin='lower', cmap='viridis')
    axes[0, 2].set_title(f"3. Final Reward (GP + {dog_max}*DoG)", fontsize=14)
    fig.colorbar(im2, ax=axes[0, 2])
    
    # Row 2: Noise & Profile
    # D. Turbulence Mask
    im3 = axes[1, 0].imshow(turb_np, origin='lower', cmap='plasma')
    axes[1, 0].set_title("4. Turbulence Mask", fontsize=14)
    fig.colorbar(im3, ax=axes[1, 0])
    
    # E. Variance Heatmap (Noise Std Dev)
    im4 = axes[1, 1].imshow(noise_std_np, origin='lower', cmap='magma')
    axes[1, 1].set_title(f"5. Variance Heatmap (Scale={turbulence_scale})", fontsize=14)
    fig.colorbar(im4, ax=axes[1, 1])
    
    # F. 1D Cross-section
    peak_idx = torch.argmax(dog.view(-1))
    px, py = peak_idx // grid_size, peak_idx % grid_size
    slice_reward = true_reward_np[px, :]
    slice_noise = noise_std_np[px, :]
    
    ax5 = axes[1, 2]
    ax5_twin = ax5.twinx()
    p1, = ax5.plot(slice_reward, label='True Reward', color='blue', linewidth=2.5)
    p2, = ax5_twin.plot(slice_noise, label='Noise Std Dev', color='red', linestyle='--', linewidth=2.5)
    
    ax5.set_xlabel("Grid X (slice through peak)", fontsize=12)
    ax5.set_ylabel("Reward Value", fontsize=12)
    ax5_twin.set_ylabel("Noise Std Dev", fontsize=12)
    ax5_twin.set_ylim(0, max(0.1, noise_std_np.max() * 1.2))
    ax5.set_title("6. Cross-section Profile", fontsize=14)
    ax5.legend(handles=[p1, p2], loc='upper right', fontsize=10)
    ax5.grid(alpha=0.3)
    
    plt.suptitle(f"Environment Features: Gradient={valley_gradient_mag}, Turbulence Scale={turbulence_scale}", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = "feature_visualization.png"
    plt.savefig(save_path)
    print(f"Feature visualization saved to: {os.path.abspath(save_path)}")


if __name__ == "__main__":
    print("--- Running Dog Max Scaling Visualization ---")
    visualize_dog_max_scaling(length_scale=4.0)
    
    # print("\n--- Running Dog Refinement Visualization ---")
    # visualize_features(grid_size=33, valley_gradient_mag=0.001, turbulence_scale=0.2)
