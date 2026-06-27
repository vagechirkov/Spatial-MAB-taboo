import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sbi_pipelines.pipeline_4d_cnn import Embedding4DCNN
from sbi_pipelines.plotting.plot_search_trajectories import plot_group_trajectories

def load_environments(repo_path=None):
    if repo_path is None:
        repo_path = os.getenv("SOCIAL_GENERALIZATION_REPO", "/tmp/socialGeneralization")
    print("Loading environments from socialGeneralization repo...")
    envs_dict = {}
    for agent_id, prefix in zip([1, 2, 3, 4], ['A', 'B', 'C', 'D']):
        envs_path = os.path.join(repo_path, f"environments/{prefix}_canon.json")
        with open(envs_path, 'r') as f:
            data = json.load(f)
            
        parsed = []
        for env in data:
            mat = np.zeros((11, 11))
            for item in env:
                x = item['x1'] - 1
                y = item['x2'] - 1
                mat[y, x] = item['payoff']
            parsed.append(mat)
        envs_dict[agent_id] = parsed
    return envs_dict

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    envs = load_environments()
    l_mean = 0.5155
    l_std = 0.2374
    
    print("Loading human data...")
    human_data_path = os.getenv("HUMAN_DATA_PATH", "human_data/e1_data_extended.csv")
    df = pd.read_csv(human_data_path)
    
    print("Loading original MLE fits...")
    orig_fits_path = os.getenv("ORIG_FITS_PATH", "human_data/fit+pars_e1.csv")
    if not os.path.exists(orig_fits_path):
        repo_path = os.getenv("SOCIAL_GENERALIZATION_REPO", "/tmp/socialGeneralization")
        orig_fits_path = os.path.join(repo_path, "Data/fit+pars_e1.csv")
        
    if os.path.exists(orig_fits_path):
        original_fits = pd.read_csv(orig_fits_path)
        original_fits["participant_id"] = original_fits["group"].astype(str).str.cat(original_fits["agent"].astype(str), sep="_")
        original_fits = original_fits[original_fits['model'] == 'VS'].reset_index()
    else:
        original_fits = pd.DataFrame(columns=['participant_id', 'group', 'agent', 'model', 'lambda', 'beta', 'tau', 'par'])
        print(f"Warning: Could not find {orig_fits_path}. Proceeding without original fits.")
    
    participants = df.groupby(['group', 'agent'])
    
    x_lists = []
    p_ids = []
    
    for (group, agent), pdf in participants:
        p_id = f"{group}_{agent}"
        if not original_fits.empty and p_id not in original_fits['participant_id'].values:
            continue
            
        pdf = pdf.sort_values(['round', 'trial'])
        x_list = np.zeros((1, 7, 15, 3, 11, 11), dtype=np.float32)
        
        for rd in range(1, 8):
            rd_df = pdf[pdf['round'] == rd]
            if len(rd_df) == 0:
                continue
                
            env_id = int(rd_df['env'].iloc[0])
            if env_id >= len(envs[agent]):
                continue
                
            land = envs[agent][env_id]
            land_norm = (land - l_mean) / l_std
            
            social_df = df[(df['group'] == group) & (df['round'] == rd) & (df['agent'] != agent)]
            
            for t in range(1, 16):
                trial_df = rd_df[rd_df['trial'] == t]
                if len(trial_df) == 0:
                    continue
                
                x_list[0, rd-1, t-1, 0, :, :] = land_norm
                
                choice = int(trial_df['choice'].iloc[0])
                px, py = choice % 11, choice // 11
                x_list[0, rd-1, t-1, 1, px, py] = 1.0
                
                soc_t_df = social_df[social_df['trial'] == t]
                for _, soc_row in soc_t_df.iterrows():
                    soc_choice = int(soc_row['choice'])
                    sx, sy = soc_choice % 11, soc_choice // 11
                    x_list[0, rd-1, t-1, 2, sx, sy] += 1.0
                    
        x_lists.append(x_list.reshape(-1))
        p_ids.append(p_id)
        
    x_test = torch.tensor(np.array(x_lists), dtype=torch.float32).to(device)
    
    print("Loading posterior...")
    post_path = os.getenv("POSTERIOR_PATH", "/results_20260610/pipeline_4d_cnn/posterior_4d_cnn.pt")
    if os.path.exists(post_path):
        posterior = torch.load(post_path, map_location=device)
    else:
        posterior = None
        print(f"Warning: Could not find posterior at {post_path}. Skipping parameter estimation.")
    
    print("Estimating parameters...")
    df_wide_full = pd.DataFrame({'participant_id': p_ids})
    
    params = ['lambda', 'beta', 'tau', 'alpha']
    orig_names = ['lambda', 'beta', 'tau', 'par']
    
    results = {f'mean_{i}': [] for i in range(4)}
    results.update({f'hpdi_lower_{i}': [] for i in range(4)})
    results.update({f'hpdi_upper_{i}': [] for i in range(4)})
    
    for i in range(len(p_ids)):
        _x = x_test[i]
        
        if posterior is not None:
            # sbiutils warning is handled in pipeline_4d_cnn
            samples = posterior.sample((10_000,), x=_x, show_progress_bars=False).cpu().numpy()
            
            for p_idx in range(4):
                param_samples = samples[:, p_idx]
                results[f'mean_{p_idx}'].append(np.mean(param_samples))
                lower, upper = np.quantile(param_samples, [0.05, 0.95]) # 90% HPDI
                results[f'hpdi_lower_{p_idx}'].append(lower)
                results[f'hpdi_upper_{p_idx}'].append(upper)
            
    if posterior is not None:
        for k, v in results.items():
            df_wide_full[k] = v
        
    df_wide_full = pd.merge(df_wide_full, original_fits[['participant_id'] + orig_names], on='participant_id', how='left')
    
    if posterior is not None and not original_fits.empty:
        print("Plotting...")
        fig, axes = plt.subplots(1, 4, figsize=(20, 11))
        axes = axes.flatten()

        for idx, p in enumerate(params):
            means = df_wide_full[f'mean_{idx}']
            orig_fits = df_wide_full[orig_names[idx]].copy()

            # remove outliers
            orig_fits[np.abs(orig_fits - np.mean(orig_fits)) > 2 * np.std(orig_fits)] = np.mean(orig_fits)
            
            lowers = df_wide_full[f'hpdi_lower_{idx}']
            uppers = df_wide_full[f'hpdi_upper_{idx}']
            participants = df_wide_full['participant_id']

            # Determine HPDI color based on MLE overlap
            hpdi_colors = []
            for orig, low, up in zip(orig_fits, lowers, uppers):
                if low <= orig <= up:
                    hpdi_colors.append('green')
                else:
                    hpdi_colors.append('red')

            axes[idx].scatter(orig_fits, participants, color='purple', label='Original fit')
            
            for i, (par, l, u, c) in enumerate(zip(participants, lowers, uppers, hpdi_colors)):
                lbl = '90% HPDI (Overlap)' if c == 'green' and hpdi_colors.index(c) == i else None
                lbl_no = '90% HPDI (No Overlap)' if c == 'red' and hpdi_colors.index(c) == i else None
                axes[idx].hlines(par, l, u, color=c, alpha=0.7, label=lbl or lbl_no)

            axes[idx].set_title(f'Parameter {p}')
            axes[idx].set_ylabel('Participant')
            axes[idx].set_xlabel('Parameter Value')
            axes[idx].tick_params(axis='y', labelsize=6)
            
            # Deduplicate legend
            handles, labels = axes[idx].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            axes[idx].legend(by_label.values(), by_label.keys())

        plt.tight_layout()
        plt.savefig("human_estimation_plot.png")
        print("Plot saved to sbi_pipelines/human_estimation_plot.png")
    else:
        print("Skipping HPDI overlap plot because fitting results (MLE or posterior) are missing.")
    
    # --- New plotting logic for groups ---
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "group_plots")
    os.makedirs(output_dir, exist_ok=True)
    
    plot_group_trajectories(df_wide_full, df, envs, output_dir, orig_names=orig_names)
    
if __name__ == "__main__":
    main()
