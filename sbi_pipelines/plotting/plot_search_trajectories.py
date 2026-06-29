import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_group_trajectories(df_wide_full, df, envs, output_dir, orig_names=None):
    print(f"Generating group plots in {output_dir}...")
    
    if orig_names is None:
        orig_names = ['lambda', 'beta', 'tau', 'par']
        
    groups = list(set([p.split('_')[0] for p in df_wide_full['participant_id']]))
    symbols = [r'$\lambda$', r'$\beta$', r'$\tau$', r'$\alpha$']
    
    for group_str in groups:
        group_id = int(group_str)
        group_df = df_wide_full[df_wide_full['participant_id'].str.startswith(f"{group_str}_")].copy()
        group_df['agent_id'] = group_df['participant_id'].apply(lambda x: int(x.split('_')[1]))
        group_df = group_df.sort_values('agent_id')
        
        n_agents = len(group_df)
        if n_agents == 0: 
            continue
            
        fig, axes = plt.subplots(n_agents, 7, figsize=(28, 4 * n_agents))
        if n_agents == 1:
            axes = np.expand_dims(axes, 0)
            
        fig.suptitle(f"Group {group_id}", fontsize=20, y=0.98)
        
        for r, (_, row) in enumerate(group_df.iterrows()):
            agent_id = row['agent_id']
            p_id = row['participant_id']
            
            pdf = df[(df['group'] == group_id) & (df['agent'] == agent_id)]
            pdf = pdf.sort_values(['round', 'trial'])
            
            param_parts = []
            for p_idx, sym in enumerate(symbols):
                orig_n = orig_names[p_idx]
                has_sbi = f'mean_{p_idx}' in row and pd.notna(row.get(f'mean_{p_idx}'))
                has_mle = orig_n in row and pd.notna(row.get(orig_n))
                
                if has_sbi and has_mle:
                    mean_val = row[f'mean_{p_idx}']
                    low = row[f'hpdi_lower_{p_idx}']
                    up = row[f'hpdi_upper_{p_idx}']
                    orig_val = row[orig_n]
                    param_parts.append(f"{sym}: {mean_val:.2f} [{low:.2f}, {up:.2f}] (MLE: {orig_val:.2f})")
                elif has_sbi:
                    mean_val = row[f'mean_{p_idx}']
                    low = row[f'hpdi_lower_{p_idx}']
                    up = row[f'hpdi_upper_{p_idx}']
                    param_parts.append(f"{sym}: {mean_val:.2f} [{low:.2f}, {up:.2f}]")
                elif has_mle:
                    orig_val = row[orig_n]
                    param_parts.append(f"{sym} (MLE: {orig_val:.2f})")
                    
            param_str = " | ".join(param_parts)
            
            for rd in range(1, 8):
                ax = axes[r, rd-1]
                
                # Use the middle subplot (rd=4) for the long parameter subtitle
                if rd == 4 and param_str:
                    ax.set_title(f"Agent {agent_id}  —  {param_str}\nRound {rd}", fontsize=14)
                elif rd == 4:
                    ax.set_title(f"Agent {agent_id}\nRound {rd}", fontsize=14)
                else:
                    ax.set_title(f"Round {rd}", fontsize=14)
                
                rd_df = pdf[pdf['round'] == rd]
                if len(rd_df) == 0:
                    ax.axis('off')
                    continue
                    
                env_id = int(rd_df['env'].iloc[0])
                if agent_id in envs and env_id < len(envs[agent_id]):
                    land = envs[agent_id][env_id]
                    ax.imshow(land, origin='lower', cmap='viridis')
                
                px_list = []
                py_list = []
                for _, t_row in rd_df.iterrows():
                    choice = int(t_row['choice'])
                    px, py = choice % 11, choice // 11
                    px_list.append(px)
                    py_list.append(py)
                    
                if px_list:
                    ax.plot(px_list, py_list, marker='o', color='white', markersize=4, linestyle='-', linewidth=1.5, alpha=0.8)
                    ax.scatter([px_list[0]], [py_list[0]], color='lime', marker='^', s=150, label='Start', zorder=5, edgecolors='black')
                    ax.scatter([px_list[-1]], [py_list[-1]], color='red', marker='s', s=100, label='End', zorder=5, edgecolors='black')
                
                ax.set_xticks([])
                ax.set_yticks([])
                if rd == 1 and r == 0:
                    ax.legend(loc='upper right')
                    
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(output_dir, f"group_{group_id}.png"))
        plt.close(fig)
        
    print("Done generating group plots!")
