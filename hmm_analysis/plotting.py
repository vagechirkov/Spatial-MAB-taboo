import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np
import os
import sys
import pymc as pm

def plot_forest(trace, out_dir):
    import arviz as az
    import matplotlib.pyplot as plt
    
    # Set figure size in rcParams before calling plot_forest
    plt.rcParams['figure.figsize'] = (12, 12)
    plt.rcParams['font.size'] = 14
    
    az.plot_forest(
        trace, 
        var_names=["mu_jump_pop_0", "mu_jump_pop_1", "mu_rew_pop_0", "mu_rew_pop_1", "p_trans_pop", "sigma_jump", "sigma_rew", "sigma_trans"],
        combined=True
    )
    
    # Try to rotate x labels on current figure
    fig = plt.gcf()
    for ax in fig.axes:
        ax.tick_params(axis='x', labelsize=14, rotation=45)
        ax.tick_params(axis='y', labelsize=14, rotation=0)
        
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "forest_plot.png"), bbox_inches='tight')
    plt.close('all')

def plot_trace(trace, out_dir):
    import arviz as az
    import matplotlib.pyplot as plt
    az.plot_trace(trace, var_names=["mu_jump_pop_0", "mu_jump_pop_1", "mu_rew_pop_0", "mu_rew_pop_1", "p_trans_pop", "sigma_jump", "sigma_rew", "sigma_trans"])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "trace_plot.png"), bbox_inches='tight')
    plt.close('all')

def plot_state_characteristics(decoded_df, map_params, out_dir):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    c0 = '#0072B2' # Exploit
    c1 = '#D55E00' # Explore
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Map to readable labels
    df_plot = decoded_df.copy()
    df_plot['State'] = df_plot['hidden_state'].map({0: 'Exploitation', 1: 'Exploration'})
    palette = {'Exploitation': c0, 'Exploration': c1}
    
    sns.histplot(data=df_plot, x='jump', hue='State', stat='density', common_norm=False, palette=palette, ax=axes[0], alpha=0.6, bins=15)
    axes[0].set_title("Jump Length Density by State")
    
    sns.histplot(data=df_plot, x='reward', hue='State', stat='density', common_norm=False, palette=palette, ax=axes[1], alpha=0.6, bins=15)
    axes[1].set_title("Reward Density by State")
    
    sns.histplot(data=df_plot, x='trial', hue='State', stat='density', common_norm=False, palette=palette, ax=axes[2], alpha=0.6, bins=8)
    axes[2].set_title("Trial Time Density by State")
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "state_characteristics.png"), bbox_inches='tight')
    plt.close(fig)
    
    # Empirical transitions
    # Compute next state
    df_trans = decoded_df.copy()
    df_trans['state_t'] = df_trans['hidden_state']
    # Group by group, agent and round to shift safely
    df_trans['state_t1'] = df_trans.groupby(['group', 'agent', 'round'])['hidden_state'].shift(-1)
    df_trans = df_trans.dropna(subset=['state_t1'])
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    p_trans = map_params['p_trans_pop']
    
    # Exploit (0) -> Explore (1)
    df_exploit = df_trans[df_trans['state_t'] == 0].copy()
    if len(df_exploit) > 0:
        df_exploit['switch'] = (df_exploit['state_t1'] == 1).astype(int)
        
        df_exploit['reward_bin'] = pd.qcut(df_exploit['reward'], q=5, duplicates='drop')
        sns.pointplot(data=df_exploit, x='reward_bin', y='switch', ax=axes[0, 0], color=c0)
        axes[0, 0].axhline(p_trans[0, 1], ls='--', color=c0, alpha=0.7, label=f'Baseline P={p_trans[0, 1]:.2f}')
        axes[0, 0].legend()
        axes[0, 0].set_title("P(Exploit -> Explore) vs Reward")
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        df_exploit['jump_bin'] = pd.qcut(df_exploit['jump'], q=5, duplicates='drop')
        sns.pointplot(data=df_exploit, x='jump_bin', y='switch', ax=axes[0, 1], color=c0)
        axes[0, 1].axhline(p_trans[0, 1], ls='--', color=c0, alpha=0.7)
        axes[0, 1].set_title("P(Exploit -> Explore) vs Jump Length")
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        sns.pointplot(data=df_exploit, x='trial', y='switch', ax=axes[0, 2], color=c0)
        axes[0, 2].axhline(p_trans[0, 1], ls='--', color=c0, alpha=0.7)
        axes[0, 2].set_title("P(Exploit -> Explore) vs Time in Round")
        
    # Explore (1) -> Exploit (0)
    df_explore = df_trans[df_trans['state_t'] == 1].copy()
    if len(df_explore) > 0:
        df_explore['switch'] = (df_explore['state_t1'] == 0).astype(int)
        try: df_explore['reward_bin'] = pd.qcut(df_explore['reward'], q=5, duplicates='drop')
        except ValueError: df_explore['reward_bin'] = pd.cut(df_explore['reward'], bins=5)
        
        sns.pointplot(data=df_explore, x='reward_bin', y='switch', ax=axes[1, 0], color=c1)
        axes[1, 0].axhline(p_trans[1, 0], ls='--', color=c1, alpha=0.7, label=f'Baseline P={p_trans[1, 0]:.2f}')
        axes[1, 0].legend()
        axes[1, 0].set_title("P(Explore -> Exploit) vs Reward")
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        try: df_explore['jump_bin'] = pd.qcut(df_explore['jump'], q=5, duplicates='drop')
        except ValueError: df_explore['jump_bin'] = pd.cut(df_explore['jump'], bins=5)
        
        sns.pointplot(data=df_explore, x='jump_bin', y='switch', ax=axes[1, 1], color=c1)
        axes[1, 1].axhline(p_trans[1, 0], ls='--', color=c1, alpha=0.7)
        axes[1, 1].set_title("P(Explore -> Exploit) vs Jump Length")
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        sns.pointplot(data=df_explore, x='trial', y='switch', ax=axes[1, 2], color=c1)
        axes[1, 2].axhline(p_trans[1, 0], ls='--', color=c1, alpha=0.7)
        axes[1, 2].set_title("P(Explore -> Exploit) vs Time in Round")
        
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "empirical_transitions.png"), bbox_inches='tight')
    plt.close('all')

def plot_trajectories(sequences, map_params, out_dir):
    """Plot sequences colored by predicted state using 7-round grid with env backgrounds."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    import os
    os.makedirs(out_dir, exist_ok=True)
    
    sys_path = sys.path.copy()
    sys.path.append('.') 
    from sbi_pipelines.estimate_human_4d import load_environments
    from hmm_analysis.decoding import viterbi
    envs = load_environments()
    sys.path = sys_path
    
    p_init = map_params['p_init']
    p_trans_subj = map_params['p_trans_subj']
    alpha_j_subj = map_params['alpha_j_subj']
    beta_j_subj = map_params['beta_j_subj']
    alpha_r_subj = map_params['alpha_r_subj']
    beta_r_subj = map_params['beta_r_subj']
    max_jump = map_params['max_jump']
    max_rew = map_params['max_rew']
    
    markers = ['o', '^'] # 0: Exploit, 1: Explore
    c0 = '#0072B2' # Blue for Exploitation
    c1 = '#D55E00' # Vermilion for Exploration
    colors = [c0, c1]
    
    unique_agents = sorted(list(set([s['agent'] for s in sequences])))
    agent_to_idx = {agent: i for i, agent in enumerate(unique_agents)}
    
    from collections import defaultdict
    group_seqs = defaultdict(lambda: defaultdict(list))
    for seq in sequences:
        group_seqs[seq['group']][seq['agent']].append(seq)
        
    for group_id, agent_dict in group_seqs.items():
        agents = sorted(agent_dict.keys())
        n_agents = len(agents)
        if n_agents == 0:
            continue
            
        fig, axes = plt.subplots(n_agents, 7, figsize=(28, 4 * n_agents))
        if n_agents == 1:
            axes = np.expand_dims(axes, 0)
            
        fig.suptitle(f"Group {group_id} HMM Trajectories", fontsize=24, y=1.02)
        
        for row_idx, agent_id in enumerate(agents):
            a_idx = agent_to_idx[agent_id]
            seq_list = agent_dict[agent_id]
            seq_list.sort(key=lambda x: x['round'])
            
            for seq in seq_list:
                rd = seq['round']
                if rd < 1 or rd > 7:
                    continue
                    
                ax = axes[row_idx, rd-1]
                if row_idx == 0:
                    ax.set_title(f"Round {rd}", fontsize=18)
                if rd == 1:
                    ax.set_ylabel(f"Participant {agent_id}", fontsize=16)
                    
                env_id = seq['env']
                if agent_id in envs and env_id < len(envs[agent_id]):
                    land = envs[agent_id][env_id]
                    # Keep background non-transparent (alpha=1.0)
                    ax.imshow(land, origin='lower', cmap='viridis', alpha=1.0)
                    
                coords = seq['coords']
                # use subject specific parameters for viterbi
                states = viterbi(
                    seq['jumps'], seq['rewards'], max_jump, max_rew, p_init, 
                    p_trans_subj[a_idx], alpha_j_subj[a_idx], beta_j_subj[a_idx], 
                    alpha_r_subj[a_idx], beta_r_subj[a_idx]
                )
                padded_states = np.concatenate([[states[0]], states])
                
                px_list = coords[:, 1] # Column is X-axis
                py_list = coords[:, 0] # Row is Y-axis
                
                # Plot the path line
                ax.plot(px_list, py_list, color='gray', linewidth=1.5, alpha=0.6, zorder=1)
                
                # Scatter points colored by STATE, shaped by state
                for k in range(2):
                    mask = (padded_states == k)
                    if np.any(mask):
                        sc = ax.scatter(coords[mask, 1], coords[mask, 0], c=colors[k], 
                                        marker=markers[k], s=100, edgecolor='black', zorder=2)
                                   
                # Mark start/end uniquely with a star or square
                ax.scatter([px_list[0]], [py_list[0]], color='white', marker='*', s=300, zorder=3, edgecolors='black')
                ax.scatter([px_list[-1]], [py_list[-1]], color='white', marker='s', s=150, zorder=3, edgecolors='black')
                
                ax.set_xticks([])
                ax.set_yticks([])
                    
            for rd in range(1, 8):
                if rd not in [s['round'] for s in seq_list]:
                    axes[row_idx, rd-1].axis('off')
                    
        # Flat Legend at the bottom
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=c0, markersize=12, label='Exploitation'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor=c1, markersize=12, label='Exploration'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='white', markeredgecolor='black', markersize=18, label='Start'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='white', markeredgecolor='black', markersize=12, label='End')
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.01), fontsize=16)
        
        plt.tight_layout(rect=[0, 0.06, 1, 0.95])
        plt.savefig(os.path.join(out_dir, f"group_{group_id}_trajectories.png"), bbox_inches='tight')
        plt.close(fig)
