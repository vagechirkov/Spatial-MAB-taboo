import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def analyze_transition_performance_correlation(out_dir):
    viterbi_df_path = os.path.join(out_dir, "state_predictions_viterbi.csv")
    trace_summary_path = os.path.join(out_dir, "trace_summary.csv")
    
    if not os.path.exists(viterbi_df_path) or not os.path.exists(trace_summary_path):
        print("Required files not found. Ensure the HMM pipeline has completed.")
        return
        
    print(f"Loading data from {out_dir}...")
    df = pd.read_csv(viterbi_df_path)
    trace_summary = pd.read_csv(trace_summary_path, index_col=0)
    
    # Calculate performance per round, then average per agent
    round_stats = []
    max_rew_overall = df['reward'].max()
    
    grouped = df.groupby(['group', 'agent', 'round'])
    for name, group_df in grouped:
        group_df = group_df.sort_values('trial')
        if len(group_df) > 1:
            r1 = group_df.iloc[0]['reward']
            total_r = group_df['reward'].sum()
            n_trials = len(group_df)
            adjusted_cum_rew = total_r - r1 * n_trials
            
            theo_max = (n_trials - 1) * max_rew_overall
            theo_min = -theo_max
            
            norm_rew = (adjusted_cum_rew - theo_min) / (theo_max - theo_min + 1e-8)
            
            round_stats.append({
                'agent': name[1],
                'norm_cum_reward': norm_rew
            })
            
    perf_df = pd.DataFrame(round_stats).groupby('agent')['norm_cum_reward'].mean().reset_index()
    
    # Extract transition probabilities for each agent
    unique_agents = sorted(df['agent'].unique())
    
    trans_data = []
    for i, agent in enumerate(unique_agents):
        try:
            # We look up by the exact row index string in trace_summary
            p_00 = trace_summary.loc[f"p_trans_subj[{i}, 0, 0]", 'mean']
            p_01 = trace_summary.loc[f"p_trans_subj[{i}, 0, 1]", 'mean']
            p_10 = trace_summary.loc[f"p_trans_subj[{i}, 1, 0]", 'mean']
            p_11 = trace_summary.loc[f"p_trans_subj[{i}, 1, 1]", 'mean']
        except KeyError:
            print(f"Warning: Could not find transition probabilities for agent index {i}")
            continue
            
        trans_data.append({
            'agent': agent,
            'p_00': p_00,
            'p_01': p_01,
            'p_10': p_10,
            'p_11': p_11
        })
        
    trans_df = pd.DataFrame(trans_data)
    
    if len(trans_df) == 0:
        print("Could not extract subject transition probabilities.")
        return
        
    # Merge performance with transition probabilities
    merged = pd.merge(perf_df, trans_df, on='agent')
    
    print("Plotting correlations...")
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # Matrix components
    components = [
        ('p_00', 'P(Local → Local)', 0, 0),
        ('p_01', 'P(Local → Global)', 0, 1),
        ('p_10', 'P(Global → Local)', 1, 0),
        ('p_11', 'P(Global → Global)', 1, 1)
    ]
    
    for col, title, row_idx, col_idx in components:
        ax = axes[row_idx, col_idx]
        sns.regplot(x=col, y='norm_cum_reward', data=merged, ax=ax, scatter_kws={'alpha':0.6}, line_kws={'color': 'red'})
        
        # Calculate Pearson correlation
        r, p = stats.pearsonr(merged[col], merged['norm_cum_reward'])
        
        ax.set_title(f"{title}\nr = {r:.3f}, p = {p:.3f}", fontsize=14)
        ax.set_xlabel(title, fontsize=12)
        ax.set_ylabel("Mean Norm. Cum. Reward" if col_idx == 0 else "", fontsize=12)
        ax.grid(True, alpha=0.3)
        
    plt.suptitle("Correlation: HMM Transition Probabilities vs. Foraging Performance", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "transition_performance_correlation.png"), bbox_inches='tight', dpi=150)
    plt.close('all')
    print(f"Saved correlation plot to {out_dir}/transition_performance_correlation.png")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python transition_performance_correlation.py <out_dir>")
        sys.exit(1)
    analyze_transition_performance_correlation(sys.argv[1])
