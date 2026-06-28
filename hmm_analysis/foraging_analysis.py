import pandas as pd
import numpy as np
import bambi as bmb
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

def analyze_foraging(decoded_csv_path):
    out_dir = os.path.dirname(decoded_csv_path)
    print(f"Loading Viterbi states from {decoded_csv_path}...")
    df = pd.read_csv(decoded_csv_path)
    
    # We want one data point per round.
    # p(Global Exploration) = proportion of state 1
    # Cumulative Reward = Sum of rewards in round - Reward in trial 1
    
    round_stats = []
    max_rew_overall = df['reward'].max()
    
    grouped = df.groupby(['group', 'agent', 'round'])
    for name, group_df in grouped:
        group_df = group_df.sort_values('trial')
        
        # p(Global Exploration)
        p_global_exploration = group_df['hidden_state'].mean()
        
        # Cumulative reward - trial 1 reward
        if len(group_df) > 1:
            r1 = group_df.iloc[0]['reward']
            total_r = group_df['reward'].sum()
            n_trials = len(group_df)
            adjusted_cum_rew = total_r - r1 * n_trials
            
            # Theoretical max positive difference is (number of trials - 1) * max possible reward
            theo_max = (n_trials - 1) * max_rew_overall
            theo_min = -theo_max
            
            # Map [-theo_max, theo_max] to [0, 1] so negative rewards don't get clipped to 0
            norm_rew = (adjusted_cum_rew - theo_min) / (theo_max - theo_min + 1e-8)
            norm_rew = np.clip(norm_rew, 1e-4, 1 - 1e-4) # strict (0, 1) bounds for Beta
            
            round_stats.append({
                'group': name[0],
                'agent': name[1],
                'round': name[2],
                'p_global_exploration': p_global_exploration,
                'norm_cum_reward': norm_rew,
                'adjusted_cum_rew': adjusted_cum_rew
            })
            
    stats_df = pd.DataFrame(round_stats)
    
    # Bin p_global_exploration into 0.0-0.1, 0.1-0.2, 0.2-0.3, and >0.3
    bins = [0.0, 0.1, 0.2, 0.3, 1.0]
    labels = ['0.0-0.1', '0.1-0.2', '0.2-0.3', '>0.3']
    stats_df['p_global_exploration_bin'] = pd.cut(stats_df['p_global_exploration'], bins=bins, labels=labels, include_lowest=True, right=True)
    
    # Convert to string to ensure it is treated as categorical
    stats_df['p_global_exploration_bin'] = stats_df['p_global_exploration_bin'].astype(str)
    
    # Bambi Beta Regression
    print(f"Building Beta Regression for Optimal Foraging with Bambi ({len(stats_df)} rounds)...")
    
    # Use categorical binned p_global_exploration as predictor
    model = bmb.Model("norm_cum_reward ~ p_global_exploration_bin", stats_df, family="beta")
    fitted = model.fit(draws=2000, tune=1000, chains=4, cores=4)
    
    print("\nBeta Regression Summary:")
    summary = az.summary(fitted)
    print(summary)
    summary.to_csv(os.path.join(out_dir, "beta_regression_summary.csv"))
    
    # Plotting using Bambi's plot_predictions
    print("\nGenerating Posterior Predictive Plot with Bambi...")
    
    try:
        plot_obj = bmb.interpret.plot_predictions(
            model, 
            fitted, 
            conditional="p_global_exploration_bin",
            prob=0.95
        )
        
        if hasattr(plot_obj, 'layout'):
            plot_obj = plot_obj.layout(size=(10, 6))
            
        if hasattr(plot_obj, 'theme'):
            plot_obj = plot_obj.theme({"axes.grid": True, "grid.alpha": 0.5, "axes.spines.right": False, "axes.spines.top": False, "font.size": 14})
            
        if hasattr(plot_obj, 'label'):
            plot_obj = plot_obj.label(x="Proportion of Global Exploration (Binned)", y="Expected Normalized Cumulative Reward")
        
        # In Bambi 0.18+, this returns a seaborn.objects.Plot instance
        if hasattr(plot_obj, 'save'):
            plot_obj.save(os.path.join(out_dir, "optimal_foraging_posterior_predictive.png"))
        else:
            # Fallback if it somehow returned matplotlib axes
            if isinstance(plot_obj, tuple):
                fig = plot_obj[0]
            elif hasattr(plot_obj, 'figure'):
                fig = plot_obj.figure
            else:
                fig = plt.gcf()
            fig.savefig(os.path.join(out_dir, "optimal_foraging_posterior_predictive.png"), bbox_inches='tight')
            
        plt.close('all')
        
    except Exception as e:
        print(f"Could not generate posterior predictive plot: {e}")
        
    print("Optimal foraging analysis completed.")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python foraging_analysis.py <path_to_viterbi_states.csv>")
        sys.exit(1)
    analyze_foraging(sys.argv[1])
