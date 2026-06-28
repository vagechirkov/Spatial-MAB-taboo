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
        
        # Number of Global Exploration steps (state 1)
        n_global_exploration = (group_df['hidden_state'] == 1).sum()
        
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
                'participant_id': f"{name[0]}_{name[1]}",
                'round': name[2],
                'n_global_exploration': n_global_exploration,
                'norm_cum_reward': norm_rew,
                'adjusted_cum_rew': adjusted_cum_rew
            })
            
    stats_df = pd.DataFrame(round_stats)
    stats_df['round_half'] = np.where(stats_df['round'] < 4, 'first_half', 'second_half')
    
    # Bin number of global exploration steps (0-1, 2-3, 4-5, 6-7, >=8)
    bins = [-1, 1, 3, 5, 7, 100]
    labels = ['0-1', '2-3', '4-5', '6-7', '>=8']
    stats_df['n_global_exploration_bin'] = pd.cut(stats_df['n_global_exploration'], bins=bins, labels=labels, include_lowest=False, right=True)
    
    # Convert to string to ensure it is treated as categorical
    stats_df['n_global_exploration_bin'] = stats_df['n_global_exploration_bin'].astype(str)
    
    # Bambi Beta Regression with new predictors
    print(f"Building Beta Regression for Optimal Foraging with Bambi ({len(stats_df)} rounds)...")
    
    # Use categorical binned n_global_exploration as predictor, plus round_half and random effects
    # model = bmb.Model("norm_cum_reward ~ n_global_exploration_bin + round_half + (1|participant_id)", stats_df, family="beta")
    model = bmb.Model("norm_cum_reward ~ n_global_exploration_bin + (1|participant_id)", stats_df, family="beta")
    fitted = model.fit(draws=2000, tune=1000, chains=4, cores=4)
    
    print("\nBeta Regression Summary:")
    summary = az.summary(fitted)
    print(summary)
    summary.to_csv(os.path.join(out_dir, "beta_regression_summary.csv"))
    
    # Plotting matching the transition prediction theme
    print("\nGenerating Posterior Predictive Plot...")
    try:
        valid_labels = [l for l in labels if l in stats_df['n_global_exploration_bin'].values]
        
        intercept = fitted.posterior['Intercept'].values
        
        # Determine beta_round
        if 'round_half' in fitted.posterior:
            if hasattr(fitted.posterior['round_half'], 'round_half_dim'):
                beta_round = fitted.posterior['round_half'].sel(round_half_dim='second_half').values
            else:
                beta_round = fitted.posterior['round_half'].values
        elif 'round_half[second_half]' in fitted.posterior:
            beta_round = fitted.posterior['round_half[second_half]'].values
        else:
            beta_round = np.zeros_like(intercept)
            
        fig, ax = plt.subplots(figsize=(8, 6))
        x_positions = np.arange(len(labels))
        
        # for half, color, beta_r in [('First Half (Rounds 1-4)', 'blue', 0), ('Second Half (Rounds 5-8)', 'orange', beta_round)]:
        for half, color, beta_r in [('Posterior Mean ± 90% HPDI', 'purple', 0)]:
            mu_mean_list = []
            mu_lower_list = []
            mu_upper_list = []
            
            for label in labels:
                if label not in valid_labels:
                    mu_mean_list.append(np.nan)
                    mu_lower_list.append(np.nan)
                    mu_upper_list.append(np.nan)
                    continue
                
                # Retrieve bin coefficient
                beta_bin = np.zeros_like(intercept)
                if 'n_global_exploration_bin' in fitted.posterior:
                    if hasattr(fitted.posterior['n_global_exploration_bin'], 'n_global_exploration_bin_dim'):
                        if label in fitted.posterior['n_global_exploration_bin'].coords['n_global_exploration_bin_dim'].values:
                            beta_bin = fitted.posterior['n_global_exploration_bin'].sel(n_global_exploration_bin_dim=label).values
                elif f'n_global_exploration_bin[{label}]' in fitted.posterior:
                    beta_bin = fitted.posterior[f'n_global_exploration_bin[{label}]'].values
                
                logits = intercept + beta_bin + beta_r
                mu = 1 / (1 + np.exp(-logits))
                
                mu_mean_list.append(mu.mean())
                mu_lower_list.append(np.percentile(mu, 5))
                mu_upper_list.append(np.percentile(mu, 95))
                
            mu_mean = np.array(mu_mean_list)
            mu_lower = np.array(mu_lower_list)
            mu_upper = np.array(mu_upper_list)
            
            valid_mask = ~np.isnan(mu_mean)
            x_valid = x_positions[valid_mask]
            
            yerr_lower = mu_mean[valid_mask] - mu_lower[valid_mask]
            yerr_upper = mu_upper[valid_mask] - mu_mean[valid_mask]
            
            ax.errorbar(x_valid, mu_mean[valid_mask], yerr=[yerr_lower, yerr_upper], 
                        color=color, linewidth=2, marker='o', markersize=8, 
                        capsize=5, capthick=2, elinewidth=2, label=half)
            
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_title("Optimal Foraging: Reward vs Global Exploration", fontsize=16)
        ax.set_xlabel("Number of Global Exploration Steps", fontsize=14)
        ax.set_ylabel("Norm. Cum. Reward (Trial 1 Subtracted)", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "optimal_foraging_posterior_predictive.png"), bbox_inches='tight', dpi=150)
        plt.close('all')
            
    except Exception as e:
        print(f"Could not generate posterior predictive plot: {e}")
        import traceback
        traceback.print_exc()
        
    print("Optimal foraging analysis completed.")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python foraging_analysis.py <path_to_viterbi_states.csv>")
        sys.exit(1)
    analyze_foraging(sys.argv[1])
