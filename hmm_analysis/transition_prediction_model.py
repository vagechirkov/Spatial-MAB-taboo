import pandas as pd
import numpy as np
import bambi as bmb
import arviz as az
import matplotlib.pyplot as plt
import os
import sys

def predict_transitions(viterbi_df_path):
    out_dir = os.path.dirname(viterbi_df_path)
    print(f"Loading Viterbi states from {viterbi_df_path}...")
    df = pd.read_csv(viterbi_df_path)
    
    df = df.sort_values(['group', 'agent', 'round', 'trial'])
    
    # Predict next state based on current state (t-1)
    df['state_t1'] = df.groupby(['group', 'agent', 'round'])['hidden_state'].shift(-1)
    df = df.dropna(subset=['state_t1'])
    
    # Focus only on instances where current state is Local Exploration (State 0)
    df_exploit = df[df['hidden_state'] == 0].copy()
    
    if len(df_exploit) == 0:
        print("No Local Exploration states found. Cannot run transition model.")
        return
        
    # Bambi requires the target to be integer for Bernoulli by default, or just specify family
    df_exploit['target_global'] = df_exploit['state_t1'].astype(int)
    
    # Normalize predictors
    df_exploit['reward_norm'] = (df_exploit['reward'] - df_exploit['reward'].mean()) / df_exploit['reward'].std()
    df_exploit['trial_norm'] = (df_exploit['trial'] - df_exploit['trial'].mean()) / df_exploit['trial'].std()
    
    print(f"Building Logistic Regression Model with Bambi ({len(df_exploit)} samples)...")
    
    # Define model with Bambi
    model = bmb.Model("target_global ~ trial_norm * reward_norm", df_exploit, family="bernoulli")
    fitted = model.fit(draws=2000, tune=1000, chains=4, cores=4)
    
    print("\nLogistic Regression Summary:")
    summary = az.summary(fitted)
    print(summary)
    summary.to_csv(os.path.join(out_dir, "transition_prediction_summary.csv"))
    
    # Plotting
    plt.rcParams['figure.figsize'] = (5, 3)
    plt.rcParams['font.size'] = 10
    az.plot_forest(fitted, combined=True)
    plt.axvline(0, color='r', linestyle='--', alpha=0.5)
    plt.title("Transition Predictors (Local -> Global)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "transition_prediction_forest.png"), bbox_inches='tight')
    plt.close('all')
    
    print("\nGenerating Posterior Predictive Plot...")
    
    # Get original scales
    r_mean = df_exploit['reward'].mean()
    r_std = df_exploit['reward'].std()
    t_mean = df_exploit['trial'].mean()
    t_std = df_exploit['trial'].std()
    
    # Create grids
    reward_grid_norm = np.linspace(df_exploit['reward_norm'].min(), df_exploit['reward_norm'].max(), 50)
    reward_grid_orig = reward_grid_norm * r_std + r_mean
    
    trial_grid_norm = np.linspace(df_exploit['trial_norm'].min(), df_exploit['trial_norm'].max(), 50)
    trial_grid_orig = trial_grid_norm * t_std + t_mean
    
    plt.rcParams['figure.figsize'] = (15, 5)
    fig, axes = plt.subplots(1, 3)
    
    # Extract posterior means
    b0 = fitted.posterior['Intercept'].mean().values
    b_t = fitted.posterior['trial_norm'].mean().values
    b_r = fitted.posterior['reward_norm'].mean().values
    b_int = fitted.posterior['trial_norm:reward_norm'].mean().values
    
    # 1. Marginal Reward (Trial = mean = 0 in norm space)
    logits_r = b0 + b_r * reward_grid_norm
    p_r = 1 / (1 + np.exp(-logits_r))
    axes[0].plot(reward_grid_orig, p_r, color='black', linewidth=2)
    post_logits_r = fitted.posterior['Intercept'].values[..., None] + fitted.posterior['reward_norm'].values[..., None] * reward_grid_norm
    post_logits_r = post_logits_r.reshape(-1, len(reward_grid_norm))
    post_p_r = 1 / (1 + np.exp(-post_logits_r))
    axes[0].fill_between(reward_grid_orig, np.percentile(post_p_r, 2.5, axis=0), np.percentile(post_p_r, 97.5, axis=0), color='black', alpha=0.15)
    axes[0].set_title("Marginal: Reward", fontsize=14)
    axes[0].set_xlabel("Reward (t-1)", fontsize=12)
    axes[0].set_ylabel("P(Transition to Global)", fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # 2. Marginal Trial (Reward = mean = 0 in norm space)
    logits_t = b0 + b_t * trial_grid_norm
    p_t = 1 / (1 + np.exp(-logits_t))
    axes[1].plot(trial_grid_orig, p_t, color='black', linewidth=2)
    post_logits_t = fitted.posterior['Intercept'].values[..., None] + fitted.posterior['trial_norm'].values[..., None] * trial_grid_norm
    post_logits_t = post_logits_t.reshape(-1, len(trial_grid_norm))
    post_p_t = 1 / (1 + np.exp(-post_logits_t))
    axes[1].fill_between(trial_grid_orig, np.percentile(post_p_t, 2.5, axis=0), np.percentile(post_p_t, 97.5, axis=0), color='black', alpha=0.15)
    axes[1].set_title("Marginal: Trial", fontsize=14)
    axes[1].set_xlabel("Trial", fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # 3. Interaction: Reward sliced by Trial Quantiles
    trial_p25 = np.percentile(df_exploit['trial_norm'], 25)
    trial_p50 = np.percentile(df_exploit['trial_norm'], 50)
    trial_p75 = np.percentile(df_exploit['trial_norm'], 75)
    
    for trial_val, label, color in zip([trial_p25, trial_p50, trial_p75], 
                                       ['Early Trial (25th %)', 'Mid Trial (50th %)', 'Late Trial (75th %)'],
                                       ['blue', 'purple', 'red']):
        
        logits = b0 + b_t * trial_val + b_r * reward_grid_norm + b_int * trial_val * reward_grid_norm
        p_transition = 1 / (1 + np.exp(-logits))
        axes[2].plot(reward_grid_orig, p_transition, label=label, color=color, linewidth=2)
        
        post_logits = (
            fitted.posterior['Intercept'].values[..., None] + 
            fitted.posterior['trial_norm'].values[..., None] * trial_val +
            fitted.posterior['reward_norm'].values[..., None] * reward_grid_norm +
            fitted.posterior['trial_norm:reward_norm'].values[..., None] * (trial_val * reward_grid_norm)
        )
        post_logits = post_logits.reshape(-1, len(reward_grid_norm))
        post_p = 1 / (1 + np.exp(-post_logits))
        
        axes[2].fill_between(reward_grid_orig, np.percentile(post_p, 2.5, axis=0), np.percentile(post_p, 97.5, axis=0), color=color, alpha=0.15)
        
    axes[2].set_title("Interaction: Reward × Trial Stage", fontsize=14)
    axes[2].set_xlabel("Reward (t-1)", fontsize=12)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "transition_prediction_plot.png"), bbox_inches='tight')
    plt.close('all')
    
    print("Transition prediction model completed.")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python transition_prediction_model.py <path_to_viterbi_states.csv>")
        sys.exit(1)
    predict_transitions(sys.argv[1])
