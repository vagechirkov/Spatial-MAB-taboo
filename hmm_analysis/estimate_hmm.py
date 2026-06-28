import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
from dotenv import load_dotenv

import pytensor
import pytensor.tensor as pt
import pymc as pm

load_dotenv()

def choice_to_coord(choice, grid_size=11):
    """Convert 1D choice index to 2D coordinates."""
    x = choice % grid_size
    y = choice // grid_size
    return np.array([x, y])

def extract_sequences(df):
    """
    Extract jump lengths and rewards for each trial.
    Returns:
        sequences: list of dicts with 'jumps', 'rewards', 'coords', 'group', 'agent', 'trial'
    """
    sequences = []
    
    grouped = df.groupby(['group', 'agent', 'round'])
    
    for name, group_df in grouped:
        group_df = group_df.sort_values('trial')
        
        choices = group_df['choice'].values
        rewards = group_df['reward'].values
        coords = np.array([choice_to_coord(c) for c in choices])
        
        if len(coords) > 1:
            diffs = np.diff(coords, axis=0)
            distances = np.linalg.norm(diffs, axis=1)
            distances = np.where(distances == 0, 1e-3, distances)
            
            # We match rewards to the jumps (reward of the destination choice)
            seq_rewards = rewards[1:]
            
            sequences.append({
                'group': name[0],
                'agent': name[1],
                'round': name[2],
                'env': int(group_df['env'].iloc[0]),
                'jumps': distances,
                'rewards': seq_rewards,
                'coords': coords
            })
            
    return sequences

def hmm_logp_func(jumps_norm, rewards_norm, mask, p_init, p_trans, alpha_j, beta_j, alpha_r, beta_r):
    """Forward algorithm for log-likelihood in PyMC using padded sequences"""
    # jumps, rewards, mask have shape (max_len, n_seqs)
    
    y_j = pt.shape_padright(jumps_norm) # (max_len, n_seqs, 1)
    y_r = pt.shape_padright(rewards_norm)
    mask_tensor = pt.as_tensor_variable(mask)
    
    # Emission probabilities for both jump length and reward
    logp_j = pm.logp(pm.Beta.dist(alpha=alpha_j, beta=beta_j), y_j)
    logp_r = pm.logp(pm.Beta.dist(alpha=alpha_r, beta=beta_r), y_r)
    
    emission_logp = logp_j + logp_r # (max_len, n_seqs, n_states)
    
    def step(logp_emission_t, mask_t, prev_log_alpha, p_trans):
        p_trans_safe = pt.clip(p_trans, 1e-10, 1.0)
        log_p_trans = pt.log(p_trans_safe)
        term = pt.shape_padright(prev_log_alpha) + log_p_trans
        max_term = pt.max(term, axis=1, keepdims=True)
        log_sum = max_term[:, 0, :] + pt.log(pt.sum(pt.exp(term - max_term), axis=1))
        
        curr_log_alpha = log_sum + logp_emission_t
        
        # Apply mask: if padded (mask_t == 0), freeze the log_alpha state
        mask_t_expanded = pt.shape_padright(mask_t)
        curr_log_alpha = pt.switch(mask_t_expanded, curr_log_alpha, prev_log_alpha)
        return curr_log_alpha
    
    first_log_alpha = pt.log(p_init) + emission_logp[0]
    
    log_alpha = pytensor.scan(
        fn=step,
        sequences=[emission_logp[1:], mask_tensor[1:]],
        outputs_info=[first_log_alpha],
        non_sequences=[p_trans],
        strict=True,
        return_updates=False
    )
    
    # The final state of log_alpha is at the end since padded steps are frozen
    final_log_alpha = log_alpha[-1] # (n_seqs, n_states)
    
    max_final = pt.max(final_log_alpha, axis=1, keepdims=True)
    log_prob_seqs = max_final[:, 0] + pt.log(pt.sum(pt.exp(final_log_alpha - max_final), axis=1))
    
    return log_prob_seqs

def viterbi(jumps, rewards, max_jump, max_rew, p_init, p_trans, alpha_j, beta_j, alpha_r, beta_r):
    """
    Viterbi decoding to find the most likely state sequence.
    (Executed outside PyMC using NumPy after MAP estimation).
    """
    from scipy.stats import beta as beta_dist
    import numpy as np
    n_steps = len(jumps)
    n_states = 2
    
    # Normalize inputs exactly like the PyMC model
    j_norm = np.clip(jumps / max_jump, 1e-4, 1 - 1e-4)
    r_norm = np.clip(rewards / max_rew, 1e-4, 1 - 1e-4)
    
    log_p_trans = np.log(p_trans + 1e-12)
    log_p_init = np.log(p_init + 1e-12)
    
    # Precompute log emissions using Beta dist
    log_emissions = np.zeros((n_steps, n_states))
    for k in range(n_states):
        log_j = beta_dist.logpdf(j_norm, a=alpha_j[k], b=beta_j[k])
        log_r = beta_dist.logpdf(r_norm, a=alpha_r[k], b=beta_r[k])
        log_emissions[:, k] = log_j + log_r
        
    T1 = np.zeros((n_states, n_steps))
    T2 = np.zeros((n_states, n_steps), dtype=int)
    
    T1[:, 0] = log_p_init + log_emissions[0, :]
    
    for i in range(1, n_steps):
        for j in range(n_states):
            probs = T1[:, i-1] + log_p_trans[:, j] + log_emissions[i, j]
            T1[j, i] = np.max(probs)
            T2[j, i] = np.argmax(probs)
            
    states = np.zeros(n_steps, dtype=int)
    states[-1] = np.argmax(T1[:, -1])
    for i in range(n_steps-2, -1, -1):
        states[i] = T2[states[i+1], i+1]
        
    return states

def plot_state_characteristics(sequences, map_params, out_dir="hmm_analysis/plots"):
    """Visualize empirical distributions of step lengths and rewards per inferred state, and empirical transitions."""
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import pandas as pd
    import seaborn as sns
    
    os.makedirs(out_dir, exist_ok=True)
    
    p_init = map_params['p_init']
    p_trans = map_params['p_trans']
    alpha_j = map_params['alpha_j']
    beta_j = map_params['beta_j']
    alpha_r = map_params['alpha_r']
    beta_r = map_params['beta_r']
    max_jump = map_params['max_jump']
    max_rew = map_params['max_rew']
    
    
    all_jumps_0, all_jumps_1 = [], []
    all_rewards_0, all_rewards_1 = [], []
    all_times_0, all_times_1 = [], []
    
    trans_data = []
    
    # Color-blind friendly scheme
    c0 = '#0072B2' # Blue for Exploitation
    c1 = '#D55E00' # Vermilion for Exploration
    
    for seq in sequences:
        # viterbi needs to use Normal distribution for jumps now
        states = viterbi(
            seq['jumps'], seq['rewards'], max_jump, max_rew, p_init, p_trans, 
            alpha_j, beta_j, alpha_r, beta_r
        )
        jumps = seq['jumps']
        rewards = seq['rewards']
        times = np.arange(len(states))
        
        all_jumps_0.extend(jumps[states == 0])
        all_jumps_1.extend(jumps[states == 1])
        all_rewards_0.extend(rewards[states == 0])
        all_rewards_1.extend(rewards[states == 1])
        all_times_0.extend(times[states == 0])
        all_times_1.extend(times[states == 1])
        
        for t in range(len(states) - 1):
            trans_data.append({
                'state_t': states[t],
                'state_t1': states[t+1],
                'jump_t': jumps[t],
                'reward_t': rewards[t],
                'time_t': t
            })
            
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].hist(all_jumps_0, bins=10, alpha=0.6, color=c0, density=True, label='Exploitation')
    axes[0].hist(all_jumps_1, bins=10, alpha=0.6, color=c1, density=True, label='Exploration')
    axes[0].set_title("Jump Length Density by State")
    axes[0].set_xlabel("Jump Length")
    axes[0].legend()
    
    axes[1].hist(all_rewards_0, bins=10, alpha=0.6, color=c0, density=True, label='Exploitation')
    axes[1].hist(all_rewards_1, bins=10, alpha=0.6, color=c1, density=True, label='Exploration')
    axes[1].set_title("Reward Density by State")
    axes[1].set_xlabel("Reward")
    axes[1].legend()
    
    axes[2].hist(all_times_0, bins=8, alpha=0.6, color=c0, density=True, label='Exploitation')
    axes[2].hist(all_times_1, bins=8, alpha=0.6, color=c1, density=True, label='Exploration')
    axes[2].set_title("Trial Time Density by State")
    axes[2].set_xlabel("Trial ID in sequence")
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "state_characteristics.png"), bbox_inches='tight')
    plt.close(fig)
    
    # Empirical Transition Plot (2 rows: Exploit->Explore, Explore->Exploit)
    df_trans = pd.DataFrame(trans_data)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Row 1: Exploit (0) -> Explore (1)
    df_exploit = df_trans[df_trans['state_t'] == 0].copy()
    if len(df_exploit) > 0:
        df_exploit['switch'] = (df_exploit['state_t1'] == 1).astype(int)
        
        df_exploit['reward_bin'] = pd.qcut(df_exploit['reward_t'], q=5, duplicates='drop')
        sns.pointplot(data=df_exploit, x='reward_bin', y='switch', ax=axes[0, 0], color=c0)
        baseline_01 = p_trans[0, 1]
        axes[0, 0].axhline(baseline_01, ls='--', color=c0, alpha=0.7, label=f'Baseline P={baseline_01:.2f}')
        axes[0, 0].legend()
        axes[0, 0].set_title("P(Exploit -> Explore) vs Reward")
        axes[0, 0].set_ylabel("P(Switch)")
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        df_exploit['jump_bin'] = pd.qcut(df_exploit['jump_t'], q=5, duplicates='drop')
        sns.pointplot(data=df_exploit, x='jump_bin', y='switch', ax=axes[0, 1], color=c0)
        axes[0, 1].axhline(baseline_01, ls='--', color=c0, alpha=0.7)
        axes[0, 1].set_title("P(Exploit -> Explore) vs Jump Length")
        axes[0, 1].set_ylabel("P(Switch)")
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        sns.pointplot(data=df_exploit, x='time_t', y='switch', ax=axes[0, 2], color=c0)
        axes[0, 2].axhline(baseline_01, ls='--', color=c0, alpha=0.7)
        axes[0, 2].set_title("P(Exploit -> Explore) vs Time in Round")
        axes[0, 2].set_ylabel("P(Switch)")
        
    # Row 2: Explore (1) -> Exploit (0)
    df_explore = df_trans[df_trans['state_t'] == 1].copy()
    if len(df_explore) > 0:
        df_explore['switch'] = (df_explore['state_t1'] == 0).astype(int)
        
        # Use quantiles or fixed bins if not enough unique values
        try:
            df_explore['reward_bin'] = pd.qcut(df_explore['reward_t'], q=5, duplicates='drop')
        except ValueError:
            df_explore['reward_bin'] = pd.cut(df_explore['reward_t'], bins=5)
            
        sns.pointplot(data=df_explore, x='reward_bin', y='switch', ax=axes[1, 0], color=c1)
        baseline_10 = p_trans[1, 0]
        axes[1, 0].axhline(baseline_10, ls='--', color=c1, alpha=0.7, label=f'Baseline P={baseline_10:.2f}')
        axes[1, 0].legend()
        axes[1, 0].set_title("P(Explore -> Exploit) vs Reward")
        axes[1, 0].set_ylabel("P(Switch)")
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        try:
            df_explore['jump_bin'] = pd.qcut(df_explore['jump_t'], q=5, duplicates='drop')
        except ValueError:
            df_explore['jump_bin'] = pd.cut(df_explore['jump_t'], bins=5)
            
        sns.pointplot(data=df_explore, x='jump_bin', y='switch', ax=axes[1, 1], color=c1)
        axes[1, 1].axhline(baseline_10, ls='--', color=c1, alpha=0.7)
        axes[1, 1].set_title("P(Explore -> Exploit) vs Jump Length")
        axes[1, 1].set_ylabel("P(Switch)")
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        sns.pointplot(data=df_explore, x='time_t', y='switch', ax=axes[1, 2], color=c1)
        axes[1, 2].axhline(baseline_10, ls='--', color=c1, alpha=0.7)
        axes[1, 2].set_title("P(Explore -> Exploit) vs Time in Round")
        axes[1, 2].set_ylabel("P(Switch)")
        
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "empirical_transitions.png"), bbox_inches='tight')
    plt.close(fig)

def plot_trajectories(sequences, map_params, out_dir="hmm_analysis/plots"):
    """Plot sequences colored by predicted state using 7-round grid with env backgrounds."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    import os
    os.makedirs(out_dir, exist_ok=True)
    
    import sys
    sys.path.append('.') 
    from sbi_pipelines.estimate_human_4d import load_environments
    envs = load_environments()
    
    p_init = map_params['p_init']
    p_trans = map_params['p_trans']
    alpha_j = map_params['alpha_j']
    beta_j = map_params['beta_j']
    alpha_r = map_params['alpha_r']
    beta_r = map_params['beta_r']
    max_jump = map_params['max_jump']
    max_rew = map_params['max_rew']
    
    
    markers = ['o', '^'] # 0: Exploit, 1: Explore
    c0 = '#0072B2' # Blue for Exploitation
    c1 = '#D55E00' # Vermilion for Exploration
    colors = [c0, c1]
    
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
        param_text = f"P(Exploit->Exploit): {np.round(p_trans[0,0],2)} | P(Explore->Explore): {np.round(p_trans[1,1],2)}"
        fig.text(0.5, 0.98, param_text, ha='center', fontsize=18, color='black')
        
        for row_idx, agent_id in enumerate(agents):
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
                # use normal params for viterbi now
                states = viterbi(
                    seq['jumps'], seq['rewards'], max_jump, max_rew, p_init, p_trans, 
                    alpha_j, beta_j, alpha_r, beta_r
                )
                padded_states = np.concatenate([[states[0]], states])
                
                px_list = coords[:, 0]
                py_list = coords[:, 1]
                
                # Plot the path line
                ax.plot(px_list, py_list, color='gray', linewidth=1.5, alpha=0.6, zorder=1)
                
                # Scatter points colored by STATE, shaped by state
                for k in range(2):
                    mask = (padded_states == k)
                    if np.any(mask):
                        sc = ax.scatter(coords[mask, 0], coords[mask, 1], c=colors[k], 
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

def main():
    import sys
    import pandas as pd
    import numpy as np
    import pymc as pm
    import pytensor.tensor as pt
    import os
    
    # Check for correct data file
    human_data_path = 'data/human_data/e1_data_extended.csv'
    if not os.path.exists(human_data_path):
        human_data_path = '/groups/romanczuk/Chirkov_mab/sbi/gp_ucb_vs_06_2026/human_data/e1_data_extended.csv'
        
    print(f"Loading data from {human_data_path}")
    if not os.path.exists(human_data_path):
        print("Data file not found!")
        sys.exit(1)
        
    df = pd.read_csv(human_data_path)
    sequences = extract_sequences(df)
    print(f"Extracted {len(sequences)} valid trial sequences.")
    
    # Extract sequences for 3 full groups
    unique_groups = []
    for seq in sequences:
        if seq['group'] not in unique_groups:
            unique_groups.append(seq['group'])
    target_groups = unique_groups[:] # 3
    use_seqs = [seq for seq in sequences if seq['group'] in target_groups]
    print(f"Using {len(use_seqs)} sequences (3 full groups) for the HMM estimation.")
    
    # Create padded arrays
    max_len = max(len(s['jumps']) for s in use_seqs)
    n_seqs = len(use_seqs)
    
    jumps_mat = np.zeros((max_len, n_seqs))
    rewards_mat = np.zeros((max_len, n_seqs))
    mask_mat = np.zeros((max_len, n_seqs), dtype=bool)
    
    for i, seq in enumerate(use_seqs):
        L = len(seq['jumps'])
        jumps_mat[:L, i] = seq['jumps']
        rewards_mat[:L, i] = seq['rewards']
        mask_mat[:L, i] = True
        
    # Global normalization to [1e-4, 1 - 1e-4]
    max_jump = np.max(jumps_mat) + 1e-3
    max_rew = np.max(rewards_mat) + 1e-3
    
    jumps_mat_norm = np.clip(jumps_mat / max_jump, 1e-4, 1.0 - 1e-4)
    rewards_mat_norm = np.clip(rewards_mat / max_rew, 1e-4, 1.0 - 1e-4)
        
    with pm.Model() as hmm_model:
        # Transition matrix
        p_trans = pm.Dirichlet('p_trans', a=np.ones((2, 2)), shape=(2, 2))
        p_init = pm.Dirichlet('p_init', a=np.ones(2))
        
        # State 0 (Exploitation): Small Jumps, High Rewards
        # State 1 (Exploration): Large Jumps, Low Rewards
        
        # Jump Means (Beta is [0, 1])
        mu_jump_0 = pm.Uniform('mu_jump_0', 0.0, 0.5)
        mu_jump_1 = pm.Uniform('mu_jump_1', mu_jump_0, 1.0) # Forces mu_jump_1 > mu_jump_0
        mu_jump = pt.stack([mu_jump_0, mu_jump_1])
        
        # Reward Means (Beta is [0, 1])
        mu_rew_1 = pm.Uniform('mu_rew_1', 0.0, 0.5)
        mu_rew_0 = pm.Uniform('mu_rew_0', mu_rew_1, 1.0) # Forces mu_rew_0 > mu_rew_1
        mu_rew = pt.stack([mu_rew_0, mu_rew_1])
        
        # Concentration parameters for Beta
        # Higher kappa = narrower distribution
        kappa_jump = pm.Exponential('kappa_jump', lam=0.1, shape=2)
        kappa_rew = pm.Exponential('kappa_rew', lam=0.1, shape=2)
        
        # Convert Mean/Kappa to Alpha/Beta
        alpha_j = pm.Deterministic('alpha_j', mu_jump * kappa_jump)
        beta_j = pm.Deterministic('beta_j', (1.0 - mu_jump) * kappa_jump)
        
        alpha_r = pm.Deterministic('alpha_r', mu_rew * kappa_rew)
        beta_r = pm.Deterministic('beta_r', (1.0 - mu_rew) * kappa_rew)
        
        # Calculate log-likelihood for all sequences using normalized data
        logp_seqs = hmm_logp_func(
            jumps_mat_norm, rewards_mat_norm, mask_mat, 
            p_init, p_trans, alpha_j, beta_j, alpha_r, beta_r
        )
            
        total_logp = pt.sum(logp_seqs)
        pm.Potential('total_hmm_logp', total_logp)
        pm.Deterministic('seq_logp', logp_seqs)

        print("Starting MCMC sampling...")
        cores = int(os.environ.get('SLURM_CPUS_PER_TASK', 96))
        chains = 20
        print(f"Using {cores} cores and {chains} chains.")
        
        trace = pm.sample(draws=40_000, tune=10_000, chains=chains, cores=cores, 
                          return_inferencedata=True, progressbar=True,
                          init='jitter+adapt_diag', target_accept=0.95)
        
    print("\nEstimation complete. Summary:")
    summary = pm.stats.summary(trace, var_names=['p_trans', 'mu_jump_0', 'mu_jump_1', 'kappa_jump', 'mu_rew_0', 'mu_rew_1', 'kappa_rew'])
    print(summary)
    
    os.makedirs('hmm_analysis/results', exist_ok=True)
    summary.to_csv('hmm_analysis/results/hmm_summary.csv')
    
    print("\nModel Fit Evaluation (LOO):")
    try:
        import arviz_stats
        import xarray as xr
        
        log_likelihood = xr.DataArray(
            trace.posterior['seq_logp'].values, 
            dims=['chain', 'draw', 'sequence']
        )
        trace.add_groups({"log_likelihood": {"total_hmm_logp": log_likelihood}})
        
        loo = arviz_stats.loo(trace)
        print(loo)
    except Exception as e:
        print(f"Could not compute LOO: {e}")
        
    map_params = {
        'p_init': trace.posterior['p_init'].mean(dim=['chain', 'draw']).values,
        'p_trans': trace.posterior['p_trans'].mean(dim=['chain', 'draw']).values,
        'alpha_j': trace.posterior['alpha_j'].mean(dim=['chain', 'draw']).values,
        'beta_j': trace.posterior['beta_j'].mean(dim=['chain', 'draw']).values,
        'alpha_r': trace.posterior['alpha_r'].mean(dim=['chain', 'draw']).values,
        'beta_r': trace.posterior['beta_r'].mean(dim=['chain', 'draw']).values,
        'max_jump': max_jump,
        'max_rew': max_rew
    }
    
    print("\nGenerating model diagram...")
    try:
        g = pm.model_to_graphviz(hmm_model)
        g.render('hmm_analysis/results/model_graph', format='png', cleanup=True)
        print("Model diagram saved to hmm_analysis/results/model_graph.png")
    except Exception as e:
        print(f"Could not generate model diagram: {e}")
        
    import arviz as az
    import matplotlib.pyplot as plt
    az.rcParams['plot.max_subplots'] = 200

    print("Generating trace dist plots...")
    try:
        az.plot_trace_dist(trace, combined=True)
        plt.savefig('hmm_analysis/results/trace_plot.png', bbox_inches='tight')
        plt.close('all')
        print("Trace dist plot saved to hmm_analysis/results/trace_plot.png")
    except Exception as e:
        print(f"Could not generate trace dist plot: {e}")

    print("Generating forest plots...")
    try:
        axes = az.plot_forest(
            trace,
            var_names=["mu_jump_0", "mu_jump_1", "kappa_jump", "mu_rew_0", "mu_rew_1", "kappa_rew", "p_trans"],
            combined=True,
            ci_probs=[0.5, 0.95],
            figsize=(10, 8)
        )
        if hasattr(axes, '__iter__'):
            fig = axes[0].figure
        else:
            fig = axes.figure
        fig.savefig('hmm_analysis/results/forest_plot.png', bbox_inches='tight')
        plt.close(fig)
        print("Forest plot saved to hmm_analysis/results/forest_plot.png")
    except Exception as e:
        print(f"Could not generate forest plot: {e}")
        
    print("\nGenerating Trajectory Visualizations...")
    plot_trajectories(use_seqs, map_params)
    plot_state_characteristics(use_seqs, map_params)
    print("Visualizations saved to hmm_analysis/plots/")

if __name__ == "__main__":
    main()
