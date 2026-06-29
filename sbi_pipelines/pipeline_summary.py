import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sbi.inference import SNPE
from sbi.utils import BoxUniform
import sbi.neural_nets.embedding_nets as emb
from sbi.neural_nets import posterior_nn
from sbi.analysis import pairplot
from sbi.diagnostics import run_tarp, check_tarp
from sbi.analysis import plot_tarp

def compute_advanced_summaries(choices, rewards):
    # choices: (N, 7, 15, 4, 2)
    # rewards: (N, 7, 15, 4)
    N, R, S, A, _ = choices.shape
    
    # metrics shape: (N, R, S, A, 13)
    metrics = np.zeros((N, R, S, A, 13), dtype=np.float32)
    
    for i in range(N):
        for r in range(R):
            for a in range(A):
                c = choices[i, r, :, a, :] # (15, 2)
                rew = rewards[i, r, :, a] # (15,)
                
                # Identify neighbor indices (all agents except a)
                neighbors_c = np.delete(choices[i, r], a, axis=1) # (15, 3, 2)
                neighbors_rew = np.delete(rewards[i, r], a, axis=1) # (15, 3)
                
                max_reward = -np.inf
                max_reward_coord = c[0]
                visited = set()
                revisits = 0
                
                for t in range(S):
                    # 1. previous reward
                    prev_rew = rew[t-1] if t > 0 else 0.0
                    
                    # 2. maximum reward achieved so far (excluding current)
                    # wait, usually it's before the choice or including? Let's include up to t
                    curr_rew = rew[t]
                    if curr_rew > max_reward:
                        max_reward = curr_rew
                        max_reward_coord = c[t]
                        
                    # 3. running average
                    run_avg = np.mean(rew[:t+1])
                    
                    # 4. search distance
                    dist = np.linalg.norm(c[t] - c[t-1]) if t > 0 else 0.0
                    
                    # 5. minimum search distance
                    # min over all previous steps
                    dists = [np.linalg.norm(c[k] - c[k-1]) for k in range(1, t+1)] if t > 0 else [0.0]
                    min_dist = np.min(dists)
                    
                    # 7. number of uncovered tiles
                    coord_tuple = tuple(c[t])
                    if coord_tuple in visited:
                        is_revisit = 1.0
                        revisits += 1
                    else:
                        is_revisit = 0.0
                        visited.add(coord_tuple)
                    num_uncovered = len(visited)
                    
                    # 8. distance from highest discovered reward
                    dist_from_max = np.linalg.norm(c[t] - max_reward_coord)
                    
                    # 9. repeat visits (binary flag)
                    # 10. proportion of revisits
                    prop_revisits = revisits / (t + 1)
                    
                    # Social features
                    if t > 0:
                        prev_soc_c = neighbors_c[t-1] # (3, 2)
                        dists_to_soc = np.linalg.norm(c[t] - prev_soc_c, axis=1)
                        min_dist_social = np.min(dists_to_soc)
                        avg_dist_social = np.mean(dists_to_soc)
                        max_soc_rew = np.max(neighbors_rew[t-1])
                    else:
                        min_dist_social = 0.0
                        avg_dist_social = 0.0
                        max_soc_rew = 0.0
                    
                    metrics[i, r, t, a, 0] = prev_rew
                    metrics[i, r, t, a, 1] = max_reward
                    metrics[i, r, t, a, 2] = run_avg
                    metrics[i, r, t, a, 3] = dist
                    metrics[i, r, t, a, 4] = min_dist
                    metrics[i, r, t, a, 6] = num_uncovered
                    metrics[i, r, t, a, 7] = dist_from_max
                    metrics[i, r, t, a, 8] = is_revisit
                    metrics[i, r, t, a, 9] = prop_revisits
                    metrics[i, r, t, a, 10] = min_dist_social
                    metrics[i, r, t, a, 11] = avg_dist_social
                    metrics[i, r, t, a, 12] = max_soc_rew
                    
                # 6. mean running average
                run_avgs = [np.mean(rew[:k+1]) for k in range(S)]
                mean_run_avg = np.mean(run_avgs)
                metrics[i, r, :, a, 5] = mean_run_avg
                
    # average across agents (axis=3)
    avg_metrics = np.mean(metrics, axis=3) # (N, 7, 15, 10)
    return avg_metrics.reshape(N, -1)

def load_data(data_path):
    data = torch.load(data_path)
    c = data['choices'].numpy()
    r = data['rewards'].numpy()
    summaries = compute_advanced_summaries(c, r)
    return data['theta'], torch.tensor(summaries, dtype=torch.float32)

def train_npe(theta, x, prior):
    print("Training NPE with Summary Stats...")
    # input_dim = 15 steps * 7 stats = 105
    embedding_net = emb.FCEmbedding(input_dim=x.shape[1], output_dim=21)
    neural_posterior = posterior_nn(model="nsf", embedding_net=embedding_net)
    inference = SNPE(prior=prior, density_estimator=neural_posterior)
    posterior_net = inference.append_simulations(theta, x).train(show_train_summary=True)
    return inference.build_posterior(posterior_net)

def evaluate_recovery(posterior, theta_test, x_test, n_samples=10_000):
    print("Evaluating Parameter Recovery...")
    results = []
    
    # Evaluate for each test sample
    for i in range(len(theta_test)):
        _theta = theta_test[i]
        _x = x_test[i]
        
        samples = posterior.sample((n_samples,), x=_x, show_progress_bars=False)
        samples_np = samples.cpu().numpy()
        
        for param_idx in range(len(_theta)):
            param_samples = samples_np[:, param_idx]
            mean_val = np.mean(param_samples)
            hpdi = np.quantile(param_samples, [0.05, 0.95])
            
            results.append({
                'test_idx': i,
                'parameter': param_idx,
                'true_value': _theta[param_idx].item(),
                'mean_recovered': mean_val,
                'hpdi_lower': hpdi[0],
                'hpdi_upper': hpdi[1]
            })
            
    return pd.DataFrame(results)

def plot_recovery(df_results, param_names, out_path):
    df_results['parameter_name'] = df_results['parameter'].map(dict(enumerate(param_names)))
    g = sns.FacetGrid(df_results, col="parameter_name", col_wrap=4, height=4, sharex=False, sharey=False, col_order=param_names)
    g.map_dataframe(
        lambda data, color: plt.errorbar(
            data['true_value'],
            data['mean_recovered'],
            yerr=[data['mean_recovered'] - data['hpdi_lower'], data['hpdi_upper'] - data['mean_recovered']],
            fmt='o', alpha=0.5, capsize=0, color=color
        )
    )
    for ax in g.axes.flat:
        xlim = ax.get_xlim()
        ax.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], 'k--', alpha=0.6)
        ax.set_xlabel("True Parameter")
        ax.set_ylabel("Recovered Mean")
        
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--out_dir", type=str, required=True, help="Path to output directory")
    args = parser.parse_args()
    
    data_dir = args.data_dir
    out_dir = os.path.join(args.out_dir, "pipeline_summary")
    os.makedirs(out_dir, exist_ok=True)
    
    lb = [0.1, 0.01, 0.005, 0.0]
    ub = [5.0, 2.0,  0.1,   1.0]
    prior = BoxUniform(low=torch.tensor(lb), high=torch.tensor(ub))
    param_names = [r"$\lambda$", r"$\beta$", r"$\tau$", r"$\alpha$"]
    
    theta_train, x_train = load_data(os.path.join(data_dir, "train_data.pt"))
    theta_test, x_test = load_data(os.path.join(data_dir, "test_data.pt"))
    
    # Train
    posterior = train_npe(theta_train, x_train, prior)
    
    # Save posterior
    torch.save(posterior, os.path.join(out_dir, "posterior_summary.pt"))
    
    # Evaluate
    df_results = evaluate_recovery(posterior, theta_test, x_test)
    df_results.to_csv(os.path.join(out_dir, "recovery_summary.csv"), index=False)
    
    # Plot
    plot_recovery(df_results, param_names, os.path.join(out_dir, "plot_recovery_summary.png"))
    
    # Posterior Diagnostics (TARP)
    print("Running TARP diagnostics...")
    ecp, alpha_tarp = run_tarp(
        theta_test,
        x_test,
        posterior,
        num_workers=-1,
        num_posterior_samples=1000
    )
    atc, ks_pval = check_tarp(ecp, alpha_tarp)
    print(f"TARP ATC: {atc:.4f} (Should be close to 0)")
    print(f"TARP KS p-val: {ks_pval:.4f} (Should be > 0.05)")
    
    fig, ax = plot_tarp(ecp, alpha_tarp)
    plt.savefig(os.path.join(out_dir, "tarp_summary.png"))
    plt.close()
    
    # Simulation-Based Calibration (SBC)
    try:
        from sbi.diagnostics import run_sbc, check_sbc
        print("Running SBC diagnostics...")
        ranks, dap_samples = run_sbc(theta_test, x_test, posterior, num_posterior_samples=1000)
        sbc_checks = check_sbc(ranks, theta_test, dap_samples, num_posterior_samples=1000)
        print("SBC KS p-values:", sbc_checks['ks_pvals'])
        
        # Plot SBC ranks
        fig, axes = plt.subplots(1, ranks.shape[1], figsize=(12, 3))
        if ranks.shape[1] == 1:
            axes = [axes]
        for i in range(ranks.shape[1]):
            sns.histplot(ranks[:, i].numpy(), bins=20, ax=axes[i], stat='density', color='gray')
            axes[i].set_title(param_names[i])
            axes[i].axhline(1.0 / 1000.0, color='red', linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "sbc_summary.png"))
        plt.close()
    except Exception as e:
        import traceback
        print("SBC diagnostics failed or not available:")
        traceback.print_exc()
    
    print("Done Pipeline Summary!")
