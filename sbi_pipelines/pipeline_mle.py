import os
import torch
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from joblib import Parallel, delayed

from abm.agent import value_shaping

def compute_nll_agent_round(params, choices, rewards, social_choices, social_rewards):
    length_scale, beta, tau, alpha = params
    nll = 0.0
    meshgrid_flatten = np.array(np.meshgrid(range(11), range(11)), dtype=np.int32).reshape(2, -1).T
    meshgrid_dict = {tuple(coord): i for i, coord in enumerate(meshgrid_flatten)}
    
    for t in range(1, 15):
        X_priv = choices[:t]
        y_priv = rewards[:t].reshape(-1, 1)
        X_soc = [soc_c[:t] for soc_c in social_choices]
        y_soc = [soc_r[:t].reshape(-1, 1) for soc_r in social_rewards]
        
        value_final = value_shaping(
            X_priv, y_priv, X_soc, y_soc,
            X_predict=meshgrid_flatten,
            length_scale_private=length_scale,
            length_scale_social=length_scale,
            observation_noise_private=0.0001,
            observation_noise_social=0.0001,
            beta_private=beta,
            beta_social=beta,
            alpha=alpha,
            tau=tau,
            random_state=None
        )
        logits = value_final / tau
        logits = np.exp(np.clip(logits, -40, 40))
        probs = logits.ravel()
        probs = np.nan_to_num(probs, nan=1e-12)
        probs += 1e-12
        probs /= probs.sum()
        
        choice = tuple(choices[t])
        idx = meshgrid_dict[choice]
        nll -= np.log(probs[idx] + 1e-12)
        
    return nll

def compute_total_nll(log_params, c_test_item, r_test_item):
    """
    Computes total negative log likelihood.
    log_params: log(lambda), log(beta), log(tau), log(alpha/(1-alpha))
    """
    # Transform back
    length_scale = np.exp(log_params[0])
    beta = np.exp(log_params[1])
    tau = np.exp(log_params[2])
    alpha = 1.0 / (1.0 + np.exp(-log_params[3])) # sigmoid for alpha in [0,1]
    
    params = (length_scale, beta, tau, alpha)
    
    total_nll = 0.0
    for r in range(7):
        for a in range(4):
            choices = c_test_item[r, :, a, :]
            rewards = r_test_item[r, :, a]
            
            social_choices = []
            social_rewards = []
            for soc_a in range(4):
                if soc_a != a:
                    social_choices.append(c_test_item[r, :, soc_a, :])
                    social_rewards.append(r_test_item[r, :, soc_a])
                    
            total_nll += compute_nll_agent_round(params, choices, rewards, social_choices, social_rewards)
    return total_nll

def optimize_one_sample(i, true_theta, c_item, r_item):
    print(f"Starting optimization for sample {i}...")
    # Bounds for log_params:
    # lambda: [0.1, 5.0] -> log(lambda): [-2.3, 1.6]
    # beta: [0.01, 2.0] -> log(beta): [-4.6, 0.7]
    # tau: [0.005, 0.1] -> log(tau): [-5.3, -2.3]
    # alpha: [0.0, 1.0] -> logit(alpha): [-5.0, 5.0] (approx)
    bounds = [(-2.3, 1.6), (-4.6, 0.7), (-5.3, -2.3), (-5.0, 5.0)]
    
    res = opt.differential_evolution(
        compute_total_nll,
        bounds,
        args=(c_item, r_item),
        maxiter=15, # smaller maxiter to save time
        popsize=10, 
        workers=1 # single thread per optimization, we parallelize across samples
    )
    
    best_log_params = res.x
    length_scale = np.exp(best_log_params[0])
    beta = np.exp(best_log_params[1])
    tau = np.exp(best_log_params[2])
    alpha = 1.0 / (1.0 + np.exp(-best_log_params[3]))
    
    recovered = [length_scale, beta, tau, alpha]
    
    results = []
    for param_idx in range(4):
        results.append({
            'test_idx': i,
            'parameter': param_idx,
            'true_value': true_theta[param_idx].item(),
            'mean_recovered': recovered[param_idx],
            'nll': res.fun
        })
    print(f"Sample {i} done. True: {true_theta.numpy()}, Recovered: {recovered}")
    return results

def plot_recovery(df_results, param_names, out_path):
    df_results['parameter_name'] = df_results['parameter'].map(dict(enumerate(param_names)))
    g = sns.FacetGrid(df_results, col="parameter_name", col_wrap=4, height=4, sharex=False, sharey=False)
    g.map_dataframe(
        lambda data, color: plt.scatter(
            data['true_value'],
            data['mean_recovered'],
            alpha=0.5, color=color
        )
    )
    for ax in g.axes.flat:
        xlim = ax.get_xlim()
        ax.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], 'k--', alpha=0.6)
        ax.set_xlabel("True Parameter")
        ax.set_ylabel("Recovered MLE")
        
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
    out_dir = os.path.join(args.out_dir, "pipeline_mle")
    os.makedirs(out_dir, exist_ok=True)
    
    test_data = torch.load(os.path.join(data_dir, "test_data.pt"))
    
    theta_test = test_data['theta']
    c_test = test_data['choices'].numpy()
    r_test = test_data['rewards'].numpy()
    
    # Run MLE on 20 samples
    n_samples_to_eval = 20
    
    out = Parallel(n_jobs=20)(
        delayed(optimize_one_sample)(i, theta_test[i], c_test[i], r_test[i])
        for i in range(n_samples_to_eval)
    )
    
    flat_results = [item for sublist in out for item in sublist]
    df_results = pd.DataFrame(flat_results)
    df_results.to_csv(os.path.join(out_dir, "recovery_mle.csv"), index=False)
    
    param_names = [r"$\lambda$", r"$\beta$", r"$\tau$", r"$\alpha$"]
    plot_recovery(df_results, param_names, os.path.join(out_dir, "plot_recovery_mle.png"))
    
    print("Done Pipeline MLE!")
