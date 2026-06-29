import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sbi_pipelines.simulate import parallel_simulate
from sbi_pipelines.pipeline_summary import compute_advanced_summaries, load_data as load_data_summary
from sbi_pipelines.pipeline_cnn import prepare_cnn_data, RecurrentCNNEmbedding

import __main__
__main__.RecurrentCNNEmbedding = RecurrentCNNEmbedding

def run_ppc(data_dir, res_dir, n_test_samples=10, n_posterior_samples=50, n_jobs=20):
    print("Running Posterior Predictive Checks (PPC)...")
    out_dir = os.path.join(res_dir, "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    
    test_data = torch.load(os.path.join(data_dir, "test_data.pt"), weights_only=False)
    
    # Load posteriors
    post_cnn = torch.load(os.path.join(res_dir, "pipeline_cnn", "posterior_cnn.pt"), weights_only=False)
    post_sum = torch.load(os.path.join(res_dir, "pipeline_summary", "posterior_summary.pt"), weights_only=False)
    
    # Get x_test for each pipeline
    train_data = torch.load(os.path.join(data_dir, "train_data.pt"), weights_only=False)
    _, _, fit_stats = prepare_cnn_data(train_data)
    _, x_test_cnn, _ = prepare_cnn_data(test_data, fit_stats)
    _, x_test_sum = load_data_summary(os.path.join(data_dir, "test_data.pt"))
    
    # True summaries
    true_c = test_data['choices'].numpy()
    true_r = test_data['rewards'].numpy()
    true_summaries_all = compute_advanced_summaries(true_c, true_r) # shape (N, 7 * 15 * 10)
    
    # Reshape true summaries to get per-feature averages across round/step
    # shape: (N, 7, 15, 10) -> mean over rounds and steps -> (N, 10)
    true_sum_reshaped = true_summaries_all.reshape(len(true_c), 7, 15, 10)
    true_scalar_metrics = true_sum_reshaped.mean(axis=(1, 2)) # (N, 10)
    
    metric_names = [
        "Prev Reward", "Max Reward", "Run Avg", "Search Dist", "Min Search Dist",
        "Mean Run Avg", "Uncovered Tiles", "Dist to Max", "Revisit Flag", "Prop Revisits"
    ]
    
    # Pick first N test samples
    indices = np.arange(n_test_samples)
    
    for i in indices:
        print(f"PPC for test sample {i}...")
        x_c = x_test_cnn[i]
        x_s = x_test_sum[i]
        
        # Sample theta
        theta_cnn = post_cnn.sample((n_posterior_samples,), x=x_c, show_progress_bars=False)
        theta_sum = post_sum.sample((n_posterior_samples,), x=x_s, show_progress_bars=False)
        
        # Simulate
        # Note: parallel_simulate expects shape (N, 4). theta_cnn is (N, 4)
        print("  Simulating CNN posteriors...")
        c_cnn, r_cnn, _, _ = parallel_simulate(theta_cnn, n_jobs=n_jobs)
        print("  Simulating Summary posteriors...")
        c_sum, r_sum, _, _ = parallel_simulate(theta_sum, n_jobs=n_jobs)
        
        # Compute stats
        sum_cnn = compute_advanced_summaries(c_cnn, r_cnn).reshape(n_posterior_samples, 7, 15, 10).mean(axis=(1, 2))
        sum_sum = compute_advanced_summaries(c_sum, r_sum).reshape(n_posterior_samples, 7, 15, 10).mean(axis=(1, 2))
        
        true_vals = true_scalar_metrics[i]
        
        # Plot
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        for m_idx in range(10):
            sns.kdeplot(sum_cnn[:, m_idx], ax=axes[m_idx], color='blue', label='CNN NPE')
            sns.kdeplot(sum_sum[:, m_idx], ax=axes[m_idx], color='green', label='Summary NPE')
            axes[m_idx].axvline(true_vals[m_idx], color='red', linestyle='--', label='True')
            axes[m_idx].set_title(metric_names[m_idx])
            if m_idx == 0:
                axes[m_idx].legend()
                
        plt.suptitle(f"Posterior Predictive Check - Test Sample {i}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"ppc_sample_{i}.png"))
        plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--res_dir", type=str, required=True, help="Path to results directory")
    args = parser.parse_args()
    
    run_ppc(args.data_dir, args.res_dir)
