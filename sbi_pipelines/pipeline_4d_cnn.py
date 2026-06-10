import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sbi.inference import SNPE
from sbi.utils import BoxUniform
from sbi.neural_nets import posterior_nn
from sbi.diagnostics import run_tarp, check_tarp
from sbi.analysis import plot_tarp

# Bypass sbi's overly-eager z-score warning which causes CUDA OOM on massive tensors
import sbi.utils.sbiutils as sbiutils
import sbi.inference.trainers.npe.npe_base as npe_base
sbiutils.warn_if_invalid_for_zscoring = lambda x: None
npe_base.warn_if_invalid_for_zscoring = lambda x: None

class Embedding4DCNN(nn.Module):
    def __init__(self, output_dim=21):
        super().__init__()
        
        # 3D CNN processes spatio-temporal features natively without manual recurrent loops
        # Input: (batch * 7, Channels=3, Time=15, Height=11, Width=11)
        self.conv3d = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)), # Output: (16, 7, 5, 5)
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)), # Output: (32, 3, 2, 2)
            nn.Flatten(),
            nn.Linear(32 * 3 * 2 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        # x is originally flattened: (batch_size, 7 * 15 * 3 * 11 * 11)
        # Reshape to extract frames: (batch_size, 7, 15, 3, 11, 11)
        x = x.view(batch_size, 7, 15, 3, 11, 11)
        
        # nn.Conv3d expects (N, Channels, Depth, Height, Width)
        # Permute to move Channels (dim 3) before Depth/Time (dim 2)
        x = x.permute(0, 1, 3, 2, 4, 5) # (batch, 7, 3, 15, 11, 11)
        
        # Merge batch and round to process all rounds through Conv3d efficiently
        x = x.reshape(batch_size * 7, 3, 15, 11, 11)
        
        # Extract Spatio-Temporal embeddings
        emb = self.conv3d(x) # (batch * 7, output_dim)
        
        # Reshape back and average over rounds
        emb = emb.view(batch_size, 7, -1)
        final_emb = torch.mean(emb, dim=1)
        
        return final_emb

def prepare_4d_data(data, fit_stats=None):
    # data has choices: (N, 7, 15, 4, 2), rewards: (N, 7, 15, 4), landscapes: (N, 7, 4, 11, 11)
    c = data['choices'].numpy()
    r = data['rewards'].numpy()
    l = data['landscapes'].numpy()
    N, R, S, A, _ = c.shape
    
    if fit_stats is None:
        r_mean, r_std = np.mean(r), np.std(r) + 1e-8
        l_mean, l_std = np.mean(l), np.std(l) + 1e-8
        fit_stats = (r_mean, r_std, l_mean, l_std)
    else:
        r_mean, r_std, l_mean, l_std = fit_stats
        
    x_list = np.zeros((N, R, S, 3, 11, 11), dtype=np.float32)
    
    for i in range(N):
        for rd in range(R):
            land = l[i, rd, 0, :, :]
            land = (land - l_mean) / l_std
            
            for t in range(S):
                # Channel 0: Landscape
                x_list[i, rd, t, 0, :, :] = land
                
                # Channel 1: Private Agent (put 1.0 at location)
                px, py = int(c[i, rd, t, 0, 0]), int(c[i, rd, t, 0, 1])
                x_list[i, rd, t, 1, px, py] = 1.0
                
                # Channel 2: Social Agents (put 1.0 at location)
                for a in range(1, A):
                    sx, sy = int(c[i, rd, t, a, 0]), int(c[i, rd, t, a, 1])
                    x_list[i, rd, t, 2, sx, sy] += 1.0
                    
    x_flat = x_list.reshape(N, -1)
    return data['theta'], torch.tensor(x_flat, dtype=torch.float32), fit_stats

def train_npe(theta, x, prior, chunk_size=5000):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training NPE with 4D CNN on {device}...")
    embedding_net = Embedding4DCNN(output_dim=21).to(device)
    # Disable z_score_x to prevent sbi from destroying spatial structures via independent scaling
    neural_posterior = posterior_nn(model="nsf", embedding_net=embedding_net, z_score_x='none')
    
    inference = SNPE(prior=prior, density_estimator=neural_posterior, device=device)
    posterior_net = None
    
    n_samples = len(theta)
    print(f"Total samples: {n_samples}. Training in chunks of {chunk_size} to avoid OOM.")
    
    for i in range(0, n_samples, chunk_size):
        end_idx = min(i + chunk_size, n_samples)
        print(f"\n--- Training on dataset chunk {i} to {end_idx} ---")
        
        theta_chunk = theta[i:end_idx]
        x_chunk = x[i:end_idx]
        
        # Append only the current chunk
        inference = inference.append_simulations(theta_chunk, x_chunk)
        
        if posterior_net is None:
            posterior_net = inference.train(show_train_summary=True)
        else:
            posterior_net = inference.train(show_train_summary=True, retrain_from_scratch=False)
            
        # Manually clear data from the inference object to avoid OOM by accumulation
        # Accommodates different versions of sbi
        if hasattr(inference, '_theta_bank'):
            inference._theta_bank = []
            inference._x_bank = []
        if hasattr(inference, '_data_theta'):
            inference._data_theta = None
            inference._data_x = None
        if hasattr(inference, '_dataset'):
            inference._dataset = None
            
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
            hpdi = np.quantile(param_samples, [0.1, 0.9])
            
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
        lambda data, color: plt.scatter(
            data['true_value'],
            data['mean_recovered'],
            # yerr=[data['mean_recovered'] - data['hpdi_lower'], data['hpdi_upper'] - data['mean_recovered']],
            # fmt='o', alpha=0.5, capsize=0, 
            color=color
        )
    )

    # g.map_dataframe(
    #     lambda data, color: plt.errorbar(
    #         data['true_value'],
    #         data['mean_recovered'],
    #         yerr=[
    #             (data['mean_recovered'] - data['hpdi_lower']).clip(lower=0),
    #             (data['hpdi_upper'] - data['mean_recovered']).clip(lower=0)
    #         ],
    #         fmt='o', alpha=0.5, capsize=0, color=color
    #     )
    # )
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
    out_dir = os.path.join(args.out_dir, "pipeline_4d_cnn")
    os.makedirs(out_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lb = [0.1, 0.01, 0.005, 0.0]
    ub = [5.0, 2.0,  0.1,   1.0]
    prior = BoxUniform(low=torch.tensor(lb, device=device), high=torch.tensor(ub, device=device), device=device)
    param_names = [r"$\lambda$", r"$\beta$", r"$\tau$", r"$\alpha$"]
    
    train_data = torch.load(os.path.join(data_dir, "train_data.pt"))
    theta_train, x_train, fit_stats = prepare_4d_data(train_data)
    
    test_data = torch.load(os.path.join(data_dir, "test_data.pt"))
    theta_test, x_test, _ = prepare_4d_data(test_data, fit_stats)
    
    # Train
    posterior = train_npe(theta_train, x_train, prior, chunk_size=20_000)
    
    # Save posterior
    torch.save(posterior, os.path.join(out_dir, "posterior_4d_cnn.pt"))
    posterior = torch.load(os.path.join(out_dir, "posterior_4d_cnn.pt"), map_location=device)
    
    theta_test = theta_test.to(device)
    x_test = x_test.to(device)
    
    # Evaluate
    df_results = evaluate_recovery(posterior, theta_test, x_test)
    df_results.to_csv(os.path.join(out_dir, "recovery_4d_cnn.csv"), index=False)
    
    # Plot
    plot_recovery(df_results, param_names, os.path.join(out_dir, "plot_recovery_4d_cnn.png"))
    
    # Posterior Diagnostics (TARP)
    print("Running TARP diagnostics...")
    posterior.to("cpu")
    theta_test_cpu = theta_test.to("cpu")
    x_test_cpu = x_test.to("cpu")
    
    ecp, alpha_tarp = run_tarp(
        theta_test_cpu,
        x_test_cpu,
        posterior,
        num_workers=-1,
        num_posterior_samples=1000
    )
    atc, ks_pval = check_tarp(ecp, alpha_tarp)
    print(f"TARP ATC: {atc:.4f} (Should be close to 0)")
    print(f"TARP KS p-val: {ks_pval:.4f} (Should be > 0.05)")
    
    fig, ax = plot_tarp(ecp, alpha_tarp)
    plt.savefig(os.path.join(out_dir, "tarp_4d_cnn.png"))
    plt.close()
    
    # Simulation-Based Calibration (SBC)
    try:
        from sbi.diagnostics import run_sbc, check_sbc
        print("Running SBC diagnostics...")
        ranks, dap_samples = run_sbc(theta_test_cpu, x_test_cpu, posterior, num_posterior_samples=1000)
        sbc_checks = check_sbc(ranks, theta_test_cpu, dap_samples, num_posterior_samples=1000)
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
        plt.savefig(os.path.join(out_dir, "sbc_4d_cnn.png"))
        plt.close()
    except Exception as e:
        import traceback
        print("SBC diagnostics failed or not available:")
        traceback.print_exc()
    
    print("Done Pipeline 4D CNN!")
