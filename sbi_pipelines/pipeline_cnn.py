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

class TransformerCNNEmbedding(nn.Module):
    def __init__(self, output_dim=21):
        super().__init__()
        
        # CNN for the 11x11 landscape (no pooling to preserve fine-grained details)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 11 * 11, 32),
            nn.ReLU()
        )
        
        # Social Transformer (Cross-Agent Attention)
        # Each agent's input is 3 features: (c_x, c_y, r). Project to d_model=32.
        self.agent_proj = nn.Linear(3, 32)
        social_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64, batch_first=True)
        self.social_transformer = nn.TransformerEncoder(social_layer, num_layers=1)
        
        # Temporal Transformer (Sequence Attention)
        self.pos_encoder = nn.Parameter(torch.randn(1, 15, 32))
        temporal_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64, batch_first=True)
        self.temporal_transformer = nn.TransformerEncoder(temporal_layer, num_layers=2)
        
        # Fully connected layer for the landscape-conditioned sequence token
        self.fc = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        # x shape: (batch_size, 7 * (121 + 15 * 12)) -> we need to reshape it back
        # Let's define the flattened input format:
        # For each of the 7 rounds:
        #   - Landscape: 11x11 = 121
        #   - Sequence: 15 * 12 = 180
        # Total per round = 301
        # Total per sample = 7 * 301 = 2107
        batch_size = x.shape[0]
        x = x.view(batch_size, 7, 301)
        
        round_embeddings = []
        for r in range(7):
            round_data = x[:, r, :]
            
            # Extract landscape and sequence
            landscape = round_data[:, :121].view(batch_size, 1, 11, 11)
            sequence = round_data[:, 121:].view(batch_size, 15, 12)
            
            # Process landscape
            cnn_emb = self.cnn(landscape) # (batch, 32)
            
            # --- Process Social Graph ---
            # sequence contains (batch, 15, 12)
            # The 12 features per step are: priv_c (2), priv_r (1), soc_c (6), soc_r (3)
            # We reshape this into 4 agents (1 private + 3 social), each with 3 features (c_x, c_y, r).
            priv_agent = sequence[:, :, 0:3] # (batch, 15, 3)
            soc1 = torch.cat([sequence[:, :, 3:5], sequence[:, :, 9:10]], dim=2) # (batch, 15, 3)
            soc2 = torch.cat([sequence[:, :, 5:7], sequence[:, :, 10:11]], dim=2) # (batch, 15, 3)
            soc3 = torch.cat([sequence[:, :, 7:9], sequence[:, :, 11:12]], dim=2) # (batch, 15, 3)
            
            # Stack agents into a graph: (batch, 15, 4, 3)
            agents = torch.stack([priv_agent, soc1, soc2, soc3], dim=2)
            
            # Flatten batch and time to process all social graphs in parallel
            agents_flat = agents.view(batch_size * 15, 4, 3)
            
            # Project and run Social Transformer
            agents_proj = self.agent_proj(agents_flat) # (batch*15, 4, 32)
            social_out = self.social_transformer(agents_proj) # (batch*15, 4, 32)
            
            # Extract the updated representation for the private agent (index 0)
            priv_social_out = social_out[:, 0, :] # (batch*15, 32)
            
            # Reshape back to sequence format for Temporal Transformer
            temporal_input = priv_social_out.view(batch_size, 15, 32)
            
            # --- Process Temporal Sequence ---
            # Add positional encoding
            temporal_input = temporal_input + self.pos_encoder # (batch, 15, 32)
            
            # Prepend the landscape embedding as the first token in the sequence
            cnn_token = cnn_emb.unsqueeze(1) # (batch, 1, 32)
            combined_seq = torch.cat([cnn_token, temporal_input], dim=1) # (batch, 16, 32)
            
            # Pass the combined sequence through the temporal transformer
            trans_out = self.temporal_transformer(combined_seq)
            
            # The first token now contains the sequence summary conditioned on the landscape
            emb = self.fc(trans_out[:, 0, :])
            round_embeddings.append(emb)
            
        # Average over rounds
        round_embeddings = torch.stack(round_embeddings, dim=1)
        final_emb = torch.mean(round_embeddings, dim=1)
        return final_emb

def prepare_cnn_data(data, fit_stats=None):
    # data has choices: (N, 7, 15, 4, 2), rewards: (N, 7, 15, 4), landscapes: (N, 7, 4, 11, 11)
    N = data['choices'].shape[0]
    c = data['choices'].numpy()
    r = data['rewards'].numpy()
    l = data['landscapes'].numpy()
    
    # Scale inputs
    c = (c - 5.0) / 5.0  # [0,10] to [-1,1]
    
    if fit_stats is None:
        r_mean, r_std = np.mean(r), np.std(r) + 1e-8
        l_mean, l_std = np.mean(l), np.std(l) + 1e-8
        fit_stats = (r_mean, r_std, l_mean, l_std)
    else:
        r_mean, r_std, l_mean, l_std = fit_stats
        
    r = (r - r_mean) / r_std
    l = (l - l_mean) / l_std
    
    # We will use agent 0's perspective for simplicity
    x_list = []
    for i in range(N):
        sample_features = []
        for rd in range(7):
            # Landscape for agent 0
            land = l[i, rd, 0, :, :].flatten() # 121
            
            # Sequence for agent 0
            priv_c = c[i, rd, :, 0, :] # (15, 2)
            priv_r = r[i, rd, :, 0].reshape(15, 1) # (15, 1)
            
            soc_c = c[i, rd, :, 1:, :].reshape(15, 6) # 3 agents * 2
            soc_r = r[i, rd, :, 1:].reshape(15, 3) # 3 agents * 1
            
            seq = np.concatenate([priv_c, priv_r, soc_c, soc_r], axis=1).flatten() # 15 * 12 = 180
            
            round_features = np.concatenate([land, seq]) # 301
            sample_features.append(round_features)
            
        sample_features = np.concatenate(sample_features) # 7 * 301 = 2107
        x_list.append(sample_features)
        
    return data['theta'], torch.tensor(np.array(x_list), dtype=torch.float32), fit_stats

def train_npe(theta, x, prior):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training NPE with Transformer CNN on {device}...")
    embedding_net = TransformerCNNEmbedding(output_dim=21).to(device)
    # Disable z_score_x to prevent sbi from destroying spatial structures via independent scaling
    neural_posterior = posterior_nn(model="nsf", embedding_net=embedding_net, z_score_x='none')
    inference = SNPE(prior=prior, density_estimator=neural_posterior, device=device)
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
    out_dir = os.path.join(args.out_dir, "pipeline_cnn")
    os.makedirs(out_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lb = [0.1, 0.01, 0.005, 0.0]
    ub = [5.0, 2.0,  0.1,   1.0]
    prior = BoxUniform(low=torch.tensor(lb, device=device), high=torch.tensor(ub, device=device), device=device)
    param_names = [r"$\lambda$", r"$\beta$", r"$\tau$", r"$\alpha$"]
    
    train_data = torch.load(os.path.join(data_dir, "train_data.pt"))
    theta_train, x_train, fit_stats = prepare_cnn_data(train_data)
    
    test_data = torch.load(os.path.join(data_dir, "test_data.pt"))
    theta_test, x_test, _ = prepare_cnn_data(test_data, fit_stats)
    
    # Train
    posterior = train_npe(theta_train, x_train, prior)
    
    # Save posterior
    torch.save(posterior, os.path.join(out_dir, "posterior_cnn.pt"))
    posterior = torch.load(os.path.join(out_dir, "posterior_cnn.pt"), map_location=device)
    
    theta_test = theta_test.to(device)
    x_test = x_test.to(device)
    
    # Evaluate
    df_results = evaluate_recovery(posterior, theta_test, x_test)
    df_results.to_csv(os.path.join(out_dir, "recovery_cnn.csv"), index=False)
    
    # Plot
    plot_recovery(df_results, param_names, os.path.join(out_dir, "plot_recovery_cnn.png"))
    
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
    plt.savefig(os.path.join(out_dir, "tarp_cnn.png"))
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
        plt.savefig(os.path.join(out_dir, "sbc_cnn.png"))
        plt.close()
    except Exception as e:
        import traceback
        print("SBC diagnostics failed or not available:")
        traceback.print_exc()
    
    print("Done Pipeline CNN!")
