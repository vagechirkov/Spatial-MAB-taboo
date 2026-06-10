import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sbi.analysis import pairplot
from sbi_pipelines.pipeline_summary import load_data as load_data_summary
from sbi_pipelines.pipeline_4d_cnn import prepare_4d_data, Embedding4DCNN
import __main__
__main__.Embedding4DCNN = Embedding4DCNN


def plot_posteriors(data_dir, res_dir):
    print("Generating pairplots for 5 test samples...")
    out_dir = os.path.join(res_dir, "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    
    # Load posteriors
    post_cnn = torch.load(os.path.join(res_dir, "pipeline_4d_cnn", "posterior_4d_cnn.pt"), weights_only=False, map_location="cpu")
    post_sum = torch.load(os.path.join(res_dir, "pipeline_summary", "posterior_summary.pt"), weights_only=False, map_location="cpu")
    
    # Load test data
    test_data = torch.load(os.path.join(data_dir, "test_data.pt"), weights_only=False)
    
    # Get x_test
    train_data = torch.load(os.path.join(data_dir, "train_data.pt"), weights_only=False)
    _, _, fit_stats = prepare_4d_data(train_data)    
    theta_test, x_test_cnn, _ = prepare_4d_data(test_data, fit_stats)
    _, x_test_sum = load_data_summary(os.path.join(data_dir, "test_data.pt"))
    
    # Load MLE results
    mle_df = pd.read_csv(os.path.join(res_dir, "pipeline_mle", "recovery_mle.csv"))
    
    param_names = [r"$\lambda$", r"$\beta$", r"$\tau$", r"$\alpha$"]
    
    # Pick 5 samples
    for i in range(30):
        _theta = theta_test[i]
        _x_cnn = x_test_cnn[i]
        _x_sum = x_test_sum[i]
        
        # MLE estimate for this test sample
        mle_est = np.zeros(4)
        for p_idx in range(4):
            val = mle_df[(mle_df['test_idx'] == i) & (mle_df['parameter'] == p_idx)]['mean_recovered'].values
            if len(val) > 0:
                mle_est[p_idx] = val[0]
            else:
                mle_est[p_idx] = np.nan
        mle_tensor = torch.tensor(mle_est, dtype=torch.float32)
        
        # Sample from posteriors
        samples_cnn = post_cnn.sample((200_000,), x=_x_cnn, show_progress_bars=False)
        samples_sum = post_sum.sample((200_000,), x=_x_sum, show_progress_bars=False)
        
        fig, axes = pairplot(
            samples=[samples_sum, samples_cnn],
            points=[_theta, mle_tensor],
            points_colors=['red', 'black'],
            labels=param_names,
            figsize=(10, 10),
            upper='contour',
            diag='kde'
        )
        plt.legend(
            ["Summary", "4DCNN", "True", "MLE"],
            frameon=False,
            fontsize=8,
            loc="upper right"
        )

        plt.suptitle(f"Posterior Pairplot - Test Sample {i}", fontsize=16)
        plt.subplots_adjust(top=0.92) # Leave space for suptitle
        plt.savefig(os.path.join(out_dir, f"pairplot_sample_{i}.png"))
        plt.close()
        print(f"Saved pairplot for sample {i}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--res_dir", type=str, required=True, help="Path to results directory")
    args = parser.parse_args()
    
    plot_posteriors(args.data_dir, args.res_dir)
