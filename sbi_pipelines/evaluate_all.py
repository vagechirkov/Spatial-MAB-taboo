import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def compare_recoveries(res_dir):
    paths = {
        'Summary Stats NPE': os.path.join(res_dir, "pipeline_summary", "recovery_summary.csv"),
        'Recurrent CNN NPE': os.path.join(res_dir, "pipeline_cnn", "recovery_cnn.csv"),
        'MLE': os.path.join(res_dir, "pipeline_mle", "recovery_mle.csv")
    }
    
    out_dir = os.path.join(res_dir, "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    
    dfs = []
    for name, path in paths.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['Method'] = name
            dfs.append(df)
            
    if not dfs:
        print("No recovery data found.")
        return
        
    df_all = pd.concat(dfs, ignore_index=True)
    
    # Calculate NRMSE
    # Prior bounds: lb = [0.1, 0.01, 0.005, 0.0], ub = [5.0, 2.0, 0.1, 1.0]
    param_ranges = {
        0: 4.9,
        1: 1.99,
        2: 0.095,
        3: 1.0
    }
    
    df_all['range'] = df_all['parameter'].map(param_ranges)
    df_all['normalized_error'] = (df_all['mean_recovered'] - df_all['true_value']) / df_all['range']
    df_all['squared_n_error'] = df_all['normalized_error'] ** 2
    
    # Map parameter names
    param_names_list = [r"$\lambda$", r"$\beta$", r"$\tau$", r"$\alpha$"]
    param_names = {0: param_names_list[0], 1: param_names_list[1], 2: param_names_list[2], 3: param_names_list[3]}
    df_all['parameter_name'] = df_all['parameter'].map(param_names)
    
    method_order = ['MLE', 'Summary Stats NPE', 'Recurrent CNN NPE']
    
    # Plot NRMSE comparison
    rmse_df = df_all.groupby(['Method', 'parameter_name'])['squared_n_error'].mean().reset_index()
    rmse_df['NRMSE'] = rmse_df['squared_n_error'] ** 0.5
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=rmse_df, x='parameter_name', y='NRMSE', hue='Method', order=param_names_list, hue_order=method_order)
    plt.title("Parameter Recovery NRMSE Comparison (Normalized by Prior Range)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparison_nrmse.png"))
    plt.close()

    # Plot Scatter Comparison
    g = sns.FacetGrid(df_all, col="parameter_name", hue="Method", col_wrap=4, height=4, sharex=False, sharey=False, col_order=param_names_list, hue_order=method_order)
    g.map_dataframe(
        lambda data, color, **kwargs: plt.scatter(
            data['true_value'],
            data['mean_recovered'],
            alpha=0.5, color=color, label=data['Method'].iloc[0] if len(data)>0 else ''
        )
    )
    for ax in g.axes.flat:
        xlim = ax.get_xlim()
        ax.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], 'k--', alpha=0.6)
        ax.set_xlabel("True Parameter")
        ax.set_ylabel("Recovered Parameter")
        
    g.add_legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparison_scatter.png"))
    plt.close()
    
    print("Comparison plots generated in", out_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_dir", type=str, required=True, help="Path to results directory")
    args = parser.parse_args()
    
    compare_recoveries(args.res_dir)
