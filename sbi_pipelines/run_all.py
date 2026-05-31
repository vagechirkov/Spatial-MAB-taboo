import os
import argparse
import subprocess
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Run all SBI and MLE pipelines")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the simulation data directory")
    parser.add_argument("--skip-mle", action="store_true", help="Skip the MLE pipeline")
    args = parser.parse_args()

    # Create timestamped results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.dirname(os.path.normpath(args.data_dir))
    out_dir = os.path.join(base_dir, f"results_{timestamp}")
    
    print(f"Starting pipelines...")
    print(f"Data Directory: {args.data_dir}")
    print(f"Output Directory: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Summary Stats NPE
    print("\n--- Running Summary Stats NPE Pipeline ---")
    subprocess.run([
        "python", "-m", "sbi_pipelines.pipeline_summary",
        "--data_dir", args.data_dir,
        "--out_dir", out_dir
    ], check=True)
    
    # 2. Recurrent CNN NPE
    print("\n--- Running Recurrent CNN NPE Pipeline ---")
    subprocess.run([
        "python", "-m", "sbi_pipelines.pipeline_cnn",
        "--data_dir", args.data_dir,
        "--out_dir", out_dir
    ], check=True)
    
    # 3. MLE
    if not args.skip_mle:
        print("\n--- Running MLE Pipeline ---")
        subprocess.run([
            "python", "-m", "sbi_pipelines.pipeline_mle",
            "--data_dir", args.data_dir,
            "--out_dir", out_dir
        ], check=True)
    else:
        print("\n--- Skipping MLE Pipeline ---")
    
    # 4. Evaluate All
    print("\n--- Running Evaluation ---")
    subprocess.run([
        "python", "-m", "sbi_pipelines.evaluate_all",
        "--res_dir", out_dir
    ], check=True)
    
    # 5. Plot Pairplots
    print("\n--- Generating Posterior Pairplots ---")
    subprocess.run([
        "python", "-m", "sbi_pipelines.plot_pairplots",
        "--data_dir", args.data_dir,
        "--res_dir", out_dir
    ], check=True)
    
    # 6. PPC Diagnostics
    print("\n--- Running PPC Diagnostics ---")
    subprocess.run([
        "python", "-m", "sbi_pipelines.ppc_diagnostics",
        "--data_dir", args.data_dir,
        "--res_dir", out_dir
    ], check=True)
    
    print(f"\nAll pipelines completed successfully! Results are in {out_dir}")

if __name__ == "__main__":
    main()
