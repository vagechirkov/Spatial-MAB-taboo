from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import mesa
import numpy as np
import pandas as pd

from abm.model import SocialGPModel


def parse_csv_floats(raw_values: str) -> list[float]:
    values = []
    for token in raw_values.split(","):
        token = token.strip()
        if token:
            values.append(float(token))

    if not values:
        raise ValueError("Expected at least one numeric value")

    return values


def inclusive_float_range(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("step must be positive")
    if stop < start:
        raise ValueError("stop must be greater than or equal to start")

    values = np.arange(start, stop + 0.5 * step, step, dtype=float)
    return np.round(values, 12).tolist()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a parameter sweep for SocialGPModel and stream batch results to CSV. "
            "Each parameter combination is run independently to keep memory usage bounded."
        )
    )

    parser.add_argument("--grid-size", type=int, default=33)
    parser.add_argument("--lambda-true", type=float, default=4.5)
    parser.add_argument("--target-correlation", type=float, default=0.9)
    parser.add_argument("--n-agents", type=int, default=1)
    parser.add_argument("--n-runs", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--alpha", type=float, default=0.0)

    parser.add_argument("--beta-start", type=float, default=0.0)
    parser.add_argument("--beta-stop", type=float, default=0.5)
    parser.add_argument("--beta-step", type=float, default=0.025)

    parser.add_argument("--tau-offset", type=float, default=0.005)
    parser.add_argument("--tau-start", type=float, default=0.01)
    parser.add_argument("--tau-stop", type=float, default=0.10)
    parser.add_argument("--tau-step", type=float, default=0.01)

    parser.add_argument(
        "--length-scale-multipliers",
        type=str,
        default="0.5,1.0,2.0",
        help="Comma-separated multipliers applied to lambda_true for agent length_scale.",
    )

    parser.add_argument("--output-csv", type=str, default="parameter_sweep_corr_dog.csv")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--data-collection-period", type=int, default=-1)
    parser.add_argument("--number-processes", type=int, default=None)
    parser.add_argument(
        "--display-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--num-jobs", type=int, default=1)
    parser.add_argument("--job-index", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=5)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.num_jobs <= 0:
        raise ValueError("num_jobs must be >= 1")
    if args.job_index < 0:
        raise ValueError("job_index must be in [0, num_jobs)")
    if args.job_index >= args.num_jobs:
        # If job_index is out of range, wrap around using modulo
        args.job_index = args.job_index % args.num_jobs

    beta_values = inclusive_float_range(args.beta_start, args.beta_stop, args.beta_step)
    tau_base_values = inclusive_float_range(args.tau_start, args.tau_stop, args.tau_step)
    tau_values = [args.tau_offset + tau_value for tau_value in tau_base_values]
    length_scale_multipliers = parse_csv_floats(args.length_scale_multipliers)
    length_scale_values = [args.lambda_true * multiplier for multiplier in length_scale_multipliers]

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.append:
        output_path.unlink()

    write_header = not output_path.exists()

    reward_params = {
        "length_scale": args.lambda_true,
        "target_correlation": args.target_correlation,
    }

    total_combinations = len(beta_values) * len(length_scale_values) * len(tau_values)
    assigned_combinations = sum(
        1
        for combo_index in range(total_combinations)
        if combo_index % args.num_jobs == args.job_index
    )
    processed_combinations = 0
    total_rows_written = 0

    for combo_index, (beta_value, length_scale_value, tau_value) in enumerate(
        product(beta_values, length_scale_values, tau_values)
    ):
        if combo_index % args.num_jobs != args.job_index:
            continue
        run_parameters = {
            "n": args.n_agents,
            "grid_size": args.grid_size,
            "beta": beta_value,
            "length_scale": length_scale_value,
            "tau": tau_value,
            "alpha": args.alpha,
            "reward_env_type": "corr_dog",
            "reward_env_params": [reward_params],
            "collect_agent_reporters": True,
            "model_reporters_to_collect": [["mean_cumulative_reward"]],
            "agent_reporters_to_collect": [[
                "distance_to_global_peak",
                "distance_to_local_peak",
                "reward",
                "global_max",
                "local_max",
                "no_max"
            ]],
        }

        batch_results = mesa.batch_run(
            SocialGPModel,
            parameters=run_parameters,
            max_steps=args.max_steps,
            display_progress=args.display_progress,
            data_collection_period=args.data_collection_period,
            number_processes=args.number_processes,
            rng=[None] * args.n_runs,
        )

        combo_df = pd.DataFrame(batch_results)
        if combo_df.empty:
            continue

        combo_df.to_csv(output_path, mode="a", header=write_header, index=False)
        write_header = False

        processed_combinations += 1
        total_rows_written += len(combo_df)

        if args.log_every > 0 and processed_combinations % args.log_every == 0:
            print(
                "Processed "
                f"{processed_combinations}/{assigned_combinations} assigned combinations "
                f"(job {args.job_index + 1}/{args.num_jobs})."
            )

    print(
        "Done. "
        f"Processed {processed_combinations}/{assigned_combinations} assigned combinations "
        f"out of {total_combinations} total combinations. "
        f"Wrote {total_rows_written} rows to {output_path}."
    )


if __name__ == "__main__":
    main()
