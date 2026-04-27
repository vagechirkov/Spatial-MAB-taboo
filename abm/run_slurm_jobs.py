from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import mesa
import numpy as np
import pandas as pd

from abm.model import SocialGPModel, as_batch_fixed


def parse_csv_floats(raw_values: str) -> list[float]:
    values: list[float] = []
    for token in raw_values.split(","):
        token = token.strip()
        if token:
            values.append(float(token))

    if not values:
        raise ValueError("Expected at least one numeric value")

    return values


def parse_optional_csv_floats(raw_values: str | None) -> list[float] | None:
    if raw_values is None:
        return None

    raw_values = raw_values.strip()
    if raw_values == "":
        return None

    return parse_csv_floats(raw_values)


def normalize_agent_vector(
    values: list[float] | None,
    n_agents: int,
    parameter_name: str,
) -> list[float] | None:
    if values is None:
        return None

    if n_agents <= 0:
        raise ValueError("n-agents must be a positive integer")

    if len(values) == 1:
        return values * n_agents

    if len(values) != n_agents:
        raise ValueError(
            f"{parameter_name} must contain either 1 value or n-agents ({n_agents}) values. "
            f"Got {len(values)} values."
        )

    return values


def inclusive_float_range(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("step must be positive")
    if stop < start:
        raise ValueError("stop must be greater than or equal to start")

    values = np.arange(start, stop + 0.5 * step, step, dtype=float)
    return np.round(values, 12).tolist()


def build_log_space(start: float, stop: float, num: int, base: float) -> list[float]:
    if start <= 0 or stop <= 0:
        raise ValueError("length-scale log sweep bounds must be positive")
    if num <= 0:
        raise ValueError("length-scale log sweep num must be >= 1")
    if base <= 0 or base == 1.0:
        raise ValueError("length-scale log sweep base must be positive and not equal to 1")

    log_start = np.log(start) / np.log(base)
    log_stop = np.log(stop) / np.log(base)
    values = np.logspace(log_start, log_stop, num=num, base=base, dtype=float)
    return np.round(values, 12).tolist()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run SocialGPModel jobs for Slurm or local execution and stream results to CSV. "
            "Use ranges for sweeps or set start==stop for single-configuration runs."
        )
    )

    parser.add_argument("--grid-size", type=int, default=33)
    parser.add_argument("--lambda-true", type=float, default=4.5)
    parser.add_argument("--target-correlation", type=float, default=0.9)
    parser.add_argument("--n-agents", type=int, default=1)
    parser.add_argument("--n-runs", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument(
        "--alpha-by-agent",
        type=str,
        default=None,
        help=(
            "Optional comma-separated per-agent alpha values. "
            "Length must be 1 or n-agents. "
            "When set, overrides --alpha and is passed as one fixed mesa.batch_run value."
        ),
    )

    parser.add_argument("--beta-start", type=float, default=0.0)
    parser.add_argument("--beta-stop", type=float, default=0.5)
    parser.add_argument("--beta-step", type=float, default=0.025)
    parser.add_argument(
        "--beta-by-agent",
        type=str,
        default=None,
        help=(
            "Optional comma-separated per-agent beta values. "
            "Length must be 1 or n-agents. "
            "When set, overrides beta sweep range and is passed as one fixed mesa.batch_run value."
        ),
    )

    parser.add_argument("--tau-offset", type=float, default=0.005)
    parser.add_argument("--tau-start", type=float, default=0.01)
    parser.add_argument("--tau-stop", type=float, default=0.10)
    parser.add_argument("--tau-step", type=float, default=0.01)
    parser.add_argument(
        "--tau-by-agent",
        type=str,
        default=None,
        help=(
            "Optional comma-separated per-agent tau values. "
            "Length must be 1 or n-agents. "
            "When set, overrides tau sweep range and is passed as one fixed mesa.batch_run value."
        ),
    )

    parser.add_argument(
        "--length-scale-multipliers",
        type=str,
        default="0.5,1.0,2.0",
        help="Comma-separated multipliers applied to lambda_true for agent length_scale.",
    )
    parser.add_argument(
        "--length-scale-values",
        type=str,
        default=None,
        help=(
            "Optional comma-separated absolute length_scale values. "
            "When set, overrides --length-scale-multipliers."
        ),
    )
    parser.add_argument(
        "--length-scale-log-start",
        type=float,
        default=None,
        help=(
            "Optional absolute length_scale lower bound for logarithmic sweep. "
            "Requires --length-scale-log-stop and --length-scale-log-num. "
            "When set (with companions), overrides --length-scale-multipliers."
        ),
    )
    parser.add_argument(
        "--length-scale-log-stop",
        type=float,
        default=None,
        help="Optional absolute length_scale upper bound for logarithmic sweep.",
    )
    parser.add_argument(
        "--length-scale-log-num",
        type=int,
        default=None,
        help="Optional number of logarithmically spaced absolute length_scale values.",
    )
    parser.add_argument(
        "--length-scale-log-base",
        type=float,
        default=10.0,
        help="Logarithm base for --length-scale-log-* sweep parameters.",
    )
    parser.add_argument(
        "--length-scale-multipliers-by-agent",
        type=str,
        default=None,
        help=(
            "Optional comma-separated per-agent length-scale multipliers. "
            "Length must be 1 or n-agents. "
            "When set, overrides --length-scale-multipliers sweep and is passed as one fixed mesa.batch_run value."
        ),
    )
    parser.add_argument(
        "--local-global-max-ratio", type=float, default=None,
        help=(
            "Optional local/global max ratio for MH GP reward landscape. "
            "When set, overrides default in reward_params and is passed as one fixed mesa.batch_run value."
        ),
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
        # If job_index is out of range, wrap around using modulo.
        args.job_index = args.job_index % args.num_jobs

    beta_by_agent = normalize_agent_vector(
        parse_optional_csv_floats(args.beta_by_agent),
        args.n_agents,
        "beta-by-agent",
    )
    tau_by_agent = normalize_agent_vector(
        parse_optional_csv_floats(args.tau_by_agent),
        args.n_agents,
        "tau-by-agent",
    )
    alpha_by_agent = normalize_agent_vector(
        parse_optional_csv_floats(args.alpha_by_agent),
        args.n_agents,
        "alpha-by-agent",
    )
    length_scale_multipliers_by_agent = normalize_agent_vector(
        parse_optional_csv_floats(args.length_scale_multipliers_by_agent),
        args.n_agents,
        "length-scale-multipliers-by-agent",
    )

    has_log_start = args.length_scale_log_start is not None
    has_log_stop = args.length_scale_log_stop is not None
    has_log_num = args.length_scale_log_num is not None
    if any([has_log_start, has_log_stop, has_log_num]) and not all(
        [has_log_start, has_log_stop, has_log_num]
    ):
        raise ValueError(
            "length-scale logarithmic sweep requires all of: "
            "--length-scale-log-start, --length-scale-log-stop, --length-scale-log-num"
        )

    if beta_by_agent is None:
        beta_values = inclusive_float_range(args.beta_start, args.beta_stop, args.beta_step)
    else:
        beta_values = [as_batch_fixed(beta_by_agent)]

    if tau_by_agent is None:
        tau_base_values = inclusive_float_range(args.tau_start, args.tau_stop, args.tau_step)
        tau_values = [args.tau_offset + tau_value for tau_value in tau_base_values]
    else:
        tau_values = [as_batch_fixed([args.tau_offset + tau_value for tau_value in tau_by_agent])]

    if length_scale_multipliers_by_agent is None:
        if args.length_scale_values is not None:
            length_scale_values = parse_csv_floats(args.length_scale_values)
        elif has_log_start:
            length_scale_values = build_log_space(
                start=args.length_scale_log_start,
                stop=args.length_scale_log_stop,
                num=args.length_scale_log_num,
                base=args.length_scale_log_base,
            )
        else:
            length_scale_multipliers = parse_csv_floats(args.length_scale_multipliers)
            length_scale_values = [args.lambda_true * multiplier for multiplier in length_scale_multipliers]
    else:
        length_scale_values = [
            as_batch_fixed(
                [args.lambda_true * multiplier for multiplier in length_scale_multipliers_by_agent]
            )
        ]

    if alpha_by_agent is None:
        alpha_value = args.alpha
    else:
        alpha_value = as_batch_fixed(alpha_by_agent)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.append:
        output_path.unlink()

    write_header = not output_path.exists()

    reward_params = {
        "length_scale": args.lambda_true,
        "target_correlation": args.target_correlation,
    }

    if args.local_global_max_ratio is not None:
        reward_params["local_global_max_ratio"] = args.local_global_max_ratio

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
            "alpha": alpha_value,
            "reward_env_type": "mexican_hat_gp",
            "reward_env_params": [reward_params],
            "collect_agent_reporters": True,
            "model_reporters_to_collect": [["mean_cumulative_reward"]],
            "agent_reporters_to_collect": [[
                "distance_to_global_peak",
                "distance_to_local_peak",
                "reward",
                "global_max",
                "local_max",
                "no_max",
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