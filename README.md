# Spatial-MAB-taboo


## Setup

This project uses `uv` for environment management. To set up the virtual environment and install dependencies, run:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .[dev]
```

### Jupyter Kernel

To use the virtual environment in a Jupyter notebook, register the kernel:

```bash
python -m ipykernel install --user --name spatial-mab --display-name "Python (spatial-mab)"
```

Then, when opening `illustration.ipynb`, select the **Python (spatial-mab)** kernel.

### Starting Jupyter Lab

To start the Jupyter Lab server:

```bash
uv run jupyter lab
```

Start jupyter lab on remote server:

```bash
source .venv/bin/activate && jupyter lab --no-browser --ip=0.0.0.0 --port=8888
```

## Usage

To run the simulation:

```bash
uv run python3 abm/model.py
```

## Heterogeneous Agent Parameters

`SocialGPModel` now supports heterogeneous agent hyperparameters for:

- `length_scale`
- `observation_noise`
- `beta`
- `tau`
- `alpha`
- `sigma_social`

Each can be passed as:

- a scalar: shared across all agents
- a length-1 sequence: broadcast to all agents
- a length-`n` sequence: one value per agent

Example (direct model construction):

```python
from abm.model import SocialGPModel

model = SocialGPModel(
    n=2,
    social_information_mode="social_generalization",
    beta=0.33,
    tau=0.03,
    sigma_social=12.55,
)
```

`social_information_mode` is model-wide and accepts `"value_shaping"` (the
backward-compatible default) or `"social_generalization"`. In Social
Generalization, `sigma_social` is added directly to the GP observation-noise
variance for social rows; it is not squared.

### mesa.batch_run Note

Mesa treats iterables as sweep dimensions. To pass a per-agent vector as one
fixed parameter value, wrap it with `as_batch_fixed`:

```python
import mesa
from abm.model import SocialGPModel, as_batch_fixed

params = {
    "n": 2,
    "social_information_mode": "social_generalization",
    "beta": as_batch_fixed([0.07, 0.10]),
    "tau": as_batch_fixed([0.09, 0.07]),
    "alpha": as_batch_fixed([0.12, 0.18]),
    "sigma_social": as_batch_fixed([12.55, 12.55]),
}

results = mesa.batch_run(
    SocialGPModel,
    parameters=params,
    max_steps=300,
    rng=[None] * 10,
)
```

## Batch Parameter Sweep (CSV Output)

Run the notebook-style parameter sweep from the command line:

```bash
./run_parameter_sweep.sh
```

All sweep settings are grouped at the top of `run_parameter_sweep.sh`:

- environment and runtime (`PYTHON_BIN`, `NUMBER_PROCESSES`)
- model constants (`GRID_SIZE`, `LAMBDA_TRUE`, `TARGET_CORRELATION`, `N_RUNS`, `MAX_STEPS`, `SOCIAL_INFORMATION_MODE`, `SIGMA_SOCIAL`)
- sweep ranges (`BETA_*`, `TAU_*`, `LENGTH_SCALE_MULTIPLIERS`)
- output and sharding (`NUM_JOBS`, `JOB_INDEX`, `OUTPUT_STEM`)

The script writes CSV incrementally per parameter combination to keep memory bounded.

### Length-Scale Sweep Modes

`run_parameter_sweep.sh` supports three mutually exclusive length-scale sweep modes.
Priority is first-match:

1. `LENGTH_SCALE_VALUES` (absolute CSV list)
2. `LENGTH_SCALE_LOG_START` + `LENGTH_SCALE_LOG_STOP` + `LENGTH_SCALE_LOG_NUM` (absolute log-space)
3. `LENGTH_SCALE_MULTIPLIERS` (relative to `LAMBDA_TRUE`, fallback)

#### 1) Absolute explicit values

```bash
LENGTH_SCALE_VALUES="0.1,0.2,0.5,1.0,2.0" ./run_parameter_sweep.sh
```

#### 2) Absolute logarithmic sweep

```bash
LENGTH_SCALE_LOG_START=0.1 \
LENGTH_SCALE_LOG_STOP=10 \
LENGTH_SCALE_LOG_NUM=9 \
LENGTH_SCALE_LOG_BASE=10 \
./run_parameter_sweep.sh
```

This generates values equivalent to `numpy.logspace(log_base(start), log_base(stop), num)`.

Slurm example:

```bash
sbatch --export=ALL,LENGTH_SCALE_LOG_START=0.1,LENGTH_SCALE_LOG_STOP=10,LENGTH_SCALE_LOG_NUM=9 run_parameter_sweep.sh
```

#### 3) Multiplier mode (legacy/default)

```bash
LAMBDA_TRUE=4.5 LENGTH_SCALE_MULTIPLIERS="0.1,0.5,1.0,2.0" ./run_parameter_sweep.sh
```

This evaluates absolute `length_scale = LAMBDA_TRUE * multiplier`.

Both shell launchers call the same Python runner module:

```bash
python -m abm.run_slurm_jobs
```

## Single Configuration (CSV Output)

Run one `(beta, tau, length-scale multiplier)` combination from the command line:

```bash
./run_single_job.sh
```

Override the single-point values via environment variables:

```bash
BETA=0.53 TAU=0.02 LENGTH_SCALE_MULTIPLIER=0.1 ./run_single_job.sh
```

Run the Witt Social Generalization configuration with:

```bash
SOCIAL_INFORMATION_MODE=social_generalization \
SIGMA_SOCIAL=12.55 \
BETA=0.33 \
TAU=0.03 \
LENGTH_SCALE_MULTIPLIER=0.555 \
./run_single_job.sh
```

Use `SIGMA_SOCIAL_BY_AGENT="12.55,12.55"` for heterogeneous fixed values;
the Python runner also exposes `--sigma-social-by-agent`.

## Scaling To Many Combinations / Nodes

Best practice is to shard combinations across jobs and write one CSV shard per job.

Single machine with 4 shards:

```bash
for i in 0 1 2 3; do
    NUM_JOBS=4 JOB_INDEX=$i ./run_parameter_sweep.sh &
done
wait
```

Slurm array (example):

```bash
sbatch --array=0-15 run_sweep.slurm
```

`run_parameter_sweep.sh` already reads `SLURM_ARRAY_TASK_ID` into `JOB_INDEX` by default.

After shards complete, merge them:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

paths = sorted(Path(".").glob("parameter_sweep_corr_dog_job*.csv"))
if not paths:
    raise SystemExit("No shard CSV files found")

merged = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
merged.to_csv("parameter_sweep_corr_dog.csv", index=False)
print(f"Merged {len(paths)} files into parameter_sweep_corr_dog.csv")
PY
```


## Train RL agent

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
python rl/train_agent.py
# CUDA_VISIBLE_DEVICES=1 python rl/train_agent.py
# watch -n 1 nvidia-smi
NVIDIA_TF32_OVERRIDE=1 CUDA_VISIBLE_DEVICES=0 python rl/train_agent.py --total_timesteps 500000000
NVIDIA_TF32_OVERRIDE=1 CUDA_VISIBLE_DEVICES=2 python rl/train_agent.py --length_scale 4.0 --grid_size 33 --budgets 25 50 100 200 --dog_max_range 1.2 2.2 --total_timesteps 500000000
NVIDIA_TF32_OVERRIDE=1 CUDA_VISIBLE_DEVICES=0 python rl/train_agent.py --length_scale 4.0 --grid_size 33 --budgets 15 30 45 60 --dog_max_range 1.2 1.8 --total_timesteps 500000000
NVIDIA_TF32_OVERRIDE=1 CUDA_VISIBLE_DEVICES=1 python rl/train_agent.py --length_scale 2.5 --grid_size 22 --total_timesteps 500000000

NVIDIA_TF32_OVERRIDE=1 CUDA_VISIBLE_DEVICES=2 python rl/train_agent.py --length_scale 4.0 --grid_size 33 --budgets 15 30 45 60 --dog_max_range 1.2 1.8 --total_timesteps 500000000 --turbulence_scale 0.0 --valley_gradient_mag 0.0 --noise_std 0.2 --non_dog_fraction 0.2 --hide_dog_max

NVIDIA_TF32_OVERRIDE=1 CUDA_VISIBLE_DEVICES=1 python rl/train_agent.py --length_scale 4.0 --grid_size 33 --budgets 15 30 45 60 --dog_max_range 1.0 2.0 --total_timesteps 500000000 --turbulence_scale 0.0 --valley_gradient_mag 0.0 --noise_std 0.01 --non_dog_fraction 0.0 --hide_dog_max

NVIDIA_TF32_OVERRIDE=1 CUDA_VISIBLE_DEVICES=1 python rl/train_agent.py --length_scale 4.0 --grid_size 33 --budgets 25 50 75 100 --dog_max_range 0.5 2.0 --total_timesteps 1000000000 --noise_std 0.05

NVIDIA_TF32_OVERRIDE=1 CUDA_VISIBLE_DEVICES=2 python rl/train_agent.py --length_scale 4.0 --grid_size 33 --budgets 25 50 75 100 --dog_max_range 0.5 2.0 --total_timesteps 1000000000 --hide_dog_max --noise_std 0.05

NVIDIA_TF32_OVERRIDE=1 CUDA_VISIBLE_DEVICES=0 python rl/train_agent.py --length_scale 4.0 --grid_size 33 --budgets 25 50 75 100 --dog_max_range 0.5 2.0 --total_timesteps 1000000000 --hide_time_budget --noise_std 0.05

NVIDIA_TF32_OVERRIDE=1 CUDA_VISIBLE_DEVICES=2 python rl/train_agent.py --length_scale 4.0 --grid_size 33 --budgets 15 30 45 60 --dog_max_range 1.0 2.0 --total_timesteps 500000000 --hide_dog_max --hide_time_budget --noise_std 0.05

NVIDIA_TF32_OVERRIDE=1 CUDA_VISIBLE_DEVICES=2 python rl/train_agent.py --length_scale 4.0 --grid_size 33 --budgets 15 30 45 60 --dog_max_range 1.0 2.0 --total_timesteps 500000000 --hide_dog_max --hide_time_budget --noise_std 0.2 --memory_size 3
```


### Visualize DoG advatage

```bash
python3 abm/visualization_optimal_dog.py
python3 rl/utils.py
```


## SBI Pipelines

```bash
python -m sbi_pipelines.simulate
python -m sbi_pipelines.pipeline_cnn
python -m sbi_pipelines.pipeline_summary
python -m sbi_pipelines.pipeline_mle
python -m sbi_pipelines.evaluate_all
python -m sbi_pipelines.run_all --data_dir sim_5k_1k
```
