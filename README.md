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

Each can be passed as:

- a scalar: shared across all agents
- a length-1 sequence: broadcast to all agents
- a length-`n` sequence: one value per agent

Example (direct model construction):

```python
from abm.model import SocialGPModel

model = SocialGPModel(
    n=2,
    beta=[0.07, 0.10],
    tau=[0.09, 0.07],
    alpha=[0.12, 0.18],
)
```

### mesa.batch_run Note

Mesa treats iterables as sweep dimensions. To pass a per-agent vector as one
fixed parameter value, wrap it with `as_batch_fixed`:

```python
import mesa
from abm.model import SocialGPModel, as_batch_fixed

params = {
    "n": 2,
    "beta": as_batch_fixed([0.07, 0.10]),
    "tau": as_batch_fixed([0.09, 0.07]),
    "alpha": as_batch_fixed([0.12, 0.18]),
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
- model constants (`GRID_SIZE`, `LAMBDA_TRUE`, `TARGET_CORRELATION`, `N_RUNS`, `MAX_STEPS`)
- sweep ranges (`BETA_*`, `TAU_*`, `LENGTH_SCALE_MULTIPLIERS`)
- output and sharding (`NUM_JOBS`, `JOB_INDEX`, `OUTPUT_STEM`)

The script writes CSV incrementally per parameter combination to keep memory bounded.

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
CUDA_VISIBLE_DEVICES=0 python rl/train_agent.py --environment simple_gp --budgets 50 --length_scale 4.0
CUDA_VISIBLE_DEVICES=1 python rl/train_agent.py --environment correlated_dog --budgets 15 --length_scale 4.0
CUDA_VISIBLE_DEVICES=1 python rl/train_agent.py --environment correlated_dog --budgets 100 --length_scale 4.0 --total_timesteps 1000000000
CUDA_VISIBLE_DEVICES=0 python rl/train_agent.py --environment correlated_dog --budgets 300 --length_scale 4.0 --total_timesteps 1000000000
CUDA_VISIBLE_DEVICES=2 python rl/train_agent.py --environment correlated_dog --dog_max 2.0 --budgets 100 --length_scale 4.0 --total_timesteps 1000000000
CUDA_VISIBLE_DEVICES=2 python rl/train_agent.py --environment correlated_dog --dog_max 10.0 --budgets 50 --length_scale 4.0 --total_timesteps 1000000000
CUDA_VISIBLE_DEVICES=2 python rl/train_agent.py --environment correlated_dog --budgets 500 --length_scale 4.0 --total_timesteps 1000000000


CUDA_VISIBLE_DEVICES=0 python rl/train_agent.py --environment correlated_dog --budgets 300 --length_scale 4.0 --total_timesteps 300000000
CUDA_VISIBLE_DEVICES=2 python rl/train_agent.py --environment correlated_dog --budgets 600 --length_scale 4.0 --total_timesteps 300000000
```


### Visualize DoG advatage

```bash
python3 abm/visualization_optimal_dog.py
```