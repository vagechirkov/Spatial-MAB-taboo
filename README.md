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