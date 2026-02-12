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