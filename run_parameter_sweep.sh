#!/bin/bash

#SBATCH --job-name=spatial-mab-sweep
#SBATCH --output=/scratch/%u/logs/spatial-mab-sweep_%j.log
#SBATCH --error=/scratch/%u/logs/spatial-mab-sweep_%j.err
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-0
#SBATCH --time=2:00:00

set -euo pipefail

# Resolve project root robustly for Slurm jobs (which may execute script from a spool path).
# Priority: PROJECT_DIR env var -> SLURM_SUBMIT_DIR -> current directory.
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -d "${PROJECT_DIR}" ]]; then
  echo "PROJECT_DIR does not exist: ${PROJECT_DIR}" >&2
  exit 1
fi

# Set default OUTPUT_DIR to /scratch/${USER}/parameter_sweeps if running under Slurm, otherwise current directory.
if [[ -z "${OUTPUT_DIR:-}" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    OUTPUT_DIR="/scratch/${USER}/parameter_sweeps"
  else
    OUTPUT_DIR="${PWD}"
  fi
fi


if [[ ! -d "${PROJECT_DIR}/abm" ]]; then
  echo "PROJECT_DIR must point to repository root containing ./abm" >&2
  echo "Current PROJECT_DIR: ${PROJECT_DIR}" >&2
  echo "Set PROJECT_DIR explicitly when submitting, for example:" >&2
  echo "sbatch --export=ALL,PROJECT_DIR=/home/${USER}/Spatial-MAB-taboo,OUTPUT_DIR=/scratch/${USER}/parameter_sweeps run_parameter_sweep.sh" >&2
  exit 1
fi

cd "${PROJECT_DIR}"

# Activate virtual environment from project root unless VENV_DIR is overridden.
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv}"
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  source "${VENV_DIR}/bin/activate"
fi

# Always prepend project root so `python -m abm.run_parameter_sweep` is importable.
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# Prevent thread oversubscription when using multiprocessing.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# -------------------------------
# Edit sweep parameters here
# -------------------------------
if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -x "${VENV_DIR}/bin/python" ]]; then
  PYTHON_BIN="${VENV_DIR}/bin/python"
else
  echo "PYTHON_BIN is not set and no project venv python found at ${VENV_DIR}/bin/python" >&2
  echo "Set PYTHON_BIN explicitly, for example:" >&2
  echo "sbatch --export=ALL,PYTHON_BIN=/home/${USER}/Spatial-MAB-taboo/.venv/bin/python run_parameter_sweep.sh" >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

GRID_SIZE=33
LAMBDA_TRUE=4.5
TARGET_CORRELATION=0.9
N_AGENTS=1
N_RUNS=50
MAX_STEPS=500
ALPHA=0.0

BETA_START=0.65
BETA_STOP=0.66
BETA_STEP=0.025

TAU_OFFSET=0.0
TAU_START=0.008
TAU_STOP=0.009
TAU_STEP=0.02

LENGTH_SCALE_MULTIPLIERS="0.5" # 1.0

# Parallelism inside mesa.batch_run.
# Default to Slurm's CPU allocation if available.
NUMBER_PROCESSES="${NUMBER_PROCESSES:-${SLURM_CPUS_PER_TASK:-}}"

# Job sharding controls for multi-core/multi-node runs.
# A job processes combinations where combo_index % NUM_JOBS == JOB_INDEX.
NUM_JOBS="${NUM_JOBS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
JOB_INDEX="${JOB_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"

OUTPUT_STEM="parameter_sweep_hole1"
OUTPUT_DIR="${OUTPUT_DIR:-${SLURM_SUBMIT_DIR:-.}}"

if [[ ! "${NUM_JOBS}" =~ ^[0-9]+$ ]] || [[ "${NUM_JOBS}" -lt 1 ]]; then
  echo "NUM_JOBS must be a positive integer, got: ${NUM_JOBS}" >&2
  exit 1
fi

if [[ ! "${JOB_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "JOB_INDEX must be in [0, NUM_JOBS), got JOB_INDEX=${JOB_INDEX}, NUM_JOBS=${NUM_JOBS}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

if [[ "${NUM_JOBS}" -gt 1 ]]; then
  OUTPUT_CSV="${OUTPUT_DIR}/${OUTPUT_STEM}_job${JOB_INDEX}.csv"
else
  OUTPUT_CSV="${OUTPUT_DIR}/${OUTPUT_STEM}.csv"
fi

APPEND=0
DISPLAY_PROGRESS=1
LOG_EVERY=5

cmd=(
  "${PYTHON_BIN}" -m abm.run_parameter_sweep
  --grid-size "${GRID_SIZE}"
  --lambda-true "${LAMBDA_TRUE}"
  --target-correlation "${TARGET_CORRELATION}"
  --n-agents "${N_AGENTS}"
  --n-runs "${N_RUNS}"
  --max-steps "${MAX_STEPS}"
  --alpha "${ALPHA}"
  --beta-start "${BETA_START}"
  --beta-stop "${BETA_STOP}"
  --beta-step "${BETA_STEP}"
  --tau-offset "${TAU_OFFSET}"
  --tau-start "${TAU_START}"
  --tau-stop "${TAU_STOP}"
  --tau-step "${TAU_STEP}"
  --length-scale-multipliers "${LENGTH_SCALE_MULTIPLIERS}"
  --output-csv "${OUTPUT_CSV}"
  --num-jobs "${NUM_JOBS}"
  --job-index "${JOB_INDEX}"
  --log-every "${LOG_EVERY}"
)

if [[ -n "${NUMBER_PROCESSES}" ]]; then
  cmd+=(--number-processes "${NUMBER_PROCESSES}")
fi

if [[ "${APPEND}" -eq 1 ]]; then
  cmd+=(--append)
fi

if [[ "${DISPLAY_PROGRESS}" -eq 0 ]]; then
  cmd+=(--no-display-progress)
fi

echo "Running on host: $(hostname)"
echo "PROJECT_DIR: ${PROJECT_DIR}"
echo "VENV_DIR: ${VENV_DIR}"
echo "PYTHON_BIN: ${PYTHON_BIN}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID:-none}"
echo "NUM_JOBS=${NUM_JOBS}, JOB_INDEX=${JOB_INDEX}"
echo "NUMBER_PROCESSES=${NUMBER_PROCESSES:-mesa-default}"
echo "Output CSV: ${OUTPUT_CSV}"

if ! "${PYTHON_BIN}" -c "import abm" >/dev/null 2>&1; then
  echo "Python preflight failed: cannot import 'abm' with PYTHON_BIN=${PYTHON_BIN}" >&2
  echo "PYTHONPATH=${PYTHONPATH}" >&2
  exit 1
fi

"${cmd[@]}"
