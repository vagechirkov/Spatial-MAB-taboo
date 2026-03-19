#!/bin/bash

# Single-configuration Slurm runner.
# Calls the shared Python runner directly with one (beta, tau, length-scale) combination.
#
# Example:
# sbatch --job-name=mab-dyadic \
#   --export=ALL,PROJECT_DIR=/home/${USER}/Spatial-MAB-taboo,OUTPUT_DIR=/scratch/${USER}/dyadic500,BETA=0.53,TAU=0.02,LENGTH_SCALE_MULTIPLIER=0.1 \
#   run_single_job.sh

#SBATCH --job-name=mab-dyadic-hetero
#SBATCH --output=/scratch/%u/logs/%x_%j.log
#SBATCH --error=/scratch/%u/logs/%x_%j.err
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=8G
#SBATCH --time=2:00:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -d "${PROJECT_DIR}/abm" ]]; then
  echo "PROJECT_DIR must point to repository root containing ./abm" >&2
  echo "Current PROJECT_DIR: ${PROJECT_DIR}" >&2
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

cd "${PROJECT_DIR}"

# Activate virtual environment from project root unless VENV_DIR is overridden.
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv}"
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  source "${VENV_DIR}/bin/activate"
fi

# Always prepend project root so `python -m <module>` is importable.
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# Prevent thread oversubscription when using multiprocessing.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -x "${VENV_DIR}/bin/python" ]]; then
  PYTHON_BIN="${VENV_DIR}/bin/python"
else
  echo "PYTHON_BIN is not set and no project venv python found at ${VENV_DIR}/bin/python" >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

RUNNER_MODULE="${RUNNER_MODULE:-abm.run_slurm_jobs}"

GRID_SIZE="${GRID_SIZE:-33}"
LAMBDA_TRUE="${LAMBDA_TRUE:-4.5}"
TARGET_CORRELATION="${TARGET_CORRELATION:-1.0}"
N_AGENTS="${N_AGENTS:-2}"
N_RUNS="${N_RUNS:-100}"
MAX_STEPS="${MAX_STEPS:-500}"
ALPHA="${ALPHA:-0.12}"
ALPHA_BY_AGENT="${ALPHA_BY_AGENT:-}"

BETA="${BETA:-0.53}"
BETA_STEP="${BETA_STEP:-1.0}"
BETA_BY_AGENT="${BETA_BY_AGENT:-}"

TAU_OFFSET="${TAU_OFFSET:-0.0}"
TAU="${TAU:-0.02}"
TAU_STEP="${TAU_STEP:-1.0}"
TAU_BY_AGENT="${TAU_BY_AGENT:-}"

LENGTH_SCALE_MULTIPLIER="${LENGTH_SCALE_MULTIPLIER:-0.1}"
LENGTH_SCALE_MULTIPLIERS_BY_AGENT="${LENGTH_SCALE_MULTIPLIERS_BY_AGENT:-}"

COLLECT_EVERY="${COLLECT_EVERY:-1}"
NUMBER_PROCESSES="${NUMBER_PROCESSES:-${SLURM_CPUS_PER_TASK:-}}"

JOB_NAME="${JOB_NAME:-${SLURM_JOB_NAME:-mab-dyadic-single}}"
OUTPUT_STEM_DEFAULT="$(printf '%s' "${JOB_NAME}" | tr '[:space:]/' '__' | tr -cd 'A-Za-z0-9._-')"
if [[ -z "${OUTPUT_STEM_DEFAULT}" ]]; then
  OUTPUT_STEM_DEFAULT="parameter_sweep"
fi
OUTPUT_STEM="${OUTPUT_STEM:-${OUTPUT_STEM_DEFAULT}}"
OUTPUT_CSV="${OUTPUT_DIR}/${OUTPUT_STEM}.csv"

APPEND="${APPEND:-0}"
DISPLAY_PROGRESS="${DISPLAY_PROGRESS:-1}"
LOG_EVERY="${LOG_EVERY:-5}"

mkdir -p "${OUTPUT_DIR}"

cmd=(
  "${PYTHON_BIN}" -m "${RUNNER_MODULE}"
  --grid-size "${GRID_SIZE}"
  --lambda-true "${LAMBDA_TRUE}"
  --target-correlation "${TARGET_CORRELATION}"
  --n-agents "${N_AGENTS}"
  --n-runs "${N_RUNS}"
  --max-steps "${MAX_STEPS}"
  --alpha "${ALPHA}"
  --beta-start "${BETA}"
  --beta-stop "${BETA}"
  --beta-step "${BETA_STEP}"
  --tau-offset "${TAU_OFFSET}"
  --tau-start "${TAU}"
  --tau-stop "${TAU}"
  --tau-step "${TAU_STEP}"
  --length-scale-multipliers "${LENGTH_SCALE_MULTIPLIER}"
  --output-csv "${OUTPUT_CSV}"
  --num-jobs "1"
  --job-index "0"
  --data-collection-period "${COLLECT_EVERY}"
  --log-every "${LOG_EVERY}"
)

if [[ -n "${NUMBER_PROCESSES}" ]]; then
  cmd+=(--number-processes "${NUMBER_PROCESSES}")
fi

if [[ -n "${ALPHA_BY_AGENT}" ]]; then
  cmd+=(--alpha-by-agent "${ALPHA_BY_AGENT}")
fi

if [[ -n "${BETA_BY_AGENT}" ]]; then
  cmd+=(--beta-by-agent "${BETA_BY_AGENT}")
fi

if [[ -n "${TAU_BY_AGENT}" ]]; then
  cmd+=(--tau-by-agent "${TAU_BY_AGENT}")
fi

if [[ -n "${LENGTH_SCALE_MULTIPLIERS_BY_AGENT}" ]]; then
  cmd+=(--length-scale-multipliers-by-agent "${LENGTH_SCALE_MULTIPLIERS_BY_AGENT}")
fi

if [[ "${APPEND}" -eq 1 ]]; then
  cmd+=(--append)
fi

if [[ "${DISPLAY_PROGRESS}" -eq 0 ]]; then
  cmd+=(--no-display-progress)
fi

echo "Running single-combination job via ${RUNNER_MODULE}"
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "VENV_DIR=${VENV_DIR}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "RUNNER_MODULE=${RUNNER_MODULE}"
echo "JOB_NAME=${JOB_NAME}"
echo "OUTPUT_STEM=${OUTPUT_STEM}"
echo "BETA=${BETA}, TAU=${TAU}, LENGTH_SCALE_MULTIPLIER=${LENGTH_SCALE_MULTIPLIER}"
if [[ -n "${BETA_BY_AGENT}" ]]; then
  echo "BETA_BY_AGENT=${BETA_BY_AGENT}"
fi
if [[ -n "${TAU_BY_AGENT}" ]]; then
  echo "TAU_BY_AGENT=${TAU_BY_AGENT}"
fi
if [[ -n "${ALPHA_BY_AGENT}" ]]; then
  echo "ALPHA_BY_AGENT=${ALPHA_BY_AGENT}"
fi
if [[ -n "${LENGTH_SCALE_MULTIPLIERS_BY_AGENT}" ]]; then
  echo "LENGTH_SCALE_MULTIPLIERS_BY_AGENT=${LENGTH_SCALE_MULTIPLIERS_BY_AGENT}"
fi
echo "NUMBER_PROCESSES=${NUMBER_PROCESSES:-mesa-default}"
echo "Output CSV=${OUTPUT_CSV}"

if ! "${PYTHON_BIN}" -c "import importlib; importlib.import_module('${RUNNER_MODULE}')" >/dev/null 2>&1; then
  echo "Python preflight failed: cannot import runner module '${RUNNER_MODULE}' with PYTHON_BIN=${PYTHON_BIN}" >&2
  echo "PYTHONPATH=${PYTHONPATH}" >&2
  exit 1
fi

"${cmd[@]}"
