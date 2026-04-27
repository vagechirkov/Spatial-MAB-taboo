#!/bin/bash

# Use `sbatch --job-name=<name> ... run_parameter_sweep.sh` to override.
# `%x` expands to the effective Slurm job name.
#SBATCH --job-name=lambda-oft-plot
#SBATCH --output=/scratch/%u/logs/%x_%j.log
#SBATCH --error=/scratch/%u/logs/%x_%j.err
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-100
#SBATCH --time=12:00:00

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
  echo "sbatch --job-name=mab-my-run --export=ALL,PROJECT_DIR=/home/${USER}/Spatial-MAB-taboo,OUTPUT_DIR=/scratch/${USER}/parameter_sweeps run_parameter_sweep.sh" >&2
  exit 1
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

RUNNER_MODULE="${RUNNER_MODULE:-abm.run_slurm_jobs}"

GRID_SIZE="${GRID_SIZE:-33}"
LAMBDA_TRUE="${LAMBDA_TRUE:-4.5}"
TARGET_CORRELATION="${TARGET_CORRELATION:-1.0}"
LOCAL_GLOBAL_MAX_RATIO="${LOCAL_GLOBAL_MAX_RATIO:-3.0}"
N_AGENTS="${N_AGENTS:-1}"
N_RUNS="${N_RUNS:-100}"
MAX_STEPS="${MAX_STEPS:-300}"
ALPHA="${ALPHA:-0.0}"

BETA_START="${BETA_START:-0.3}"
BETA_STOP="${BETA_STOP:-0.7}"
BETA_STEP="${BETA_STEP:-0.1}"

TAU_OFFSET="${TAU_OFFSET:-0.0}"
TAU_START="${TAU_START:-0.03}"
TAU_STOP="${TAU_STOP:-0.031}"
TAU_STEP="${TAU_STEP:-0.02}"

# Length-scale sweep modes (first match wins):
# 1) LENGTH_SCALE_VALUES (absolute CSV)
# 2) LENGTH_SCALE_LOG_START/STOP/NUM (absolute logarithmic sweep)
# 3) LENGTH_SCALE_MULTIPLIERS (relative to LAMBDA_TRUE)
LENGTH_SCALE_VALUES="${LENGTH_SCALE_VALUES:-}"
LENGTH_SCALE_LOG_START="${LENGTH_SCALE_LOG_START:-0.5}"
LENGTH_SCALE_LOG_STOP="${LENGTH_SCALE_LOG_STOP:-3.0}"
LENGTH_SCALE_LOG_NUM="${LENGTH_SCALE_LOG_NUM:-10}"
LENGTH_SCALE_LOG_BASE="${LENGTH_SCALE_LOG_BASE:-10}"
LENGTH_SCALE_MULTIPLIERS="${LENGTH_SCALE_MULTIPLIERS:-}" # 1.0

COLLECT_EVERY="${COLLECT_EVERY:-1}"

# Parallelism inside mesa.batch_run.
# Default to Slurm's CPU allocation if available.
NUMBER_PROCESSES="${NUMBER_PROCESSES:-${SLURM_CPUS_PER_TASK:-}}"

# Job sharding controls for multi-core/multi-node runs.
# A job processes combinations where combo_index % NUM_JOBS == JOB_INDEX.
NUM_JOBS="${NUM_JOBS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
JOB_INDEX="${JOB_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"

JOB_NAME="${JOB_NAME:-${SLURM_JOB_NAME:-mab-dyadic-sweep}}"
OUTPUT_STEM_DEFAULT="$(printf '%s' "${JOB_NAME}" | tr '[:space:]/' '__' | tr -cd 'A-Za-z0-9._-')"
if [[ -z "${OUTPUT_STEM_DEFAULT}" ]]; then
  OUTPUT_STEM_DEFAULT="parameter_sweep"
fi
OUTPUT_STEM="${OUTPUT_STEM:-${OUTPUT_STEM_DEFAULT}}"
OUTPUT_DIR="${OUTPUT_DIR:-${SLURM_SUBMIT_DIR:-.}}"

if [[ ! "${NUM_JOBS}" =~ ^[0-9]+$ ]] || [[ "${NUM_JOBS}" -lt 1 ]]; then
  echo "NUM_JOBS must be a positive integer, got: ${NUM_JOBS}" >&2
  exit 1
fi

if [[ ! "${JOB_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "JOB_INDEX must be in [0, NUM_JOBS), got JOB_INDEX=${JOB_INDEX}, NUM_JOBS=${NUM_JOBS}" >&2
  exit 1
fi

# if [[ "${JOB_INDEX}" -ge "${NUM_JOBS}" ]]; then
#   echo "JOB_INDEX must be in [0, NUM_JOBS), got JOB_INDEX=${JOB_INDEX}, NUM_JOBS=${NUM_JOBS}" >&2
#   exit 1
# fi

mkdir -p "${OUTPUT_DIR}"

if [[ "${NUM_JOBS}" -gt 1 ]]; then
  OUTPUT_CSV="${OUTPUT_DIR}/${OUTPUT_STEM}_job${JOB_INDEX}.csv"
else
  OUTPUT_CSV="${OUTPUT_DIR}/${OUTPUT_STEM}.csv"
fi

APPEND="${APPEND:-0}"
DISPLAY_PROGRESS="${DISPLAY_PROGRESS:-1}"
LOG_EVERY="${LOG_EVERY:-5}"

cmd=(
  "${PYTHON_BIN}" -m "${RUNNER_MODULE}"
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
  --local-global-max-ratio "${LOCAL_GLOBAL_MAX_RATIO}"
  --output-csv "${OUTPUT_CSV}"
  --num-jobs "${NUM_JOBS}"
  --data-collection-period "${COLLECT_EVERY}"
  --job-index "${JOB_INDEX}"
  --log-every "${LOG_EVERY}"
)

if [[ -n "${LENGTH_SCALE_VALUES}" ]]; then
  cmd+=(--length-scale-values "${LENGTH_SCALE_VALUES}")
elif [[ -n "${LENGTH_SCALE_LOG_START}" || -n "${LENGTH_SCALE_LOG_STOP}" || -n "${LENGTH_SCALE_LOG_NUM}" ]]; then
  if [[ -z "${LENGTH_SCALE_LOG_START}" || -z "${LENGTH_SCALE_LOG_STOP}" || -z "${LENGTH_SCALE_LOG_NUM}" ]]; then
    echo "Length-scale log sweep requires LENGTH_SCALE_LOG_START, LENGTH_SCALE_LOG_STOP, and LENGTH_SCALE_LOG_NUM" >&2
    exit 1
  fi
  cmd+=(
    --length-scale-log-start "${LENGTH_SCALE_LOG_START}"
    --length-scale-log-stop "${LENGTH_SCALE_LOG_STOP}"
    --length-scale-log-num "${LENGTH_SCALE_LOG_NUM}"
    --length-scale-log-base "${LENGTH_SCALE_LOG_BASE}"
  )
else
  cmd+=(--length-scale-multipliers "${LENGTH_SCALE_MULTIPLIERS}")
fi

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
echo "RUNNER_MODULE: ${RUNNER_MODULE}"
echo "JOB_NAME: ${JOB_NAME}"
echo "OUTPUT_STEM: ${OUTPUT_STEM}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID:-none}"
echo "NUM_JOBS=${NUM_JOBS}, JOB_INDEX=${JOB_INDEX}"
if [[ -n "${LENGTH_SCALE_VALUES}" ]]; then
  echo "Length-scale sweep mode: absolute values (${LENGTH_SCALE_VALUES})"
elif [[ -n "${LENGTH_SCALE_LOG_START}" || -n "${LENGTH_SCALE_LOG_STOP}" || -n "${LENGTH_SCALE_LOG_NUM}" ]]; then
  echo "Length-scale sweep mode: log space start=${LENGTH_SCALE_LOG_START}, stop=${LENGTH_SCALE_LOG_STOP}, num=${LENGTH_SCALE_LOG_NUM}, base=${LENGTH_SCALE_LOG_BASE}"
else
  echo "Length-scale sweep mode: multipliers (${LENGTH_SCALE_MULTIPLIERS}) * lambda_true"
fi
echo "NUMBER_PROCESSES=${NUMBER_PROCESSES:-mesa-default}"
echo "Output CSV: ${OUTPUT_CSV}"

if ! "${PYTHON_BIN}" -c "import importlib; importlib.import_module('${RUNNER_MODULE}')" >/dev/null 2>&1; then
  echo "Python preflight failed: cannot import runner module '${RUNNER_MODULE}' with PYTHON_BIN=${PYTHON_BIN}" >&2
  echo "PYTHONPATH=${PYTHONPATH}" >&2
  exit 1
fi

"${cmd[@]}"
