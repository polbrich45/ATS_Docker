#!/usr/bin/env bash
# ==============================================================================
# Production runner script matching notebooks/optimize.ipynb (QUICK = False) exactly.
#
# Usage:
#   ./run.sh         # Run production run.py in foreground
#   ./run.sh --bg    # Run production run.py in background with nohup
# ==============================================================================

set -euo pipefail

# Repository root setup
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# JAX & GPU Environment Optimization
export JAX_COMPILATION_CACHE_DIR="${HOME}/.cache/jax"
# export XLA_PYTHON_CLIENT_PREALLOCATE=false
# export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
# export PYTHONUNBUFFERED=1
export JAX_ENABLE_X64=false

mkdir -p "${JAX_COMPILATION_CACHE_DIR}"
mkdir -p "${REPO_ROOT}/runs"

BACKGROUND=false
EXTRA_ARGS=()

for arg in "$@"; do
    case "${arg}" in
        --bg|--nohup|-b|-bg)
            BACKGROUND=true
            ;;
        *)
            EXTRA_ARGS+=("${arg}")
            ;;
    esac
done

timestamp="$(date +'%Y%m%d-%H%M%S')"
LOG_FILE="${REPO_ROOT}/runs/run_cli_${timestamp}.log"
OUT_DIR="${REPO_ROOT}/runs/production_${timestamp}"

echo "================================================================="
echo "Launching Production CLI Job: run.py (matching notebooks/optimize.ipynb)"
echo "Output Directory: ${OUT_DIR}"
echo "================================================================="

# Exact configuration from notebooks/optimize.ipynb (QUICK = False)
PARAM_ARGS=(
  --lattice square
  --hamiltonian j1j2
  --lx 4
  --ly 6
  --pbc
  --j1 1.0
  --j2 0.5
  --width 24
  --depth 2
  --heads 4
  --mlp-ratio 2
  --n-layers 2
  --family XY
  --n-outer 10
  --n-nqs 1000
  --n-samples 1024
  --chunk-size 128
  --learning-rate 2e-2
  --learning-rate-final 5e-3
  --lr-schedule cosine
  --optimizer sgd
  --sr-shift 1e-2
  --sr-shift-final 5e-3
  --n-rot 1000
  --n-subspace 1024
  --subspace-sampling frequency
  --n-seed-samples 10000
  --rotation-learning-rate 1e-3
  --participation-weight 1e-2
  --nested-weight 1e-5
  --f-retain 0.1
  --n-teacher 1000
  --teacher-learning-rate 1e-3
  --teacher-reinforce-weight 1e-1
  --teacher-entropy-weight 1e-1
  --teacher-phase-weight 1e-1
  --final-solver dmrg
  --dmrg-mps-bond-dim 128
  --output-dir "${OUT_DIR}"
)

if [ "${BACKGROUND}" = true ]; then
    echo "Running in background via nohup ..."
    echo "Log File: ${LOG_FILE}"
    nohup python3 "${REPO_ROOT}/run.py" \
      "${PARAM_ARGS[@]}" \
      "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" > "${LOG_FILE}" 2>&1 & disown
    echo "Background process disowned and running."
    echo "Monitor progress with: tail -f '${LOG_FILE}'"
else
    python3 "${REPO_ROOT}/run.py" \
      "${PARAM_ARGS[@]}" \
      "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
fi
