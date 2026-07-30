#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-mllm}"
PYTHON_BIN="${PYTHON_BIN:-}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-9000}"
MINILLAVA_VLLM_BASE_URL="${MINILLAVA_VLLM_BASE_URL:-http://127.0.0.1:8000}"
MINILLAVA_MODEL="${MINILLAVA_MODEL:-minillava}"
MINILLAVA_MAX_CONCURRENCY="${MINILLAVA_MAX_CONCURRENCY:-4}"
MINILLAVA_RATE_LIMIT_PER_MIN="${MINILLAVA_RATE_LIMIT_PER_MIN:-10000}"

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export MINILLAVA_VLLM_BASE_URL
export MINILLAVA_MODEL
export MINILLAVA_MAX_CONCURRENCY
export MINILLAVA_RATE_LIMIT_PER_MIN

echo "Start FastAPI service"
echo "  root_dir                  ${ROOT_DIR}"
echo "  pythonpath                ${PYTHONPATH}"
echo "  conda_env_name            ${CONDA_ENV_NAME}"
echo "  host                      ${HOST}"
echo "  port                      ${PORT}"
echo "  minillava_vllm_base_url   ${MINILLAVA_VLLM_BASE_URL}"
echo "  minillava_model           ${MINILLAVA_MODEL}"
echo "  minillava_max_concurrency ${MINILLAVA_MAX_CONCURRENCY}"
echo "  rate_limit_per_min        ${MINILLAVA_RATE_LIMIT_PER_MIN}"

if [[ -n "${PYTHON_BIN}" ]]; then
  RUNNER=("${PYTHON_BIN}" -m uvicorn)
elif [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
  RUNNER=("${CONDA_EXE}" run -n "${CONDA_ENV_NAME}" python -m uvicorn)
elif command -v conda >/dev/null 2>&1; then
  RUNNER=(conda run -n "${CONDA_ENV_NAME}" python -m uvicorn)
elif [[ -x "${HOME}/miniconda3/bin/conda" ]]; then
  RUNNER=("${HOME}/miniconda3/bin/conda" run -n "${CONDA_ENV_NAME}" python -m uvicorn)
elif [[ -x "${HOME}/anaconda3/bin/conda" ]]; then
  RUNNER=("${HOME}/anaconda3/bin/conda" run -n "${CONDA_ENV_NAME}" python -m uvicorn)
elif [[ -x "/opt/conda/bin/conda" ]]; then
  RUNNER=("/opt/conda/bin/conda" run -n "${CONDA_ENV_NAME}" python -m uvicorn)
else
  echo "ERROR: conda was not found." >&2
  echo "Set CONDA_EXE=/path/to/conda, or run with PYTHON_BIN=/path/to/env/bin/python." >&2
  exit 127
fi

"${RUNNER[@]}" week19_end2end_service.service.app:app \
  --host "${HOST}" \
  --port "${PORT}"
