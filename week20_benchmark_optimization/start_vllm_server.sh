#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GPU_COUNT="${GPU_COUNT:-1}"
GPU_DEVICES="${GPU_DEVICES:-}"
MODEL_PATH="${MODEL_PATH:-week18_minillava_hf_vllm/outputs/vllm/minillava-hf}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-minillava}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DTYPE="${DTYPE:-float16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-${GPU_COUNT}}"
PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vllm_test}"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ "${GPU_COUNT}" -lt 1 ]]; then
  echo "ERROR: GPU_COUNT must be >= 1, got ${GPU_COUNT}" >&2
  exit 1
fi

if [[ -z "${GPU_DEVICES}" ]]; then
  GPU_DEVICES="$(seq -s, 0 "$((GPU_COUNT - 1))")"
fi

echo "Start vLLM server"
echo "  gpu_devices            ${GPU_DEVICES}"
echo "  gpu_count              ${GPU_COUNT}"
echo "  tensor_parallel_size   ${TENSOR_PARALLEL_SIZE}"
echo "  pipeline_parallel_size ${PIPELINE_PARALLEL_SIZE}"
echo "  model_path             ${MODEL_PATH}"
echo "  served_model_name      ${SERVED_MODEL_NAME}"
echo "  conda_env_name         ${CONDA_ENV_NAME}"
echo "  endpoint               http://${HOST}:${PORT}/v1/chat/completions"

cd "${ROOT_DIR}"

if [[ -n "${PYTHON_BIN}" ]]; then
  RUNNER=("${PYTHON_BIN}")
elif [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
  RUNNER=("${CONDA_EXE}" run -n "${CONDA_ENV_NAME}" python)
elif command -v conda >/dev/null 2>&1; then
  RUNNER=(conda run -n "${CONDA_ENV_NAME}" python)
elif [[ -x "${HOME}/miniconda3/bin/conda" ]]; then
  RUNNER=("${HOME}/miniconda3/bin/conda" run -n "${CONDA_ENV_NAME}" python)
elif [[ -x "${HOME}/anaconda3/bin/conda" ]]; then
  RUNNER=("${HOME}/anaconda3/bin/conda" run -n "${CONDA_ENV_NAME}" python)
elif [[ -x "/opt/conda/bin/conda" ]]; then
  RUNNER=("/opt/conda/bin/conda" run -n "${CONDA_ENV_NAME}" python)
else
  echo "ERROR: conda was not found." >&2
  echo "Set CONDA_EXE=/path/to/conda, or run with PYTHON_BIN=/path/to/env/bin/python." >&2
  exit 127
fi

CUDA_VISIBLE_DEVICES="${GPU_DEVICES}" "${RUNNER[@]}" week18_minillava_hf_vllm/scripts/vllm_openai_server.py \
  --model-path "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --dtype "${DTYPE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --pipeline-parallel-size "${PIPELINE_PARALLEL_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}"
