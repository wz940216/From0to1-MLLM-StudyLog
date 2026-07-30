#!/usr/bin/env bash
set -euo pipefail

# Benchmark the week19 backend vLLM OpenAI-compatible service directly.
# The week19 FastAPI service forwards requests to:
#   ${VLLM_BASE_URL}/v1/chat/completions

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/week20_benchmark_optimization/results/vllm}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000}"
MODEL="${MODEL:-minillava}"
TOKENIZER="${TOKENIZER:-week18_minillava_hf_vllm/outputs/vllm/minillava-hf}"
GPU_COUNT="${GPU_COUNT:-1}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-${GPU_COUNT}}"
PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-1}"

NUM_PROMPTS="${NUM_PROMPTS:-100}"
REQUEST_RATE="${REQUEST_RATE:-2}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-4}"
INPUT_LEN="${INPUT_LEN:-128}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"
TEMPERATURE="${TEMPERATURE:-0.0}"
BENCH_ENGINE="${BENCH_ENGINE:-openai}"
DATASET_NAME="${DATASET_NAME:-random}"
DATASET_PATH="${DATASET_PATH:-}"
IMAGE_PATH="${IMAGE_PATH:-${ROOT_DIR}/week03_mllm_overview_llava_demo/code/image.png}"
QUESTION="${QUESTION:-请描述这张图片。}"
BENCH_MULTIMODAL="${BENCH_MULTIMODAL:-0}"
WAIT_FOR_VLLM_TIMEOUT_S="${WAIT_FOR_VLLM_TIMEOUT_S:-300}"
WAIT_FOR_VLLM_INTERVAL_S="${WAIT_FOR_VLLM_INTERVAL_S:-5}"

mkdir -p "${RESULT_DIR}"

if [[ "${BENCH_ENGINE}" == "vllm-cli" ]]; then
  if ! command -v vllm >/dev/null 2>&1; then
    echo "ERROR: vllm command not found. Run this script inside the environment that installed vLLM." >&2
    exit 127
  fi
fi

echo "Waiting for vLLM at ${VLLM_BASE_URL}/v1/models, timeout=${WAIT_FOR_VLLM_TIMEOUT_S}s"
start_ts="$(date +%s)"
while ! curl -fsS "${VLLM_BASE_URL}/v1/models" >/dev/null; do
  now_ts="$(date +%s)"
  elapsed="$((now_ts - start_ts))"
  if [[ "${elapsed}" -ge "${WAIT_FOR_VLLM_TIMEOUT_S}" ]]; then
    echo "ERROR: vLLM service is not ready at ${VLLM_BASE_URL}/v1/models after ${elapsed}s" >&2
    echo "Check whether the vLLM HTTP server has finished loading and is listening on the expected port:" >&2
    echo "  curl -v ${VLLM_BASE_URL}/v1/models" >&2
    echo "  ss -lntp | grep ':8000\|:9000'" >&2
    echo "If vLLM was started on another host or port, rerun with VLLM_BASE_URL=http://host:port." >&2
    echo "Example start command:" >&2
    echo "  GPU_COUNT=${GPU_COUNT} bash week20_benchmark_optimization/start_vllm_server.sh" >&2
    exit 1
  fi
  echo "  still waiting... ${elapsed}s"
  sleep "${WAIT_FOR_VLLM_INTERVAL_S}"
done

echo "vLLM benchmark"
echo "  base_url       ${VLLM_BASE_URL}"
echo "  model          ${MODEL}"
echo "  tokenizer      ${TOKENIZER}"
echo "  gpu_count      ${GPU_COUNT}"
echo "  tensor_parallel_size ${TENSOR_PARALLEL_SIZE}"
echo "  pipeline_parallel_size ${PIPELINE_PARALLEL_SIZE}"
echo "  bench_engine   ${BENCH_ENGINE}"
echo "  dataset        ${DATASET_NAME}"
echo "  dataset_path   ${DATASET_PATH:-}"
echo "  image_path     ${IMAGE_PATH}"
echo "  bench_multimodal ${BENCH_MULTIMODAL}"
echo "  num_prompts    ${NUM_PROMPTS}"
echo "  request_rate   ${REQUEST_RATE}"
echo "  max_concurrency ${MAX_CONCURRENCY}"
echo "  random_input_len ${INPUT_LEN}"
echo "  random_output_len ${OUTPUT_LEN}"
echo "  result_dir     ${RESULT_DIR}"
echo "  wait_timeout_s ${WAIT_FOR_VLLM_TIMEOUT_S}"

if [[ "${BENCH_ENGINE}" == "openai" ]]; then
  python "${ROOT_DIR}/week20_benchmark_optimization/bench_vllm_openai.py"     --base-url "${VLLM_BASE_URL}"     --model "${MODEL}"     --image "${IMAGE_PATH}"     --question "${QUESTION}"     --num-prompts "${NUM_PROMPTS}"     --request-rate "${REQUEST_RATE}"     --max-concurrency "${MAX_CONCURRENCY}"     --max-tokens "${OUTPUT_LEN}"     --temperature "${TEMPERATURE}"     --result-dir "${RESULT_DIR}"     --result-filename "openai_${MODEL}_${TIMESTAMP}.jsonl"     "$@"
  exit $?
fi

BENCH_ARGS=(
  bench serve
  --backend openai-chat
  --base-url "${VLLM_BASE_URL}"
  --endpoint /v1/chat/completions
  --model "${MODEL}"
  --tokenizer "${TOKENIZER}"
  --dataset-name "${DATASET_NAME}"
  --num-prompts "${NUM_PROMPTS}"
  --request-rate "${REQUEST_RATE}"
  --max-concurrency "${MAX_CONCURRENCY}"
  --temperature "${TEMPERATURE}"
  --save-result
  --save-detailed
  --result-dir "${RESULT_DIR}"
  --result-filename "vllm_${MODEL}_${DATASET_NAME}_${TIMESTAMP}.json"
)

if [[ "${DATASET_NAME}" == "random" ]]; then
  BENCH_ARGS+=(--random-input-len "${INPUT_LEN}")
  BENCH_ARGS+=(--random-output-len "${OUTPUT_LEN}")
elif [[ "${DATASET_NAME}" == "random-mm" || "${BENCH_MULTIMODAL}" == "1" ]]; then
  BENCH_ARGS+=(--random-input-len "${INPUT_LEN}")
  BENCH_ARGS+=(--random-output-len "${OUTPUT_LEN}")
  BENCH_ARGS+=(--random-mm-limit-mm-per-prompt '{"image": 1, "video": 0}')
fi

vllm "${BENCH_ARGS[@]}" "$@"

