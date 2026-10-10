#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
MATERIAL_QUALITY_DIR="${SCRIPT_DIR:h}"
BACKEND_DIR="${MATERIAL_QUALITY_DIR:h}"
RUNTIME_DIR="${MATERIAL_QUALITY_DIR}/.runtime"
PYTHON_BIN="${MATERIAL_QUALITY_PYTHON:-${MATERIAL_QUALITY_DIR}/.venv-local/bin/python}"
STOP_FILE="${RUNTIME_DIR}/STOP"
ENV_FILE="${MATERIAL_QUALITY_DIR}/.env.local"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "本地素材质量Python环境不存在：${PYTHON_BIN}"
  exit 2
fi
if [[ -e "${STOP_FILE}" ]]; then
  echo "检测到安全停止文件，请确认后删除：${STOP_FILE}"
  exit 2
fi

mkdir -p \
  "${RUNTIME_DIR}/temp" \
  "${RUNTIME_DIR}/models/chinese_clip" \
  "${RUNTIME_DIR}/models/faster_whisper" \
  "${RUNTIME_DIR}/parallel_slots"

export MATERIAL_QUALITY_DB_HOST="${MATERIAL_QUALITY_DB_HOST:-127.0.0.1}"
export MATERIAL_QUALITY_DB_PORT="${MATERIAL_QUALITY_DB_PORT:-13306}"
export MATERIAL_QUALITY_URL_INTERVAL_SECONDS="${MATERIAL_QUALITY_URL_INTERVAL_SECONDS:-1.0}"
export MATERIAL_QUALITY_RATE_LIMIT_FILE="${MATERIAL_QUALITY_RATE_LIMIT_FILE:-${RUNTIME_DIR}/url_rate_limit.json}"
# 各个推理引擎都要受线程预算约束，避免三个素材进程重复占满全部CPU。
export OMP_NUM_THREADS="${MATERIAL_QUALITY_OMP_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${MATERIAL_QUALITY_BLAS_THREADS:-2}"
export VECLIB_MAXIMUM_THREADS="${MATERIAL_QUALITY_BLAS_THREADS:-2}"
# macOS后台任务不会继承交互式Shell路径，固定加入Homebrew和系统工具目录。
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"

cd "${BACKEND_DIR}"
exec "${PYTHON_BIN}" -u -m material_quality.feature.semantic_worker \
  --url-mode local_readonly \
  --state-mode local \
  --local-state-db "${RUNTIME_DIR}/local_tasks.sqlite3" \
  --raw-feature-root "${RUNTIME_DIR}/raw_features" \
  --model-source huggingface \
  --model-cache "${RUNTIME_DIR}/models/chinese_clip" \
  --asr-model-cache "${RUNTIME_DIR}/models/faster_whisper" \
  --temp-root "${RUNTIME_DIR}/temp" \
  --stop-file "${STOP_FILE}" \
  --parallel-slot-dir "${RUNTIME_DIR}/parallel_slots" \
  --asr-extra-slots 1 \
  --min-free-disk-gb 20 \
  --torch-threads 2 \
  --ocr-cpu-threads 2 \
  --ocr-worker-count 2 \
  --asr-cpu-threads 2 \
  --asr-num-workers 2 \
  --asr-split-min-seconds 45 \
  "$@"
