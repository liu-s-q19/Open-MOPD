#!/usr/bin/env bash
set -euo pipefail

# Download IF-only RL training data and IF evaluation data.
# The training source is the Nemotron IF-RL JSONL consumed by
# training/scripts/rl/build_nemotron_if_rl_dataset.py.  The evaluation files
# are the aligned IFEval and IFBench files released with Open-MOPD.

DATASET_ROOT="${DATASET_ROOT:-/nfs/dataset-ofs-nmgvoyagermodel-prediction/rl_lab/liushiqi/llm/dataset}"
TRAIN_REPO_ID="${TRAIN_REPO_ID:-nvidia/Nemotron-RL-instruction_following}"
EVAL_REPO_ID="${EVAL_REPO_ID:-BytedTsinghua-SIA/Open-MOPD-Data}"
TRAIN_DIR="${TRAIN_DIR:-${DATASET_ROOT}/${TRAIN_REPO_ID}}"
EVAL_DIR="${EVAL_DIR:-${DATASET_ROOT}/${EVAL_REPO_ID}}"
HF_MAX_WORKERS="${HF_MAX_WORKERS:-4}"
DOWNLOAD_BACKEND="${DOWNLOAD_BACKEND:-auto}"
HF_DOWNLOAD_ENDPOINT="${HF_DOWNLOAD_ENDPOINT:-${HF_ENDPOINT:-https://hf-mirror.com}}"

command -v hf >/dev/null 2>&1 || {
  echo "[download] error: hf CLI not found in PATH" >&2
  exit 1
}

[[ "${HF_MAX_WORKERS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "[download] error: HF_MAX_WORKERS must be a positive integer" >&2
  exit 1
}

[[ "${DOWNLOAD_BACKEND}" == "auto" || "${DOWNLOAD_BACKEND}" == "hf" || "${DOWNLOAD_BACKEND}" == "curl" ]] || {
  echo "[download] error: DOWNLOAD_BACKEND must be auto, hf, or curl" >&2
  exit 1
}

download_with_curl() {
  local repo_id="$1"
  local repo_file="$2"
  local output_dir="$3"
  local expected_size="$4"
  local output_path="${output_dir}/${repo_file}"
  local url="${HF_DOWNLOAD_ENDPOINT}/datasets/${repo_id}/resolve/main/${repo_file}?download=true"

  mkdir -p "$(dirname "${output_path}")"
  if [[ -f "${output_path}" ]] && [[ "$(stat -c '%s' "${output_path}")" == "${expected_size}" ]]; then
    echo "[download] already complete: ${output_path}"
    return 0
  fi

  echo "[download] curl fallback: ${url}"
  curl --fail --location --retry 5 --retry-delay 2 --continue-at - \
    --output "${output_path}" "${url}"
  [[ "$(stat -c '%s' "${output_path}")" == "${expected_size}" ]] || {
    echo "[download] error: unexpected size for ${output_path}" >&2
    exit 1
  }
}

download_one() {
  local repo_id="$1"
  local repo_file="$2"
  local output_dir="$3"
  local expected_size="$4"

  if [[ "${DOWNLOAD_BACKEND}" != "curl" ]]; then
    if HF_ENDPOINT="${HF_DOWNLOAD_ENDPOINT}" hf download "${repo_id}" "${repo_file}" \
      --repo-type dataset \
      --local-dir "${output_dir}" \
      --max-workers "${HF_MAX_WORKERS}"; then
      return 0
    fi
    [[ "${DOWNLOAD_BACKEND}" == "hf" ]] && return 1
    echo "[download] hf failed; trying curl fallback for ${repo_id}/${repo_file}" >&2
  fi

  download_with_curl "${repo_id}" "${repo_file}" "${output_dir}" "${expected_size}"
}

mkdir -p "${TRAIN_DIR}" "${EVAL_DIR}"

echo "[download] hf=$(command -v hf)"
hf version
echo "[download] train_repo=${TRAIN_REPO_ID}"
echo "[download] train_dir=${TRAIN_DIR}"
echo "[download] eval_repo=${EVAL_REPO_ID}"
echo "[download] eval_dir=${EVAL_DIR}"
echo "[download] curl_endpoint=${HF_DOWNLOAD_ENDPOINT}"

download_one \
  "${TRAIN_REPO_ID}" \
  instruction_following.jsonl \
  "${TRAIN_DIR}" \
  77880361

download_one \
  "${EVAL_REPO_ID}" \
  eval/if/ifeval_aligned.parquet \
  "${EVAL_DIR}" \
  268896

download_one \
  "${EVAL_REPO_ID}" \
  eval/if/ifbench_test_aligned.parquet \
  "${EVAL_DIR}" \
  215622

echo "[download] training files:"
find "${TRAIN_DIR}" -maxdepth 2 -type f -print | sort
echo "[download] evaluation files:"
find "${EVAL_DIR}" -maxdepth 4 -type f -print | sort
echo "[download] completed"
