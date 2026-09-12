#!/usr/bin/env bash
set -euo pipefail

# Matched vanilla MT-OPD baseline for Open-MOPD ablations.
#
# This wrapper keeps the existing vanilla_mopd.sh algorithm configuration but
# matches Open-MOPD's update granularity and student-log-prob top-k:
#   train_batch_size=128, n=4, ppo_mini_batch_size=32, log_prob_top_k=16.
# It intentionally does not enable token-share balancing, gap-following
# allocation, or inner-update reward refresh.
#
# The launcher runs by default. Use --dry-run for a configuration check.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPEN_MOPD_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Keep this baseline's artifacts separate from both the legacy vanilla recipe
# and the Open-MOPD run. Explicit user-provided values still take precedence.
export PROJECT_NAME="${PROJECT_NAME:-Qwen3-4B}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-vanilla_mopd}"
export OUTPUT_DIR="${OUTPUT_DIR:-${OPEN_MOPD_ROOT}/outputs/vanilla_mopd/qwen3_4b_matched}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
export LOG_PROB_TOP_K="${LOG_PROB_TOP_K:-16}"

exec "${SCRIPT_DIR}/default_mopd.sh" "$@"
