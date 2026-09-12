#!/usr/bin/env bash
set -euo pipefail

# D^3-MOPD Qwen3 launcher.
#
# The underlying launcher remains the matched vanilla M-OPD recipe.  This
# wrapper opts into the isolated D3DomainSampler and leaves the actor/teacher
# loss path unchanged.  The IF teacher follows the existing temporary Math
# fallback unless --if-teacher or IF_TEACHER_PATH is supplied.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPEN_MOPD_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DEFAULT_MATH_TEACHER="/nfs/dataset-ofs-prediction/rl_lab/liushiqi/llm/huggingface/models/Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500"
D3_SAMPLER_PATH="${OPEN_MOPD_ROOT}/training/verl/verl/utils/dataset/d3_domain_sampler.py"

export PROJECT_NAME="${PROJECT_NAME:-Qwen3-4B}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-d3_mopd}"
export OUTPUT_DIR="${OUTPUT_DIR:-${OPEN_MOPD_ROOT}/outputs/d3_mopd/qwen3_4b}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
export LOG_PROB_TOP_K="${LOG_PROB_TOP_K:-16}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"

IF_TEACHER_PATH="${IF_TEACHER_PATH:-}"
IF_TEACHER_EXPLICIT=0
SCHEDULER_STATUS_EXPLICIT=0
RUN_TRAINING=1
FORWARD_ARGS=()

die() {
    echo "[d3-mopd] ERROR: $*" >&2
    exit 2
}

usage() {
    cat <<'USAGE'
D^3-MOPD Qwen3 launcher.

Uses the matched vanilla M-OPD recipe and changes only the domain sampler.
The command runs by default; use --dry-run for a configuration check.

D^3 options:
  --if-teacher PATH             override the IF teacher (default: Math fallback)
  --scheduler-status PATH       write D^3 scheduler state to PATH

The remaining options are forwarded to default_mopd.sh, including --model,
--math-teacher, --code-teacher, --train, --val, --output, --checkpoint,
--python, --ray, --nodes, --node-rank, --gpus, --master-addr, --master-port,
--extra, --run, --dry-run, and --.
Environment overrides:
  D3_UPDATE_INTERVAL=10 D3_VELOCITY_WINDOW=10 D3_VELOCITY_WINDOWS=3
  D3_INITIAL_KL_STEPS=5 D3_EMA_WINDOW=10 D3_KL_FLOOR=0.15
  D3_TEMPERATURE=0.5 D3_MIXTURE_FLOOR=0.10 D3_BATCH_JITTER=0.30
USAGE
}

D3_UPDATE_INTERVAL="${D3_UPDATE_INTERVAL:-10}"
D3_VELOCITY_WINDOW="${D3_VELOCITY_WINDOW:-10}"
D3_VELOCITY_WINDOWS="${D3_VELOCITY_WINDOWS:-3}"
D3_INITIAL_KL_STEPS="${D3_INITIAL_KL_STEPS:-5}"
D3_EMA_WINDOW="${D3_EMA_WINDOW:-10}"
D3_KL_FLOOR="${D3_KL_FLOOR:-0.15}"
D3_TEMPERATURE="${D3_TEMPERATURE:-0.5}"
D3_MIXTURE_FLOOR="${D3_MIXTURE_FLOOR:-0.10}"
D3_BATCH_JITTER="${D3_BATCH_JITTER:-0.30}"
D3_STATUS_FILE="${D3_STATUS_FILE:-${OUTPUT_DIR}/d3_scheduler_status.json}"

while (($#)); do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --if-teacher)
            (($# >= 2)) || die "$1 requires a value"
            IF_TEACHER_PATH="$2"
            IF_TEACHER_EXPLICIT=1
            shift 2
            ;;
        --scheduler-status)
            (($# >= 2)) || die "$1 requires a value"
            D3_STATUS_FILE="$2"
            SCHEDULER_STATUS_EXPLICIT=1
            shift 2
            ;;
        --dry-run)
            RUN_TRAINING=0
            FORWARD_ARGS+=("$1")
            shift
            ;;
        --run|--exec)
            RUN_TRAINING=1
            FORWARD_ARGS+=("$1")
            shift
            ;;
        --output|--output-dir)
            (($# >= 2)) || die "$1 requires a value"
            OUTPUT_DIR="$2"
            export OUTPUT_DIR
            if ((SCHEDULER_STATUS_EXPLICIT == 0)); then
                D3_STATUS_FILE="${OUTPUT_DIR}/d3_scheduler_status.json"
            fi
            FORWARD_ARGS+=("$1" "$2")
            shift 2
            ;;
        --math-teacher)
            (($# >= 2)) || die "$1 requires a value"
            export MATH_TEACHER_PATH="$2"
            FORWARD_ARGS+=("$1" "$2")
            shift 2
            ;;
        --)
            FORWARD_ARGS+=("$@")
            break
            ;;
        *)
            FORWARD_ARGS+=("$1")
            shift
            ;;
    esac
done

if ((IF_TEACHER_EXPLICIT == 0)); then
    IF_TEACHER_PATH="${IF_TEACHER_PATH:-${MATH_TEACHER_PATH:-$DEFAULT_MATH_TEACHER}}"
fi

if ((RUN_TRAINING == 1)) && [[ ! -d "$IF_TEACHER_PATH" ]]; then
    die "IF teacher does not exist: $IF_TEACHER_PATH"
fi

[[ -f "$D3_SAMPLER_PATH" ]] || die "D3 sampler not found: $D3_SAMPLER_PATH"

echo "[d3-mopd] sampler=$D3_SAMPLER_PATH"
echo "[d3-mopd] if_teacher=$IF_TEACHER_PATH"
echo "[d3-mopd] status_file=$D3_STATUS_FILE"
echo "[d3-mopd] scheduler=gap*velocity update_interval=$D3_UPDATE_INTERVAL velocity_window=$D3_VELOCITY_WINDOW velocity_windows=$D3_VELOCITY_WINDOWS initial_kl_steps=$D3_INITIAL_KL_STEPS ema_window=$D3_EMA_WINDOW kl_floor=$D3_KL_FLOOR temperature=$D3_TEMPERATURE mixture_floor=$D3_MIXTURE_FLOOR batch_jitter=$D3_BATCH_JITTER"

# Put D^3 overrides before caller-supplied --extra values so an explicit
# caller override remains possible for ablations and smoke tests.
D3_ARGS=(
    --extra "data.sampler.class_path=file://${D3_SAMPLER_PATH}"
    --extra "data.sampler.class_name=D3DomainSampler"
    --extra "data.dataloader_num_workers=0"
    --extra "data.shuffle=False"
    --extra "+data.gen_batch_size=${TRAIN_BATCH_SIZE}"
    --extra "+data.sampler.domain_order=[math,code,if]"
    --extra "+data.sampler.d3_update_interval=${D3_UPDATE_INTERVAL}"
    --extra "+data.sampler.d3_velocity_window=${D3_VELOCITY_WINDOW}"
    --extra "+data.sampler.d3_velocity_windows=${D3_VELOCITY_WINDOWS}"
    --extra "+data.sampler.d3_initial_kl_steps=${D3_INITIAL_KL_STEPS}"
    --extra "+data.sampler.d3_ema_window=${D3_EMA_WINDOW}"
    --extra "+data.sampler.d3_kl_floor=${D3_KL_FLOOR}"
    --extra "+data.sampler.d3_temperature=${D3_TEMPERATURE}"
    --extra "+data.sampler.d3_mixture_floor=${D3_MIXTURE_FLOOR}"
    --extra "+data.sampler.d3_batch_jitter=${D3_BATCH_JITTER}"
    --extra "+data.sampler.status_file=${D3_STATUS_FILE}"
    # default_mopd.sh already creates this Hydra key with '+', so this final
    # override must omit '+' or Hydra rejects the duplicate creation.
    --extra "mt_reward_model_2.model.path=${IF_TEACHER_PATH}"
)

exec "${SCRIPT_DIR}/default_mopd.sh" "${D3_ARGS[@]}" "${FORWARD_ARGS[@]}"
