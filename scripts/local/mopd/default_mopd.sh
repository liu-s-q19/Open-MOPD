#!/usr/bin/env bash
set -euo pipefail

# Adaptive multi-node default MT-OPD launcher.
#
# This launcher intentionally does not implement reference construction.  It
# is the Qwen3 routed-MOPD baseline while the IF teacher is still training;
# the IF slot is temporarily backed by the Math teacher.
#
# The launcher runs by default.  Use --dry-run for a configuration check.
# Start it on every node for a real multi-node run.  Rank 0 starts Ray and
# launches the trainer; all other ranks start a Ray worker and remain blocked
# until the job is stopped.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPEN_MOPD_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/luban/miniconda3/envs/verl/bin/python}"
CONDA_SH="${CONDA_SH:-/home/luban/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-verl}"
RAY_BIN="${RAY_BIN:-${PYTHON_BIN%/*}/ray}"

DATA_ROOT="${DATA_ROOT:-/nfs/dataset-ofs-nmgvoyagermodel-prediction/rl_lab/liushiqi/llm/dataset}"
MODEL_PATH="${MODEL_PATH:-/nfs/dataset-ofs-prediction/rl_lab/liushiqi/llm/huggingface/models/Qwen/Qwen3-4B}"
MATH_TEACHER_PATH="${MATH_TEACHER_PATH:-/nfs/dataset-ofs-prediction/rl_lab/liushiqi/llm/huggingface/models/Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500}"
CODE_TEACHER_PATH="${CODE_TEACHER_PATH:-/nfs/dataset-ofs-prediction/rl_lab/liushiqi/llm/huggingface/models/Keven16/Qwen3-4B-Non-Thinking-RL-Code-Step300}"

TRAIN_FILE="${TRAIN_FILE:-${DATA_ROOT}/open_mopd/vanilla_mopd_4k/train.parquet}"
MATH_VAL_FILE="${MATH_VAL_FILE:-/nfs/dataset-ofs-prediction/rl_lab/liushiqi/llm/huggingface/datasets/AIME24/aime-2024-change_box.parquet}"
CODE_VAL_FILE="${CODE_VAL_FILE:-/nfs/dataset-ofs-prediction/rl_lab/liushiqi/llm/huggingface/datasets/Keven16/G-OPD-Training-Data-ours-math-author-code-pm1/Eurus/code_validation.parquet}"
IF_VAL_FILE="${IF_VAL_FILE:-${DATA_ROOT}/if_grpo/ifeval_only.parquet}"
# Comma-separated paths are converted into a Hydra list.  None of the default
# paths contains a comma; use a single combined parquet when a custom path
# itself contains commas.
VAL_FILES="${VAL_FILES:-${MATH_VAL_FILE},${CODE_VAL_FILE},${IF_VAL_FILE}}"

PROJECT_NAME="${PROJECT_NAME:-Qwen3-4B}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-default_mopd}"
OUTPUT_DIR="${OUTPUT_DIR:-${OPEN_MOPD_ROOT}/outputs/vanilla_mopd_if_as_math/qwen3_4b_adaptive}"
CKPT_DIR="${CKPT_DIR:-${OUTPUT_DIR}/checkpoints}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${OPEN_MOPD_ROOT}/tensorboard_log/${PROJECT_NAME}/${EXPERIMENT_NAME}}"

# Explicit NNODES/NODE_RANK take precedence over the scheduler variables.
NNODES="${NNODES:-${DISTRIBUTED_NODE_COUNT:-1}}"
NODE_RANK="${NODE_RANK:-${DISTRIBUTED_NODE_RANK:-0}}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    LOCAL_GPU_COUNT="${LOCAL_GPU_COUNT:-}"
    if [[ -z "$LOCAL_GPU_COUNT" ]] && command -v nvidia-smi >/dev/null 2>&1; then
        LOCAL_GPU_COUNT="$(nvidia-smi -L 2>/dev/null | awk 'NF {count += 1} END {print count + 0}')"
    fi
    if [[ "$LOCAL_GPU_COUNT" =~ ^[1-9][0-9]*$ ]]; then
        CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((LOCAL_GPU_COUNT - 1)))"
    else
        # Keep dry-runs useful in containers without nvidia-smi.  A real run
        # still validates the resulting GPU count through Ray.
        CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    fi
fi

MASTER_ADDR="${MASTER_ADDR:-${DISTRIBUTED_MASTER_HOSTS:-${VC_MASTER_HOSTS:-}}}"
MASTER_PORT="${MASTER_PORT:-${LUBAN_AVAILBLE_PORT_0:-${LUBAN_AVAILABLE_PORT_0:-${DISTRIBUTED_PYTORCH_PORT:-6386}}}}"
MASTER_ADDR="$(echo "${MASTER_ADDR}" | cut -d, -f1)"

RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_DASHBOARD_AGENT_GRPC_PORT="${RAY_DASHBOARD_AGENT_GRPC_PORT:-54405}"
RAY_DASHBOARD_AGENT_LISTEN_PORT="${RAY_DASHBOARD_AGENT_LISTEN_PORT:-54005}"
RAY_METRICS_EXPORT_PORT="${RAY_METRICS_EXPORT_PORT:-48002}"
RAY_RUNTIME_ENV_AGENT_PORT="${RAY_RUNTIME_ENV_AGENT_PORT:-51001}"
RAY_MIN_WORKER_PORT="${RAY_MIN_WORKER_PORT:-52000}"
RAY_MAX_WORKER_PORT="${RAY_MAX_WORKER_PORT:-53999}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-6000}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
N_RESPONSES="${N_RESPONSES:-4}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-128}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
REWARD_MODEL_MICRO_BATCH_SIZE_PER_GPU="${REWARD_MODEL_MICRO_BATCH_SIZE_PER_GPU:-${PPO_MICRO_BATCH_SIZE_PER_GPU}}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-256}"
# The prepared 12k-row train parquet yields about 94 batches/epoch at batch
# size 128.  Keep enough epochs for the explicit 256-step cap to be reached;
# custom datasets can override this independently.
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
SAVE_FREQ="${SAVE_FREQ:-16}"
TEST_FREQ="${TEST_FREQ:-16}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.4}"
LOG_PROB_TOP_K="${LOG_PROB_TOP_K:-256}"
RUN_TRAINING=1
HELP=0
EXTRA_OVERRIDES=()

die() {
    echo "[mt-opd-32gpu] ERROR: $*" >&2
    exit 2
}

usage() {
    cat <<'USAGE'
Adaptive multi-node default MT-OPD launcher.

The launcher runs by default. Use --dry-run for a configuration-only check.
Start it on every node for a real multi-node run. NNODES/NODE_RANK first use explicit variables, then
DISTRIBUTED_NODE_COUNT/DISTRIBUTED_NODE_RANK, and finally default to 1/0.
CUDA_VISIBLE_DEVICES is honored or detected with nvidia-smi.

Temporary teacher mapping: [Math, Code, Math] -> [math, code, if].

Options:
  --run                         start Ray/training (default)
  --dry-run                     print configuration only
  --model PATH                  override MODEL_PATH
  --math-teacher PATH           override MATH_TEACHER_PATH
  --code-teacher PATH           override CODE_TEACHER_PATH
  --train PATH                  override TRAIN_FILE
  --val PATH                    use one validation parquet
  --val-files CSV               comma-separated validation parquet paths
  --output PATH                 override OUTPUT_DIR
  --checkpoint PATH             override CKPT_DIR
  --python PATH                 override PYTHON_BIN
  --ray PATH                    override RAY_BIN
  --nodes N                     override NNODES
  --node-rank N                 override NODE_RANK
  --gpus LIST                   override CUDA_VISIBLE_DEVICES
  --master-addr HOST            override MASTER_ADDR
  --master-port PORT            override MASTER_PORT
  --extra VALUE                 append a Hydra override
  --                            pass remaining arguments to verl.trainer.main_ppo
USAGE
}

while (($#)); do
    case "$1" in
        -h|--help) HELP=1; shift ;;
        --run|--exec) RUN_TRAINING=1; shift ;;
        --dry-run) RUN_TRAINING=0; shift ;;
        --model|--model-path)
            (($# >= 2)) || die "$1 requires a value"
            MODEL_PATH="$2"
            shift 2
            ;;
        --math-teacher)
            (($# >= 2)) || die "$1 requires a value"
            MATH_TEACHER_PATH="$2"
            shift 2
            ;;
        --code-teacher)
            (($# >= 2)) || die "$1 requires a value"
            CODE_TEACHER_PATH="$2"
            shift 2
            ;;
        --train|--train-data)
            (($# >= 2)) || die "$1 requires a value"
            TRAIN_FILE="$2"
            shift 2
            ;;
        --val|--val-data)
            (($# >= 2)) || die "$1 requires a value"
            VAL_FILES="$2"
            shift 2
            ;;
        --val-files)
            (($# >= 2)) || die "$1 requires a value"
            VAL_FILES="$2"
            shift 2
            ;;
        --output|--output-dir)
            (($# >= 2)) || die "$1 requires a value"
            OUTPUT_DIR="$2"
            if [[ -z "${CHECKPOINT_DIR_EXPLICIT:-}" ]]; then
                CKPT_DIR="${OUTPUT_DIR}/checkpoints"
            fi
            shift 2
            ;;
        --checkpoint|--checkpoint-dir)
            (($# >= 2)) || die "$1 requires a value"
            CKPT_DIR="$2"
            CHECKPOINT_DIR_EXPLICIT=1
            shift 2
            ;;
        --python)
            (($# >= 2)) || die "$1 requires a value"
            PYTHON_BIN="$2"
            shift 2
            ;;
        --ray)
            (($# >= 2)) || die "$1 requires a value"
            RAY_BIN="$2"
            shift 2
            ;;
        --nodes|--nnodes)
            (($# >= 2)) || die "$1 requires a value"
            NNODES="$2"
            shift 2
            ;;
        --node-rank)
            (($# >= 2)) || die "$1 requires a value"
            NODE_RANK="$2"
            shift 2
            ;;
        --gpus)
            (($# >= 2)) || die "$1 requires a value"
            CUDA_VISIBLE_DEVICES="$2"
            shift 2
            ;;
        --master-addr)
            (($# >= 2)) || die "$1 requires a value"
            MASTER_ADDR="$2"
            shift 2
            ;;
        --master-port)
            (($# >= 2)) || die "$1 requires a value"
            MASTER_PORT="$2"
            shift 2
            ;;
        --extra)
            (($# >= 2)) || die "$1 requires a value"
            EXTRA_OVERRIDES+=("$2")
            shift 2
            ;;
        --)
            shift
            EXTRA_OVERRIDES+=("$@")
            break
            ;;
        *)
            EXTRA_OVERRIDES+=("$1")
            shift
            ;;
    esac
done

if ((HELP)); then
    usage
    exit 0
fi

[[ -f "$CONDA_SH" ]] || die "conda initialization script not found: $CONDA_SH"
source "$CONDA_SH"
conda activate "$CONDA_ENV"

VERIFIABLE_INSTRUCTIONS_ROOT="${OPEN_MOPD_ROOT}/training/third_party/verifiable-instructions"
export PYTHONPATH="${OPEN_MOPD_ROOT}/training/verl:${VERIFIABLE_INSTRUCTIONS_ROOT}:${OPEN_MOPD_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TENSORBOARD_DIR
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}"
export VLLM_ALLREDUCE_USE_SYMM_MEM="${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-INFO}"
export TORCH_DISTRIBUTED_TIMEOUT="${TORCH_DISTRIBUTED_TIMEOUT:-3600}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export CUDA_VISIBLE_DEVICES

[[ -x "$PYTHON_BIN" ]] || die "Python executable not found: $PYTHON_BIN"
[[ -x "$RAY_BIN" ]] || die "Ray executable not found: $RAY_BIN"
[[ -d "$OPEN_MOPD_ROOT/training/verl/verl" ]] || die "patched verl not found"

[[ "$NNODES" =~ ^[1-9][0-9]*$ ]] || die "NNODES must be positive: $NNODES"
[[ "$NODE_RANK" =~ ^[0-9]+$ ]] || die "NODE_RANK must be non-negative: $NODE_RANK"
((NODE_RANK < NNODES)) || die "NODE_RANK=$NODE_RANK must be smaller than NNODES=$NNODES"
[[ "$MASTER_PORT" =~ ^[0-9]+$ ]] || die "MASTER_PORT must be numeric: $MASTER_PORT"
[[ "$MAX_WAIT_SECONDS" =~ ^[1-9][0-9]*$ ]] || die "MAX_WAIT_SECONDS must be positive"

IFS=',' read -r -a VISIBLE_GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
VISIBLE_GPUS_PER_NODE=${#VISIBLE_GPU_ARRAY[@]}
EXPECTED_GPUS=$((NNODES * VISIBLE_GPUS_PER_NODE))
((VISIBLE_GPUS_PER_NODE > 0)) || die "CUDA_VISIBLE_DEVICES resolved to zero GPUs"

if ((NNODES > 1)) && [[ -z "$MASTER_ADDR" ]]; then
    if ((RUN_TRAINING)); then
        die "set DISTRIBUTED_MASTER_HOSTS=<head-node-ip> before --run"
    fi
    MASTER_ADDR="<head-node-ip>"
elif [[ -z "$MASTER_ADDR" ]]; then
    MASTER_ADDR="localhost"
fi

for ray_port in "$MASTER_PORT" "$RAY_DASHBOARD_PORT" "$RAY_DASHBOARD_AGENT_GRPC_PORT" \
    "$RAY_DASHBOARD_AGENT_LISTEN_PORT" "$RAY_METRICS_EXPORT_PORT" "$RAY_RUNTIME_ENV_AGENT_PORT"; do
    [[ "$ray_port" =~ ^[0-9]+$ ]] || die "Ray port must be numeric: $ray_port"
    if ((ray_port >= RAY_MIN_WORKER_PORT && ray_port <= RAY_MAX_WORKER_PORT)); then
        die "Ray component port $ray_port falls inside worker range $RAY_MIN_WORKER_PORT-$RAY_MAX_WORKER_PORT"
    fi
done

if ((RUN_TRAINING)); then
    [[ -d "$MODEL_PATH" ]] || die "model does not exist: $MODEL_PATH"
    [[ -f "$TRAIN_FILE" ]] || die "training parquet does not exist: $TRAIN_FILE"
    [[ -d "$MATH_TEACHER_PATH" ]] || die "Math teacher does not exist: $MATH_TEACHER_PATH"
    [[ -d "$CODE_TEACHER_PATH" ]] || die "Code teacher does not exist: $CODE_TEACHER_PATH"
fi

VAL_LIST="["
IFS=',' read -r -a VAL_ARRAY <<< "$VAL_FILES"
for val_path in "${VAL_ARRAY[@]}"; do
    [[ -n "$val_path" ]] || die "validation path cannot be empty"
    if ((RUN_TRAINING)); then
        [[ -f "$val_path" ]] || die "validation parquet does not exist: $val_path"
    fi
    [[ "$VAL_LIST" == "[" ]] || VAL_LIST+=","
    VAL_LIST+="'${val_path//\/\\}'"
done
VAL_LIST+="]"

TOTAL_TOKENS=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
DOMAIN_LIST="[math,code,if]"
GIT_COMMIT="unknown"
if command -v git >/dev/null 2>&1; then
    GIT_COMMIT="$(git -C "$OPEN_MOPD_ROOT" rev-parse --short HEAD 2>/dev/null || true)"
    [[ -n "$GIT_COMMIT" ]] || GIT_COMMIT="unknown"
fi

echo "[mt-opd-32gpu] python=$PYTHON_BIN"
echo "[mt-opd-32gpu] patched_verl=${OPEN_MOPD_ROOT}/training/verl"
echo "[mt-opd-32gpu] git_commit=$GIT_COMMIT"
echo "[mt-opd-32gpu] node_rank=$NODE_RANK nnodes=$NNODES visible_gpus=$VISIBLE_GPUS_PER_NODE expected_gpus=$EXPECTED_GPUS"
echo "[mt-opd-32gpu] master=$MASTER_ADDR:$MASTER_PORT"
echo "[mt-opd-32gpu] ray_ports dashboard=$RAY_DASHBOARD_PORT dashboard_agent_grpc=$RAY_DASHBOARD_AGENT_GRPC_PORT dashboard_agent=$RAY_DASHBOARD_AGENT_LISTEN_PORT metrics=$RAY_METRICS_EXPORT_PORT runtime=$RAY_RUNTIME_ENV_AGENT_PORT workers=$RAY_MIN_WORKER_PORT-$RAY_MAX_WORKER_PORT"
echo "[mt-opd-32gpu] model=$MODEL_PATH"
echo "[mt-opd-32gpu] teachers=[math:$MATH_TEACHER_PATH,code:$CODE_TEACHER_PATH,if:$MATH_TEACHER_PATH]"
echo "[mt-opd-32gpu] train=$TRAIN_FILE"
echo "[mt-opd-32gpu] val_files=$VAL_FILES"
echo "[mt-opd-32gpu] output=$OUTPUT_DIR checkpoint=$CKPT_DIR"
echo "[mt-opd-32gpu] vanilla train_batch=$TRAIN_BATCH_SIZE n=$N_RESPONSES ppo_mini=$PPO_MINI_BATCH_SIZE prompt_len=$MAX_PROMPT_LENGTH response_len=$MAX_RESPONSE_LENGTH epochs=$TOTAL_EPOCHS steps=$TOTAL_TRAINING_STEPS"

RAY_PORT_ARGS=(
    "--dashboard-port=$RAY_DASHBOARD_PORT"
    "--dashboard-agent-grpc-port=$RAY_DASHBOARD_AGENT_GRPC_PORT"
    "--dashboard-agent-listen-port=$RAY_DASHBOARD_AGENT_LISTEN_PORT"
    "--metrics-export-port=$RAY_METRICS_EXPORT_PORT"
    "--runtime-env-agent-port=$RAY_RUNTIME_ENV_AGENT_PORT"
    "--min-worker-port=$RAY_MIN_WORKER_PORT"
    "--max-worker-port=$RAY_MAX_WORKER_PORT"
)

TRAIN_CMD=(
    "$PYTHON_BIN" -m verl.trainer.main_ppo
    "algorithm.adv_estimator=token_reward_direct"
    "data.train_files=$TRAIN_FILE"
    "data.val_files=$VAL_LIST"
    "data.train_batch_size=$TRAIN_BATCH_SIZE"
    "data.max_prompt_length=$MAX_PROMPT_LENGTH"
    "data.max_response_length=$MAX_RESPONSE_LENGTH"
    # Keep the prepared 1:1:1 row mixture intact.  Three IF prompts are just
    # over 2048 tokens; left truncation preserves their rows instead of
    # silently filtering one domain more than the others.
    "data.filter_overlong_prompts=False"
    "data.val_filter_overlong_prompts=False"
    "data.truncation=left"
    "data.val_truncation=left"
    "data.shuffle=True"
    "data.seed=42"
    "+data.apply_chat_template_kwargs.enable_thinking=False"
    "actor_rollout_ref.model.path=$MODEL_PATH"
    "actor_rollout_ref.model.use_remove_padding=True"
    "actor_rollout_ref.model.enable_gradient_checkpointing=True"
    "actor_rollout_ref.actor.policy_loss.loss_mode=vanilla"
    "actor_rollout_ref.actor.loss_agg_mode=token-mean"
    "actor_rollout_ref.actor.use_dynamic_bsz=True"
    "actor_rollout_ref.actor.ppo_epochs=1"
    "actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU"
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$TOTAL_TOKENS"
    "actor_rollout_ref.actor.optim.lr=$ACTOR_LR"
    "actor_rollout_ref.actor.optim.weight_decay=$WEIGHT_DECAY"
    "actor_rollout_ref.actor.grad_clip=1.0"
    "actor_rollout_ref.actor.entropy_coeff=0"
    "actor_rollout_ref.actor.fsdp_config.fsdp_size=-1"
    "actor_rollout_ref.actor.fsdp_config.param_offload=False"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False"
    "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True"
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU"
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$TOTAL_TOKENS"
    "actor_rollout_ref.ref.fsdp_config.param_offload=False"
    "actor_rollout_ref.rollout.name=vllm"
    "+actor_rollout_ref.rollout.reward_mode=mt_opd"
    "actor_rollout_ref.rollout.n=$N_RESPONSES"
    "actor_rollout_ref.rollout.temperature=1.0"
    "actor_rollout_ref.rollout.top_p=1.0"
    "actor_rollout_ref.rollout.top_k=-1"
    "+actor_rollout_ref.rollout.log_prob_top_k=$LOG_PROB_TOP_K"
    "+actor_rollout_ref.rollout.top_k_strategy=only_stu"
    "+actor_rollout_ref.rollout.reward_weight_mode=student_p"
    "actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
    "actor_rollout_ref.rollout.enable_chunked_prefill=True"
    "actor_rollout_ref.rollout.max_num_batched_tokens=$TOTAL_TOKENS"
    "actor_rollout_ref.rollout.max_model_len=$TOTAL_TOKENS"
    "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU"
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$TOTAL_TOKENS"
    "actor_rollout_ref.rollout.val_kwargs.n=1"
    "actor_rollout_ref.rollout.val_kwargs.do_sample=False"
    "reward_model.enable=True"
    "reward_model.model.path=$MATH_TEACHER_PATH"
    "reward_model.model.input_tokenizer=null"
    "reward_model.micro_batch_size_per_gpu=$REWARD_MODEL_MICRO_BATCH_SIZE_PER_GPU"
    "+reward_model.reward_kwargs.compute_true_reward=False"
    "+mt_opd.teacher_domains=$DOMAIN_LIST"
    "+mt_opd.n_additional_teachers=2"
    "trainer.n_gpus_per_node=$VISIBLE_GPUS_PER_NODE"
    "trainer.nnodes=$NNODES"
    "trainer.total_epochs=$TOTAL_EPOCHS"
    "trainer.total_training_steps=$TOTAL_TRAINING_STEPS"
    "trainer.default_local_dir=$CKPT_DIR"
    "trainer.project_name=$PROJECT_NAME"
    "trainer.experiment_name=$EXPERIMENT_NAME"
    "trainer.logger=['console','tensorboard']"
    "trainer.val_before_train=True"
    "trainer.test_freq=$TEST_FREQ"
    "trainer.save_freq=$SAVE_FREQ"
    "trainer.resume_mode=auto"
    "+ray_kwargs.ray_init.address=auto"
    "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=$PYTHONPATH"
    "+ray_kwargs.ray_init.runtime_env.env_vars.TENSORBOARD_DIR=$TENSORBOARD_DIR"
)

# Teacher 1 is Code and teacher 2 is the temporary Math replacement for IF.
TRAIN_CMD+=(
    "+mt_reward_model_1.enable=True"
    "+mt_reward_model_1.model.path=$CODE_TEACHER_PATH"
    "+mt_reward_model_1.model.input_tokenizer=null"
    "+mt_reward_model_1.model.use_remove_padding=True"
    "+mt_reward_model_1.model.fsdp_config.param_offload=True"
    "+mt_reward_model_2.enable=True"
    "+mt_reward_model_2.model.path=$MATH_TEACHER_PATH"
    "+mt_reward_model_2.model.input_tokenizer=null"
    "+mt_reward_model_2.model.use_remove_padding=True"
    "+mt_reward_model_2.model.fsdp_config.param_offload=True"
)
TRAIN_CMD+=("${EXTRA_OVERRIDES[@]}")

printf '[mt-opd-32gpu] command:'
printf ' %q' "${TRAIN_CMD[@]}"
printf '\n'

if [[ "$RUN_TRAINING" != 1 ]]; then
    echo "[mt-opd-32gpu] dry-run only; pass --run or omit --dry-run to execute"
    exit 0
fi

cleanup_ray() {
    if ((NODE_RANK == 0)); then
        "$RAY_BIN" stop >/dev/null 2>&1 || true
    fi
}
trap cleanup_ray EXIT

if ((NODE_RANK == 0)); then
    "$RAY_BIN" start --head \
        --port="$MASTER_PORT" \
        --node-ip-address="$MASTER_ADDR" \
        --include-dashboard=false \
        "${RAY_PORT_ARGS[@]}"
else
    until "$RAY_BIN" start --address="$MASTER_ADDR:$MASTER_PORT" "${RAY_PORT_ARGS[@]}" --block; do
        echo "[mt-opd-32gpu] worker $NODE_RANK could not connect to Ray head; retrying in 3s..."
        sleep 3
    done
fi

if ((NODE_RANK == 0)); then
    START_TIME=$(date +%s)
    while true; do
        RAY_GPU_STATUS=$("$PYTHON_BIN" - <<'PY'
import ray
from ray._private.state import available_resources_per_node

try:
    ray.init(address="auto", logging_level="error", ignore_reinit_error=True)
    cluster_gpus = ray.cluster_resources().get("GPU", 0)
    per_node = available_resources_per_node()
    available_gpus = sum(node_info.get("GPU", node_info.get("NPU", 0)) for node_info in per_node.values())
    print(f"{int(cluster_gpus)} {int(available_gpus)}")
    ray.shutdown()
except Exception:
    print("0 0")
PY
        )
        read -r CURRENT_GPUS AVAILABLE_GPUS <<< "$RAY_GPU_STATUS"
        ELAPSED=$(( $(date +%s) - START_TIME ))
        echo "[mt-opd-32gpu] [$ELAPSED/$MAX_WAIT_SECONDS] cluster=$CURRENT_GPUS/$EXPECTED_GPUS available=$AVAILABLE_GPUS/$EXPECTED_GPUS"
        if ((CURRENT_GPUS >= EXPECTED_GPUS && AVAILABLE_GPUS >= EXPECTED_GPUS)); then
            break
        fi
        if ((ELAPSED > MAX_WAIT_SECONDS)); then
            die "timed out waiting for $EXPECTED_GPUS GPUs"
        fi
        sleep 5
    done

    mkdir -p "$OUTPUT_DIR" "$CKPT_DIR"
    # Hydra's own run directory is relative to the caller's cwd.  Anchor it
    # at the repository so logs never reappear under scripts/local/**/outputs.
    cd "$OPEN_MOPD_ROOT"
    "${TRAIN_CMD[@]}"
else
    echo "[mt-opd-32gpu] worker rank $NODE_RANK is attached to Ray; waiting for rank 0"
fi
