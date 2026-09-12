#!/usr/bin/env bash
set -euo pipefail

# Adaptive-resource IF-GRPO launcher.
# Ray startup and port handling follow:
# /nfs/dataset-ofs-nmgvoyagermodel-prediction/rl_lab/liushiqi/llm/verl-main/scripts/opd_rl/grpo_4b_box_opd_data_justrl.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPEN_MOPD_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/luban/miniconda3/envs/verl/bin/python}"
CONDA_SH="${CONDA_SH:-/home/luban/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-verl}"
RAY_BIN="${RAY_BIN:-${PYTHON_BIN%/*}/ray}"

TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-/nfs/dataset-ofs-nmgvoyagermodel-prediction/rl_lab/liushiqi/llm/dataset}"
MODEL_PATH="${MODEL_PATH:-/nfs/dataset-ofs-prediction/rl_lab/liushiqi/llm/huggingface/models/Qwen/Qwen3-4B}"
TRAIN_JSONL="${TRAIN_JSONL:-${TRAIN_DATA_ROOT}/train/instruction_following.jsonl}"
IFEVAL_PARQUET="${IFEVAL_PARQUET:-${TRAIN_DATA_ROOT}/eval/ifeval_aligned.parquet}"
IFBENCH_PARQUET="${IFBENCH_PARQUET:-${TRAIN_DATA_ROOT}/eval/ifbench_test_aligned.parquet}"
DERIVED_DATA_DIR="${DERIVED_DATA_DIR:-${TRAIN_DATA_ROOT}/if_grpo}"
TRAIN_FILE="${TRAIN_FILE:-${DERIVED_DATA_DIR}/nemotron_if_rl_train.parquet}"
VAL_FILE="${VAL_FILE:-}"
DATA_SUMMARY="${DATA_SUMMARY:-}"
DATA_READY_MARKER="${DATA_READY_MARKER:-}"
DATA_WAIT_SECONDS="${DATA_WAIT_SECONDS:-600}"

PROJECT_NAME="${PROJECT_NAME:-Open-MOPD-IF-GRPO-Qwen3-4B}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-if_grpo_32gpu}"
OUTPUT_DIR="${OUTPUT_DIR:-${OPEN_MOPD_ROOT}/outputs/if_grpo_32gpu/qwen3_4b}"
CKPT_DIR="${CKPT_DIR:-${OUTPUT_DIR}/checkpoints}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${OPEN_MOPD_ROOT}/tensorboard_log/${PROJECT_NAME}/${EXPERIMENT_NAME}}"

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

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-8}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-15360}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-10}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-5}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1000}"
TEST_FREQ="${TEST_FREQ:-5}"
SAVE_FREQ="${SAVE_FREQ:-5}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-True}"
OFFLOAD="${OFFLOAD:-False}"
# vLLM starts after the actor weights are resident on each GPU.  Keep enough
# headroom for that resident model and any colocated Ray/PyTorch allocations.
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.4}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.7}"
VAL_TOP_P="${VAL_TOP_P:-0.9}"
REWARD_SCORING_MODE="${REWARD_SCORING_MODE:-tiered_think_format}"
REBUILD_DATA="${REBUILD_DATA:-0}"
INCLUDE_IFBENCH_VAL="${INCLUDE_IFBENCH_VAL:-auto}"
RUN_TRAINING=1
HELP=0
EXTRA_OVERRIDES=()

die() {
    echo "[if-grpo] ERROR: $*" >&2
    exit 2
}

usage() {
    cat <<'USAGE'
IF-GRPO launcher for Qwen3-4B with adaptive node/GPU resources.

The launcher runs by default. Use --dry-run for configuration-only checks.
For multi-node runs, set DISTRIBUTED_MASTER_HOSTS and rank each node.

Options:
  --run                 start Ray and launch training (default)
  --dry-run             print configuration only
  --rebuild-data        rebuild derived IF train/validation parquets
  --include-ifbench     include IFBench official validation (requires checkout)
  --ifeval-only         use only IFEval validation (default without checkout)
  --model PATH          override MODEL_PATH
  --output PATH         override OUTPUT_DIR
  --extra VALUE         append a Hydra override
  --                  pass remaining arguments to verl.trainer.main_ppo
USAGE
}

while (($#)); do
    case "$1" in
        -h|--help) HELP=1; shift ;;
        --run|--exec) RUN_TRAINING=1; shift ;;
        --dry-run) RUN_TRAINING=0; shift ;;
        --rebuild-data) REBUILD_DATA=1; shift ;;
        --include-ifbench) INCLUDE_IFBENCH_VAL=1; shift ;;
        --ifeval-only) INCLUDE_IFBENCH_VAL=0; shift ;;
        --model|--model-path)
            (($# >= 2)) || die "$1 requires a value"
            MODEL_PATH="$2"
            shift 2
            ;;
        --output|--output-dir)
            (($# >= 2)) || die "$1 requires a value"
            OUTPUT_DIR="$2"
            CKPT_DIR="${OUTPUT_DIR}/checkpoints"
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
[[ -d "$VERIFIABLE_INSTRUCTIONS_ROOT/verifiable_instructions" ]] || die "verifiable-instructions package not found: $VERIFIABLE_INSTRUCTIONS_ROOT"

if ! "$PYTHON_BIN" - <<'PY'
import json

from verl.utils.reward_score.instruction_following import compute_score
from verifiable_instructions import instructions_registry

assert instructions_registry.INSTRUCTION_DICT, "empty verifiable-instructions registry"
result = compute_score(
    solution_str="<think>\n</think>hello",
    ground_truth="",
    data_source="nemotron_if_rl",
    extra_info={
        "instruction_id_list": ["keywords:existence"],
        "instruction_kwargs_json": [json.dumps({"keywords": ["hello"]})],
        "family": "keywords",
        "raw_prompt": "say hello",
    },
    scoring_mode="tiered_think_format",
)
assert result["score"] == 1.0, result
print(f"[if-grpo] reward_preflight=ok registry={len(instructions_registry.INSTRUCTION_DICT)}")
PY
then
    die "IF reward preflight failed; verify PYTHONPATH and training/third_party/verifiable-instructions on this node"
fi

REWARD_IFBENCH_REPO="${OPENOPD_IFBENCH_REPO:-}"
case "$INCLUDE_IFBENCH_VAL" in
    auto)
        if [[ -n "$REWARD_IFBENCH_REPO" && -d "$REWARD_IFBENCH_REPO" ]]; then
            INCLUDE_IFBENCH_VAL=1
        else
            INCLUDE_IFBENCH_VAL=0
        fi
        ;;
    0|1|true|false|True|False)
        if [[ "$INCLUDE_IFBENCH_VAL" == 1 || "$INCLUDE_IFBENCH_VAL" == true || "$INCLUDE_IFBENCH_VAL" == True ]]; then
            INCLUDE_IFBENCH_VAL=1
        else
            INCLUDE_IFBENCH_VAL=0
        fi
        ;;
    *) die "INCLUDE_IFBENCH_VAL must be auto, 0, or 1" ;;
esac
if ((INCLUDE_IFBENCH_VAL)); then
    [[ -n "$REWARD_IFBENCH_REPO" && -d "$REWARD_IFBENCH_REPO" ]] || die "set OPENOPD_IFBENCH_REPO to the IFBench checkout, or use --ifeval-only"
    export OPENOPD_IFBENCH_REPO="$REWARD_IFBENCH_REPO"
else
    echo "[if-grpo] IFBench checkout unavailable; using IFEval-only validation"
fi

if [[ -z "$VAL_FILE" ]]; then
    if ((INCLUDE_IFBENCH_VAL)); then
        VAL_FILE="${DERIVED_DATA_DIR}/if_eval.parquet"
    else
        VAL_FILE="${DERIVED_DATA_DIR}/ifeval_only.parquet"
    fi
fi
if [[ -z "$DATA_SUMMARY" ]]; then
    if ((INCLUDE_IFBENCH_VAL)); then
        DATA_SUMMARY="${DERIVED_DATA_DIR}/if_grpo.summary.json"
    else
        DATA_SUMMARY="${DERIVED_DATA_DIR}/ifeval_only.summary.json"
    fi
fi
if [[ -z "$DATA_READY_MARKER" ]]; then
    DATA_READY_MARKER="${DATA_SUMMARY}.ready"
fi
[[ "$NNODES" =~ ^[1-9][0-9]*$ ]] || die "NNODES must be positive: $NNODES"
[[ "$NODE_RANK" =~ ^[0-9]+$ ]] || die "NODE_RANK must be non-negative: $NODE_RANK"
((NODE_RANK < NNODES)) || die "NODE_RANK=$NODE_RANK must be smaller than NNODES=$NNODES"

IFS=',' read -r -a VISIBLE_GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
VISIBLE_GPUS_PER_NODE=${#VISIBLE_GPU_ARRAY[@]}
EXPECTED_GPUS=$((NNODES * VISIBLE_GPUS_PER_NODE))
((VISIBLE_GPUS_PER_NODE > 0)) || die "CUDA_VISIBLE_DEVICES resolved to zero GPUs"
((EXPECTED_GPUS > 0)) || die "expected GPU count must be positive"

if ((NNODES > 1)) && [[ -z "$MASTER_ADDR" ]]; then
    if ((RUN_TRAINING)); then
        die "set DISTRIBUTED_MASTER_HOSTS=<head-node-ip> before --run"
    fi
    MASTER_ADDR="<head-node-ip>"
elif [[ -z "$MASTER_ADDR" ]]; then
    MASTER_ADDR="localhost"
fi

[[ "$MASTER_PORT" =~ ^[0-9]+$ ]] || die "MASTER_PORT must be numeric: $MASTER_PORT"
for ray_port in "$MASTER_PORT" "$RAY_DASHBOARD_PORT" "$RAY_DASHBOARD_AGENT_GRPC_PORT" \
    "$RAY_DASHBOARD_AGENT_LISTEN_PORT" "$RAY_METRICS_EXPORT_PORT" "$RAY_RUNTIME_ENV_AGENT_PORT"; do
    if ((ray_port >= RAY_MIN_WORKER_PORT && ray_port <= RAY_MAX_WORKER_PORT)); then
        die "Ray component port $ray_port falls inside worker range $RAY_MIN_WORKER_PORT-$RAY_MAX_WORKER_PORT"
    fi
done

RAY_PORT_ARGS=(
    "--dashboard-port=$RAY_DASHBOARD_PORT"
    "--dashboard-agent-grpc-port=$RAY_DASHBOARD_AGENT_GRPC_PORT"
    "--dashboard-agent-listen-port=$RAY_DASHBOARD_AGENT_LISTEN_PORT"
    "--metrics-export-port=$RAY_METRICS_EXPORT_PORT"
    "--runtime-env-agent-port=$RAY_RUNTIME_ENV_AGENT_PORT"
    "--min-worker-port=$RAY_MIN_WORKER_PORT"
    "--max-worker-port=$RAY_MAX_WORKER_PORT"
)

REWARD_FUNCTION_PATH="${OPEN_MOPD_ROOT}/training/verl/verl/utils/reward_score/instruction_following.py"
BUILD_DATA_SCRIPT="${OPEN_MOPD_ROOT}/training/scripts/rl/build_if_grpo_data.py"

echo "[if-grpo] python=$PYTHON_BIN"
echo "[if-grpo] patched_verl=${OPEN_MOPD_ROOT}/training/verl"
echo "[if-grpo] node_rank=$NODE_RANK nnodes=$NNODES visible_gpus=$VISIBLE_GPUS_PER_NODE expected_gpus=$EXPECTED_GPUS"
echo "[if-grpo] master=$MASTER_ADDR:$MASTER_PORT"
echo "[if-grpo] ray_ports dashboard=$RAY_DASHBOARD_PORT dashboard_agent_grpc=$RAY_DASHBOARD_AGENT_GRPC_PORT dashboard_agent=$RAY_DASHBOARD_AGENT_LISTEN_PORT metrics=$RAY_METRICS_EXPORT_PORT runtime=$RAY_RUNTIME_ENV_AGENT_PORT workers=$RAY_MIN_WORKER_PORT-$RAY_MAX_WORKER_PORT"
echo "[if-grpo] model=$MODEL_PATH"
echo "[if-grpo] train_jsonl=$TRAIN_JSONL"
echo "[if-grpo] train_parquet=$TRAIN_FILE"
echo "[if-grpo] val_parquet=$VAL_FILE"
echo "[if-grpo] include_ifbench_val=$INCLUDE_IFBENCH_VAL"
echo "[if-grpo] output=$OUTPUT_DIR checkpoint=$CKPT_DIR"
echo "[if-grpo] grpo train_batch=$TRAIN_BATCH_SIZE n=$N_RESP_PER_PROMPT ppo_mini=$PPO_MINI_BATCH_SIZE response_len=$MAX_RESPONSE_LENGTH"

DATA_READY=0
if [[ -f "$TRAIN_FILE" && -f "$VAL_FILE" && -f "$DATA_SUMMARY" && "$REBUILD_DATA" != 1 ]]; then
    DATA_READY=1
elif ((RUN_TRAINING)) && ((NODE_RANK == 0)); then
    mkdir -p "$DERIVED_DATA_DIR"
    rm -f "$DATA_READY_MARKER"
    BUILD_ARGS=(
        "$BUILD_DATA_SCRIPT"
        --train-jsonl "$TRAIN_JSONL"
        --ifeval-parquet "$IFEVAL_PARQUET"
        --output-train-parquet "$TRAIN_FILE"
        --output-val-parquet "$VAL_FILE"
        --summary-path "$DATA_SUMMARY"
    )
    if ((INCLUDE_IFBENCH_VAL)); then
        BUILD_ARGS+=(--ifbench-parquet "$IFBENCH_PARQUET")
    fi
    if [[ "$REBUILD_DATA" == 1 || -e "$TRAIN_FILE" || -e "$VAL_FILE" || -e "$DATA_SUMMARY" ]]; then
        BUILD_ARGS+=(--force)
    fi
    "$PYTHON_BIN" "${BUILD_ARGS[@]}"
    touch "$DATA_READY_MARKER"
    DATA_READY=1
elif ((RUN_TRAINING)); then
    DATA_START_TIME=$(date +%s)
    while [[ ! -f "$TRAIN_FILE" || ! -f "$VAL_FILE" || ! -f "$DATA_SUMMARY" || ( "$REBUILD_DATA" == 1 && ! -f "$DATA_READY_MARKER" ) ]]; do
        DATA_ELAPSED=$(( $(date +%s) - DATA_START_TIME ))
        ((DATA_ELAPSED <= DATA_WAIT_SECONDS)) || die "timed out waiting for rank 0 to prepare IF data"
        echo "[if-grpo] waiting for rank 0 data preparation ($DATA_ELAPSED/$DATA_WAIT_SECONDS)"
        sleep 5
    done
    DATA_READY=1
else
    echo "[if-grpo] dry-run: derived data missing or rebuild requested; rank 0 would run $PYTHON_BIN $BUILD_DATA_SCRIPT"
fi

if ((RUN_TRAINING)) && ((DATA_READY == 0)); then
    die "IF data is not ready"
fi

cleanup_ray() {
    if ((RUN_TRAINING)) && ((NODE_RANK == 0)); then
        "$RAY_BIN" stop || true
    fi
}
trap cleanup_ray EXIT

if ((RUN_TRAINING)); then
    if ((NODE_RANK == 0)); then
        "$RAY_BIN" start --head \
            --port="$MASTER_PORT" \
            --node-ip-address="$MASTER_ADDR" \
            --include-dashboard=false \
            "${RAY_PORT_ARGS[@]}"
    else
        until "$RAY_BIN" start --address="$MASTER_ADDR:$MASTER_PORT" "${RAY_PORT_ARGS[@]}" --block; do
            echo "[if-grpo] worker $NODE_RANK could not connect to Ray head; retrying in 3s..."
            sleep 3
        done
    fi
fi

if ((RUN_TRAINING)) && ((NODE_RANK == 0)); then
    START_TIME=$(date +%s)
    while true; do
        RAY_GPU_STATUS=$("$PYTHON_BIN" -c '
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
')
        read -r CURRENT_GPUS AVAILABLE_GPUS <<< "$RAY_GPU_STATUS"
        ELAPSED=$(( $(date +%s) - START_TIME ))
        echo "[if-grpo] [$ELAPSED/$MAX_WAIT_SECONDS] cluster=$CURRENT_GPUS/$EXPECTED_GPUS available=$AVAILABLE_GPUS/$EXPECTED_GPUS"
        if ((CURRENT_GPUS >= EXPECTED_GPUS && AVAILABLE_GPUS >= EXPECTED_GPUS)); then
            break
        fi
        if ((ELAPSED > MAX_WAIT_SECONDS)); then
            die "timed out waiting for $EXPECTED_GPUS GPUs"
        fi
        sleep 5
    done
fi

MAX_TOTAL_TOKENS=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
TRAIN_CMD=(
    "$PYTHON_BIN" -m verl.trainer.main_ppo
    "data.train_files=$TRAIN_FILE"
    "data.val_files=$VAL_FILE"
    "data.prompt_key=prompt"
    "data.train_batch_size=$TRAIN_BATCH_SIZE"
    "data.max_prompt_length=$MAX_PROMPT_LENGTH"
    "data.max_response_length=$MAX_RESPONSE_LENGTH"
    "data.filter_overlong_prompts=False"
    "data.val_filter_overlong_prompts=False"
    "data.truncation=left"
    "data.val_truncation=left"
    "data.shuffle=True"
    "data.seed=42"
    "data.return_raw_chat=True"
    "+data.apply_chat_template_kwargs.enable_thinking=False"
    "actor_rollout_ref.model.path=$MODEL_PATH"
    "actor_rollout_ref.model.use_remove_padding=True"
    "actor_rollout_ref.model.enable_gradient_checkpointing=True"
    "actor_rollout_ref.actor.policy_loss.loss_mode=vanilla"
    "actor_rollout_ref.actor.loss_agg_mode=token-mean"
    "actor_rollout_ref.actor.use_dynamic_bsz=$USE_DYNAMIC_BSZ"
    "actor_rollout_ref.actor.ppo_epochs=$PPO_EPOCHS"
    "actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU"
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOTAL_TOKENS"
    "actor_rollout_ref.actor.optim.lr=$ACTOR_LR"
    "actor_rollout_ref.actor.optim.lr_warmup_steps=$LR_WARMUP_STEPS"
    "actor_rollout_ref.actor.optim.weight_decay=$WEIGHT_DECAY"
    "actor_rollout_ref.actor.grad_clip=1.0"
    "actor_rollout_ref.actor.entropy_coeff=0"
    "actor_rollout_ref.actor.fsdp_config.fsdp_size=-1"
    "actor_rollout_ref.actor.fsdp_config.param_offload=$OFFLOAD"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OFFLOAD"
    "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$USE_DYNAMIC_BSZ"
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU"
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$MAX_TOTAL_TOKENS"
    "actor_rollout_ref.ref.fsdp_config.param_offload=$OFFLOAD"
    "actor_rollout_ref.rollout.name=vllm"
    "actor_rollout_ref.rollout.n=$N_RESP_PER_PROMPT"
    "actor_rollout_ref.rollout.temperature=1.0"
    "actor_rollout_ref.rollout.top_p=1.0"
    "actor_rollout_ref.rollout.top_k=-1"
    "actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
    "actor_rollout_ref.rollout.enable_chunked_prefill=True"
    "actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_TOTAL_TOKENS"
    "actor_rollout_ref.rollout.max_model_len=$MAX_TOTAL_TOKENS"
    "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$USE_DYNAMIC_BSZ"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU"
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$MAX_TOTAL_TOKENS"
    "actor_rollout_ref.rollout.val_kwargs.temperature=$VAL_TEMPERATURE"
    "actor_rollout_ref.rollout.val_kwargs.top_p=$VAL_TOP_P"
    "actor_rollout_ref.rollout.val_kwargs.top_k=-1"
    "+actor_rollout_ref.rollout.val_kwargs.max_tokens=$MAX_RESPONSE_LENGTH"
    "actor_rollout_ref.rollout.val_kwargs.do_sample=True"
    "actor_rollout_ref.rollout.val_kwargs.n=1"
    "algorithm.adv_estimator=grpo"
    "algorithm.norm_adv_by_std_in_grpo=True"
    "algorithm.use_kl_in_reward=False"
    "algorithm.kl_ctrl.kl_coef=0.0"
    "actor_rollout_ref.actor.use_kl_loss=False"
    "actor_rollout_ref.actor.kl_loss_coef=0.0"
    "reward_model.enable=False"
    "reward_model.reward_manager=dapo"
    "custom_reward_function.path=$REWARD_FUNCTION_PATH"
    "custom_reward_function.name=reward_func"
    "+custom_reward_function.reward_kwargs.scoring_mode=$REWARD_SCORING_MODE"
    "+custom_reward_function.reward_kwargs.auto_official_eval_for_aligned_val=True"
    "trainer.n_gpus_per_node=$VISIBLE_GPUS_PER_NODE"
    "trainer.nnodes=$NNODES"
    "trainer.total_epochs=$TOTAL_EPOCHS"
    "trainer.total_training_steps=$TOTAL_TRAINING_STEPS"
    "trainer.default_local_dir=$CKPT_DIR"
    "trainer.project_name=$PROJECT_NAME"
    "trainer.experiment_name=$EXPERIMENT_NAME"
    "trainer.logger=['console','tensorboard']"
    "trainer.val_before_train=False"
    "trainer.test_freq=$TEST_FREQ"
    "trainer.save_freq=$SAVE_FREQ"
    "trainer.resume_mode=auto"
)
TRAIN_CMD+=("+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=$PYTHONPATH")
TRAIN_CMD+=("+ray_kwargs.ray_init.runtime_env.env_vars.TENSORBOARD_DIR=$TENSORBOARD_DIR")
if ((INCLUDE_IFBENCH_VAL)); then
    TRAIN_CMD+=("+ray_kwargs.ray_init.runtime_env.env_vars.OPENOPD_IFBENCH_REPO=$REWARD_IFBENCH_REPO")
fi
TRAIN_CMD+=("${EXTRA_OVERRIDES[@]}")

printf '[if-grpo] command:'
printf ' %q' "${TRAIN_CMD[@]}"
printf '\n'

if ((RUN_TRAINING)) && ((NODE_RANK == 0)); then
    mkdir -p "$OUTPUT_DIR" "$CKPT_DIR"
    # Keep Hydra's relative run artifacts under the project root as well.
    cd "$OPEN_MOPD_ROOT"
    "${TRAIN_CMD[@]}"
else
    echo "[if-grpo] dry-run only; omit --dry-run (or pass --run) on all nodes to execute"
fi
