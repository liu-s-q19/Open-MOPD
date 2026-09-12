#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"

usage() {
    cat <<'USAGE'
Run multi-teacher on-policy distillation on local paths.

Provide at least two teacher directories with repeated --teacher options or
TEACHER_MODEL_PATHS (comma-separated). TEACHER_DOMAINS can name them in the
same order; its default is math,code,if.

The command is printed without execution unless --run is supplied.
USAGE
    local_common_usage
}

local_init
local_parse_common "$@"
local_scope_output mt_opd
if ((LOCAL_HELP)); then
    usage
    exit 0
fi

if ((${#LOCAL_TEACHER_PATHS[@]} == 0)) && [[ -n "${TEACHER_MODEL_PATHS:-}" ]]; then
    IFS=',' read -r -a LOCAL_TEACHER_PATHS <<< "${TEACHER_MODEL_PATHS}"
fi
if ((${#LOCAL_TEACHER_PATHS[@]} == 0)) && [[ -n "${REWARD_MODEL_PATH:-}" ]]; then
    LOCAL_TEACHER_PATHS+=("${REWARD_MODEL_PATH}")
fi
if ((${#LOCAL_TEACHER_PATHS[@]} == 0)) && [[ -n "${TEACHER_PATH:-}" ]]; then
    LOCAL_TEACHER_PATHS+=("${TEACHER_PATH}")
fi

# FSDP requires the global batch to be divisible by the number of local GPUs.
# Using LOCAL_GPUS as the default keeps the local smoke launcher valid on both
# one- and two-GPU machines while still allowing an explicit override.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${LOCAL_GPUS}}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
N_RESPONSES="${N_RESPONSES:-1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
# Ray otherwise inherits every CPU visible in the host (this machine exposes
# 192), which can make a tiny 2-GPU smoke test spawn too many workers and hang
# before the trainer starts.  Keep this local-only; the adaptive multi-node
# launcher configures Ray through the scheduler instead.
RAY_NUM_CPUS="${RAY_NUM_CPUS:-8}"
RAY_NODE_IP="${RAY_NODE_IP:-127.0.0.1}"
REWARD_MODEL_MICRO_BATCH_SIZE_PER_GPU="${REWARD_MODEL_MICRO_BATCH_SIZE_PER_GPU:-1}"
# Some Kubernetes GPU containers expose CUDA IPC/P2P incompletely.  The
# local launcher defaults to the portable NCCL path; override with 0 when the
# local node is known to support P2P.  The multi-node launcher is unchanged.
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

local_export_runtime
cd "$LOCAL_TRAINING_DIR"
if [[ "$LOCAL_RUN" == 1 ]]; then
    local_validate_common 1 1 1
    ((${#LOCAL_TEACHER_PATHS[@]} >= 2)) || local_die "at least two --teacher paths are required"
    for teacher_path in "${LOCAL_TEACHER_PATHS[@]}"; do
        local_require_local_path teacher "$teacher_path"
        local_require_path teacher "$teacher_path"
    done
    local_prepare_output
fi

IFS=',' read -r -a domain_array <<< "${LOCAL_DOMAINS}"
if ((${#LOCAL_TEACHER_PATHS[@]} > 0)) && ((${#domain_array[@]} != ${#LOCAL_TEACHER_PATHS[@]})); then
    local_die "teacher domain count (${#domain_array[@]}) must match teacher path count (${#LOCAL_TEACHER_PATHS[@]})"
fi

domain_list="["
for i in "${!domain_array[@]}"; do
    [[ "$i" == 0 ]] || domain_list+=","
    domain_list+="${domain_array[$i]}"
done
domain_list+="]"

first_teacher="${LOCAL_TEACHER_PATHS[0]:-${REWARD_MODEL_PATH:-${LOCAL_MODEL_PATH}}}"
teacher_count="${#LOCAL_TEACHER_PATHS[@]}"
if ((teacher_count == 0)); then
    # Keep a readable dry-run command; --run still requires two real teachers.
    teacher_count=1
fi
cmd=(
    "$LOCAL_PYTHON_BIN" -m verl.trainer.main_ppo
    "algorithm.adv_estimator=token_reward_direct"
    "data.train_files=${LOCAL_TRAIN_FILE}"
    "data.val_files=${LOCAL_VAL_FILE}"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH}"
    "data.filter_overlong_prompts=True"
    "data.truncation=error"
    "actor_rollout_ref.model.path=${LOCAL_MODEL_PATH}"
    "actor_rollout_ref.rollout.name=vllm"
    "+actor_rollout_ref.rollout.reward_mode=mt_opd"
    "actor_rollout_ref.rollout.n=${N_RESPONSES}"
    "+actor_rollout_ref.rollout.log_prob_top_k=256"
    "actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))"
    "reward_model.enable=True"
    "reward_model.model.path=${first_teacher}"
    "reward_model.model.input_tokenizer=null"
    "reward_model.micro_batch_size_per_gpu=${REWARD_MODEL_MICRO_BATCH_SIZE_PER_GPU}"
    "+reward_model.reward_kwargs.compute_true_reward=False"
    "+mt_opd.teacher_domains=${domain_list}"
    "+mt_opd.n_additional_teachers=$((teacher_count - 1))"
    "trainer.n_gpus_per_node=${LOCAL_GPUS}"
    "trainer.nnodes=${LOCAL_NODES}"
    "trainer.total_epochs=${TOTAL_EPOCHS}"
    "trainer.default_local_dir=${LOCAL_CHECKPOINT_DIR}"
    "trainer.project_name=${PROJECT_NAME:-Qwen3-4B}"
    "trainer.experiment_name=${EXPERIMENT_NAME:-vanilla_mopd_local}"
    "trainer.logger=['console']"
    "ray_kwargs.ray_init.num_cpus=${RAY_NUM_CPUS}"
    "+ray_kwargs.ray_init._node_ip_address=${RAY_NODE_IP}"
    "+ray_kwargs.ray_init.include_dashboard=False"
    "+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_P2P_DISABLE=\"${NCCL_P2P_DISABLE}\""
)

for ((i = 1; i < ${#LOCAL_TEACHER_PATHS[@]}; i++)); do
    teacher_index="$i"
    teacher_path="${LOCAL_TEACHER_PATHS[$i]}"
    cmd+=(
        "+mt_reward_model_${teacher_index}.enable=True"
        "+mt_reward_model_${teacher_index}.model.path=${teacher_path}"
        "+mt_reward_model_${teacher_index}.model.input_tokenizer=null"
        "+mt_reward_model_${teacher_index}.model.use_remove_padding=True"
        "+mt_reward_model_${teacher_index}.model.fsdp_config.param_offload=True"
    )
done
if ((${#LOCAL_EXTRA_OVERRIDES[@]})); then
    cmd+=("${LOCAL_EXTRA_OVERRIDES[@]}")
fi
if ((${#LOCAL_REMAINING[@]})); then
    cmd+=("${LOCAL_REMAINING[@]}")
fi

local_run_command "${cmd[@]}"
