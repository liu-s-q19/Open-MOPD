# Local shell launchers

These launchers expose the common training and evaluation entry points as
plain shell commands. They use paths already present on the machine and do not
fetch or publish artifacts. Most launchers print their command and exit without
running it by default; the adaptive MOPD launchers run by default. Use
`--dry-run` for a configuration-only check.

Examples:

```bash
bash scripts/local/opd.sh \
  --model /data/models/student \
  --teacher /data/models/teacher \
  --train /data/train.parquet \
  --val /data/valid.parquet \
  --gpus 8

bash scripts/local/opd.sh --model /data/models/student \
  --teacher /data/models/teacher --train /data/train.parquet \
  --val /data/valid.parquet --gpus 8 --run

bash scripts/local/mopd/vanilla_mopd_local.sh \
  --model /data/models/student \
  --teacher /data/models/math \
  --teacher /data/models/code \
  --teacher /data/models/if \
  --domains math,code,if --train /data/train.parquet \
  --val /data/valid.parquet --gpus 8

# Adaptive multi-node default MT-OPD recipe. Start this on every allocated node.
bash scripts/local/mopd/default_mopd.sh --dry-run
# After checking the output, run on every allocated node. The scheduler
# supplies DISTRIBUTED_NODE_COUNT, DISTRIBUTED_NODE_RANK, and the master host.
bash scripts/local/mopd/default_mopd.sh

# Matched vanilla baseline for Open-MOPD ablations: same 32/16 budget settings,
# but without Open-MOPD's share/gap/reward-refresh mechanisms.
bash scripts/local/mopd/vanilla_mopd.sh --dry-run

# Qwen3-4B with Open-MOPD budget balancing and reward refresh. The default IF
# teacher is the current Math-teacher fallback.
bash scripts/local/mopd/open_mopd.sh --dry-run

bash scripts/local/eval.sh --model /data/models/student \
  --input /data/test.parquet --output /data/eval-output
```

Environment variables mirror the option names (`MODEL_PATH`, `TRAIN_FILE`,
`VAL_FILE`, `OUTPUT_DIR`, `CHECKPOINT_DIR`, `GPUS`, `NODES`, `PYTHON_BIN`, and
`TORCHRUN_BIN`). For teachers, use `REWARD_MODEL_PATH` for OPD or a
comma-separated `TEACHER_MODEL_PATHS` plus `TEACHER_DOMAINS` for MT-OPD.
Additional trainer overrides may be passed after `--`.

Project-local runtime outputs are written under `outputs/` and
`tensorboard_log/`. The vanilla multi-node launcher reads
`DISTRIBUTED_NODE_COUNT`, `DISTRIBUTED_NODE_RANK`, and
`DISTRIBUTED_MASTER_HOSTS`, while explicit `NNODES`, `NODE_RANK`, and
`CUDA_VISIBLE_DEVICES` take precedence.

`mopd/open_mopd.sh` keeps the Qwen3-4B recipe from the vanilla launcher but
enables Open-MOPD's token-share balancing, gap-following allocation, and
student-dependent reward refresh. It is an Open-MOPD-style Qwen3 run, not the
official SmolLM3 exact reproduction. Set `IF_TEACHER_PATH` or pass
`--if-teacher` when an independent Qwen3 IF teacher is available; otherwise the
Math teacher is used as the explicit fallback.

`mopd/vanilla_mopd.sh` is the matched vanilla control for this comparison. It
uses `ppo_mini_batch_size=32` and `log_prob_top_k=16`, while leaving the
Open-MOPD mechanisms disabled. `mopd/default_mopd.sh` remains the legacy
vanilla recipe with its original 128/256 settings.

Every launcher prepends this repository's `training/verl` directory to
`PYTHONPATH` and prints the effective value with the generated command. Any
existing `PYTHONPATH` entries are preserved after the repository path, so a
different project's local `verl` is not selected accidentally.
