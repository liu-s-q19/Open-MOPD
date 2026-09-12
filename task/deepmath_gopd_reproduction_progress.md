# Reproduction progress

## 2026-09-11 — MT-OPD teacher tokenizer switching fix

- Failure: teacher RM scoring entered `_switch_chat_template_token_level` and raised `KeyError: raw_prompt` because the primary reward model inherited `input_tokenizer=${actor_rollout_ref.model.path}` while the RL parquet loader was not asked to expose a top-level raw chat list.
- Fix: both vanilla MOPD launchers set `reward_model.model.input_tokenizer=null`. Student and all configured teachers use the same Qwen3 tokenizer family, so re-tokenizing with a second chat template is unnecessary for this run.
- Verification: launcher dry-run includes the null override, and launcher, reward, compile, and shell checks pass.

## 2026-09-11 — Python 3.12 code validation fallback

- Failure: mixed validation reached the code scorer and failed importing `pyext.RuntimeModule`; `pyext` is absent from the shared Python 3.12 environment and the repository notes that the upstream package is not Python 3.12-compatible.
- Fix: `default_compute_score` keeps the legacy `prime_code` path when `pyext` is available, otherwise falls back to the vendored Python-3.12-compatible `rllm_code_reward`; `codecontests` is normalized to rLLM's `code_contests` identifier. The rLLM adapter now also converts the validation parquet's `{inputs, outputs}` dictionary shape to the list-of-records shape required by its Codeforces/LCB scorer.
- Verification: all 1,024 rows of the actual code validation parquet parse and normalize for `apps`, `codecontests`, `codeforces`, and `taco`; fallback code scoring, reward dispatch, existing rLLM code tests, MT-OPD routing tests, launcher tests, compileall, and shell checks passed.

## 2026-09-11 — MT-OPD validation reward dispatch fix

- Failure: validation stopped in `default_compute_score` because the AIME parquet uses `data_source=math_dapo_boxed`, which was not registered.
- Fix: route `math_dapo_boxed` to the boxed-answer `math_reward` scorer. It must not use `math_dapo`, whose default parser expects an `Answer:` line.
- Verification: dispatch unit tests, MT-OPD routing tests, local launcher tests, compileall, shell syntax, and diff checks passed.

## 2026-09-09 — IF-GRPO launcher

- Reference script: `/nfs/dataset-ofs-nmgvoyagermodel-prediction/rl_lab/liushiqi/llm/verl-main/scripts/opd_rl/grpo_4b_box_opd_data_justrl.sh`.
- Semantic delta: IF-only GRPO on Open-MOPD's patched `verl.trainer.main_ppo`, using Qwen3-4B, the Nemotron instruction-following train JSONL, and aligned IFEval/IFBench validation data. It does not use OPD reference construction or teacher logits.
- Multi-node behavior follows the reference Ray component ports, worker retry loop, 32-GPU availability wait, and rank-0 cleanup. Python defaults to `/home/luban/miniconda3/envs/verl/bin/python`, with the repository patched verl first in `PYTHONPATH`.
- Verification: launcher shell syntax, dry-run command rendering, Hydra config composition, data schema/RLHFDataset smoke test, IF reward smoke test, focused IF reward tests, compileall, and patched-verl import check passed.
- Unresolved cluster checks: run the same launcher on all four nodes with `DISTRIBUTED_MASTER_HOSTS`, ranks `0..3`, and shared NFS paths; configure `OPENOPD_IFBENCH_REPO` before `--run` so IFBench official validation can load its checkout.

## 2026-09-09 — IF-GRPO default execution override

- User-requested semantic delta: `scripts/local/if_grpo_32gpu.sh` now runs Ray/training by default; `--dry-run` remains available for configuration checks. This intentionally overrides the repository's general default-dry-run convention for this launcher.
- Verification: shell syntax check and explicit `--dry-run` command rendering are required after this change; no four-node training was started locally.
- Risk: invoking the script without per-node `DISTRIBUTED_NODE_RANK`, a valid `DISTRIBUTED_MASTER_HOSTS`, and `OPENOPD_IFBENCH_REPO` will fail or attach incorrectly; use `--dry-run` first when checking a new cluster allocation.

## 2026-09-09 — IFBench checkout fallback

- Fix: when `OPENOPD_IFBENCH_REPO` is unavailable, the launcher now builds and uses an IFEval-only validation parquet instead of failing before Ray. Setting `OPENOPD_IFBENCH_REPO` or passing `--include-ifbench` enables the combined official IFEval+IFBench validation path.
- Verification target: run the data helper with the optional IFBench input, shell syntax, dry-run, and reward smoke checks; no cluster training is started locally.

## 2026-09-09 — Adaptive node/GPU resources

- Fix: aligned resource selection with the reference STAPO common launcher. `NNODES` now defaults to `DISTRIBUTED_NODE_COUNT` or 1, `CUDA_VISIBLE_DEVICES` is honored when supplied and otherwise detected with `nvidia-smi`, and the expected GPU count is computed as `NNODES × visible_gpus_per_node` without a hard-coded 32-GPU check.
- Compatibility: master host falls back through `MASTER_ADDR`, `DISTRIBUTED_MASTER_HOSTS`, and `VC_MASTER_HOSTS`; master port also accepts `LUBAN_AVAILABLE_PORT_0` and `DISTRIBUTED_PYTORCH_PORT`.
- Verification target: test single-node dry-run on the local visible GPUs and multi-node dry-run with explicit `NNODES`/rank; no training is started locally.

## 2026-09-10 — vLLM startup memory headroom

- Fix: lowered the default `ROLLOUT_GPU_MEMORY_UTILIZATION` from `0.6` to `0.4`. The failed run had only 42.1 GiB free after actor initialization while vLLM required 57.05 GiB at 0.6, so vLLM aborted before rollout initialization.
- The value remains environment-overridable for nodes with more or less resident GPU memory; verify the reported free memory on every node before raising it.

## 2026-09-10 — Ray reward dependency preflight

- Fix: explicitly add `training/third_party/verifiable-instructions` to launcher `PYTHONPATH` and pass the resulting `PYTHONPATH` through `ray_kwargs.ray_init.runtime_env.env_vars` so remote reward workers inherit it.
- Preflight: every launcher node now imports the registry and executes one local IF reward calculation before starting Ray; the reward loader also preserves the original import exception in its error message.

## 2026-09-10 — IF DAPO reward manager optional overlong config

- Parent launcher: `scripts/local/if_grpo_32gpu.sh`, using `reward_model.reward_manager=dapo` with no overlong penalty configuration.
- Semantic delta: fixed `training/verl/verl/workers/reward_manager/dapo.py` to treat a missing `overlong_buffer_cfg` as the normal no-penalty path, matching the existing Naive manager behavior. No GRPO hyperparameters or reward semantics were changed.
- Regression coverage: added `training/verl/tests/test_dapo_reward_manager_optional_overlong.py` with a minimal reward batch and `overlong_buffer_cfg=None`.
- Verification: targeted pytest, Python compileall, direct nullable-config scan, launcher shell syntax, and IF launcher dry-run.
- Unresolved risk: a cluster run still needs the same patched `PYTHONPATH` and IF verifier dependency propagation on every Ray worker; this fix only addresses the missing optional overlong configuration.

## 2026-09-10 — IF verifier kwargs compatibility

- Root cause: Nemotron IF-RL stores `count:count_increment_word` `keyword1`/`keyword2` as singleton lists, while the official verifier expects strings and calls `.strip()`.
- Fix: normalize those two fields in the runtime scorer and both IF data builders; keep intentionally list-valued fields such as `keywords:forbidden_words` unchanged. Verifier construction/check exceptions now fail closed and are reported as checker errors.
- Data audit: checked 46,391 train rows / 110,521 constraints and 541 validation rows / 834 constraints from the current parquet files; all verifier descriptions constructed successfully after normalization.
- Verification: IF reward tests (10 passed), DAPO optional-overlong regression test (1 passed), builder preflight, Python compileall, and launcher shell syntax.
- Unresolved test infrastructure: seven pre-existing GRM tests still expect a fake `requests.Session` without the production `mount()` method and use an unreachable fake endpoint; unrelated to IF verifier kwargs and not changed here.

## 2026-09-10 — IF validation batch shape mismatch

- Root cause: vLLM SPMD validation used `val_kwargs.get("max_tokens", response_length)`. Because `SamplingConfig.max_tokens` exists with value `None`, the fallback was not applied; each worker therefore padded validation responses to a different local maximum, causing `DataProto.concat` to fail with widths 16,253 and 16,278.
- Fix: added a shared `_resolve_validation_max_tokens` helper with an explicit `None` fallback, applied it to vLLM sampling and padding; the IF launcher now explicitly sets `actor_rollout_ref.rollout.val_kwargs.max_tokens=15360`.
- Backend audit: SGLang already uses a `None`-safe fallback and pads to the configured response length; HF generation already uses the configured response length. No matching nullable fallback remains in the active vLLM path.
- Verification: 25 focused rollout/validation/IF tests passed, Python compileall passed, launcher shell syntax passed, and IF dry-run printed `val_kwargs.max_tokens=15360`.
- Cluster follow-up: restart the failed Ray job so every rollout worker loads the updated NFS source; checkpoint auto-resume can then continue from the latest saved checkpoint.

## 2026-09-10 — IF launcher Ray wait-loop shell parsing

- Failure: after Ray reached `32/32`, the task reported a shell syntax error near the closing command substitution in the rank-0 GPU polling block; subsequent Ray SIGTERM messages were cleanup after the launcher aborted.
- Fix: replaced the nested heredoc inside `RAY_GPU_STATUS=$(...)` with an equivalent `PYTHON_BIN -c` command, keeping the same Ray resource query and retry/wait behavior.
- Verification: `bash -n` on the launcher and proxy scripts, IF dry-run, `git diff --check`, and no remaining heredoc/command-substitution nesting in the launcher.
- Cluster follow-up: use the current NFS launcher copy and restart the failed task; do not reuse the previously submitted script payload if the platform snapshots script content at submission time.

## 2026-09-10 — Hydra struct override for validation max tokens

- Failure: Hydra rejected `actor_rollout_ref.rollout.val_kwargs.max_tokens=15360` because the base YAML uses a structured mapping without that key, even though `SamplingConfig` supports the field.
- Fix: changed the IF launcher override to `+actor_rollout_ref.rollout.val_kwargs.max_tokens=15360`, which is the correct Hydra syntax for adding this omitted structured key.
- Verification: real Hydra `compose()` produced `val_kwargs.max_tokens=15360`, launcher `bash -n` and `git diff --check` passed; no training was started.

## 2026-09-10 — Adaptive vanilla MT-OPD launcher

- Parent/reference launcher: `scripts/local/if_grpo_32gpu.sh`; implementation target: `scripts/local/mopd/vanilla_mopd.sh`.
- Semantic delta: added an adaptive multi-node Ray wrapper for routed vanilla MT-OPD. `NNODES`/`NODE_RANK` use explicit values or `DISTRIBUTED_NODE_COUNT`/`DISTRIBUTED_NODE_RANK`; visible GPUs use `CUDA_VISIBLE_DEVICES` or `nvidia-smi`; no hard-coded 4-node/32-GPU requirement.
- Temporary teacher mapping: `[Math, Code, Math]` for `[math, code, if]`. The launcher uses fixed 1:1:1 sampling, non-thinking Qwen3-4B, standard routed per-token reverse-KL/G-OPD, and does not enable D³ scheduling or reference-operator comparison.
- Default recipe: train batch 128, rollout n=4, 256 steps, lr 1e-6, weight decay 0.1, prompt/response limits 2048/8192, log-prob top-k 256, save/test every 16 steps, and checkpoint `resume_mode=auto`.
- Data contract: train and validation parquet files are prepared before launch; the launcher only validates them and never races to write shared data from multiple nodes.
- Verification target: shell syntax, adaptive single-/multi-node dry-runs, local launcher focused tests, patched-verl import path, and compileall. No local training is started. The adaptive launcher now runs by default when invoked; use `--dry-run` for checks.
- Unresolved cluster checks: on the allocated cluster, start the launcher on every node with scheduler-provided rank/master variables, verify all expected GPUs become available in Ray, and confirm the shared NFS data/checkpoint paths.

## 2026-09-10 — Vanilla MT-OPD paths and data assets

- Multi-teacher launchers moved to `scripts/local/mopd/vanilla_mopd_local.sh` and `scripts/local/mopd/vanilla_mopd.sh` so the experiment family and vanilla semantics are explicit.
- Runtime output defaults now use the project-root `outputs/` and `tensorboard_log/` directories. Existing untracked `scripts/local/outputs/` and `scripts/local/tensorboard_log/` artifacts were moved there without overwriting a pre-existing destination.
- Prepared the balanced train parquet at `/nfs/dataset-ofs-nmgvoyagermodel-prediction/rl_lab/liushiqi/llm/dataset/open_mopd/vanilla_mopd_4k/train.parquet`: 12,000 rows, exactly 4,000 per Math/Code/IF domain, deterministic seed `20260718`, with source manifest and project-side validation manifest.
- Validation inputs remain the existing AIME24 (960), Code (1,024), and IFEval-only (541) parquets; no IFBench evaluator checkout was assumed.
- Added root `outputs/` and `tensorboard_log/` to `.gitignore`; no model weights, parquet data, or checkpoints were added to Git.
