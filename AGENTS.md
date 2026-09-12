# Open-MOPD Agent Guide

## 项目目标

Open-MOPD 是一个多教师 on-policy distillation 研究基座。本项目当前的研究主线是：在固定 student、teacher、rollout 和优化预算的条件下，比较不同的 reference-construction operator，并研究 teacher disagreement 与 teacher reliability 是否能够解释 operator 的相对表现。

研究计划的唯一主文档是：

```text
task/Multi_Teacher_OPD_Reference_Construction_研究计划.md
```

开始新的研究任务前，先阅读与当前阶段相关的章节，尤其是第 6--10 节和第 15 节。不要把“多教师路由”直接当作“多教师 reference construction”：本项目需要明确区分 matched route、arithmetic mixture、geometric consensus、oracle/verified baseline 和 adaptive selector。

## 当前阶段边界

按照研究计划，优先顺序是：

1. Phase 0：用 synthetic distributions 验证 operator 和数值公式；
2. Phase 1：构建 cross-domain teacher competence matrix；
3. Phase 2：对固定 student rollout 做 teacher fan-out 和 token-distribution profiling；
4. 只有前面阶段显示出稳定、可解释的差异后，才进入 Phase 3 的 controlled operator training。

本机只有两张 GPU，当前只做短时、缩小数据、可中断的 smoke test 或 profiling。不要默认启动完整训练、长时间 rollout、大规模 teacher fan-out 或正式论文实验。完整训练由用户单独提出并指定资源、数据规模和输出目录。

## 仓库结构与代码边界

- `training/verl/`：本项目使用的 patched verl 实现；本项目运行时优先加载这里的源码。
- `training/scripts/`：数据准备和训练辅助脚本。
- `scripts/local/`：本地 SFT、RL、OPD、MT-OPD 和评测 launcher。
- `evals/`：离线 rollout、verifier 和结果聚合。
- `experiments/`：toy verification、数据处理、分析脚本和 focused tests。
- `task/`：研究计划与研究记录；除非用户要求，不要覆盖或重写已有计划。

第一版研究代码应保持模块边界清晰。推荐把 reference construction 相关逻辑放在独立模块中，例如：

```text
reference_construction/
├── operators.py       # route / arithmetic mixture / geometric consensus
├── disagreement.py    # KL、JS、incompatibility 和 bucket statistics
├── verifier.py        # answer-level correctness
├── selector.py        # 后续的 adaptive operator selector
├── cache.py           # teacher output / prefix cache
└── diagnostics.py     # 分桶统计和可视化
```

训练循环应尽量只依赖统一接口：

```text
teacher_outputs -> reference_operator -> p_ref -> OPD_loss
```

第一版不要同时引入 teacher compression、KV-cache sharing、新并行系统或大型 selector 网络；这些会改变资源条件并掩盖 reference effect。

## Python 环境与 import 规则

共享环境为：

```text
/home/luban/miniconda3/envs/verl
```

已经验证的核心版本是 Python 3.12、PyTorch 2.8.0、vLLM 0.11.0、SGLang 0.5.2、Ray 2.51.1、Transformers 4.56.1 和 TensorDict 0.10.0。Open-MOPD 与已有 OPD 项目共享该环境时，默认不要升级或重新 pin Torch、vLLM、SGLang、Ray、Transformers、Numpy 等核心包。安装新依赖前先检查版本和 `pip check`，避免破坏已有 OPD。

本项目的 patched verl 必须排在其他项目的 verl 之前。优先使用 local launcher，因为 `scripts/local/common.sh` 会自动把本仓库的 `training/verl` 放到 `PYTHONPATH` 首位，并打印实际命令。手动运行时使用：

```bash
cd /nfs/dataset-ofs-nmgvoyagermodel-prediction/rl_lab/liushiqi/opd/Open-MOPD
source /home/luban/miniconda3/etc/profile.d/conda.sh
conda activate verl
export PYTHONPATH="$PWD/training/verl:$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

确认加载路径：

```bash
python -c 'import verl; print(verl.__file__)'
```

输出应指向本仓库的 `training/verl/verl/__init__.py`。不要依赖环境中已有的 editable `verl`，因为它可能指向其他项目，例如 `/nfs/.../llm/verl-main`。

## Launcher 使用规范

除用户明确指定的自适应 vanilla MT-OPD launcher 外，所有本地 launcher 默认只打印命令，不执行任务。`scripts/local/mopd/vanilla_mopd.sh` 默认执行训练，使用 `--dry-run` 检查路径、GPU 数量、teacher/domain 数量和 `PYTHONPATH`。

常用入口：

```bash
bash scripts/local/sft.sh --help
bash scripts/local/rl.sh --help
bash scripts/local/opd.sh --help
bash scripts/local/mopd/vanilla_mopd_local.sh --help
bash scripts/local/eval.sh --help
```

多教师 launcher 必须使 teacher 路径和 domain 列表一一对应，例如：

```bash
bash scripts/local/mopd/vanilla_mopd_local.sh \
  --model /path/to/mixsft \
  --teacher /path/to/math-teacher \
  --teacher /path/to/code-teacher \
  --teacher /path/to/if-teacher \
  --domains math,code,if \
  --train /path/to/train.parquet \
  --val /path/to/val.parquet
```

研究实验中必须记录 launcher 完整输出，尤其是：模型路径、teacher 顺序、domain 顺序、数据文件、GPU 配置、seed、optimizer、token budget、reference operator、cache 配置和输出目录。

## 实验与验证要求

### 修改前

- 先定位现有入口和测试，优先复用 Open-MOPD 自带实现。
- 明确本次修改属于 toy、profiling、controlled comparison 还是正式训练。
- 不要把一个完整训练命令当作验证；先提供最小可运行案例。

### 修改后

至少执行与改动范围匹配的验证：

```bash
# launcher 和实验工具的 focused tests
PYTHONPATH=training/verl:. pytest -q experiments/tests/test_local_shell_entrypoints.py

# Python 文件语法检查
PYTHONPATH=training/verl:. python -m compileall -q training/verl/verl experiments scripts

# shell 语法检查
bash -n scripts/local/*.sh training/install_requirements.sh
```

涉及 operator 的改动还必须包含 toy 数值测试，验证：

- arithmetic mixture 使用 log-sum-exp，不能直接乘概率或在 log-space 中错误相加；
- geometric consensus 使用加权 log-probability，并明确归一化项是否需要；
- hard route、mixture、geometric 三者使用相同 prefix、相同 teacher outputs 和相同 mask；
- sampled-token 近似与 top-k/full-vocabulary profiling 的适用边界没有混淆。

涉及 GPU 的测试应显式检查：

```bash
nvidia-smi -L
python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'
```

如果当前执行容器看不到 `/dev/nvidia*`，应将其报告为执行环境隔离，不要修改 Python 包来“修复”它。

## 数据、缓存与输出

- 不要把模型权重、parquet 数据、rollout、teacher logits、checkpoint、wandb/SwanLab 日志或大规模 profiling cache 放入 Git。
- 使用用户指定的持久化输出目录；临时验证可以使用 `/tmp` 下的目录。
- 修改 `.gitignore` 前先检查已有规则和 `git status`，不要误忽略源码、研究计划或测试。
- 不要删除用户已有的未跟踪文件，特别是 `task/` 下的研究文档。
- 每个实验应保留可重建信息：命令、环境版本、git commit、数据版本、模型/teacher 路径、seed 和关键配置。

## 研究公平性与可解释性

比较不同 reference operator 时，除 operator 外尽量固定：

- student checkpoint 和 rollout；
- teacher checkpoint、teacher 顺序和 tokenizer/vocabulary；
- prompt/domain composition；
- teacher query 数量和 cache 策略；
- batch size、sequence length、token budget、optimizer 和学习率 schedule；
- reward refresh、验证集和随机种子。

报告结果时不要只给 overall average。至少同时报告各 domain、weakest-domain、domain gap、teacher query cost、wall-clock time 和 selector regret。不要把 geometric pool 直接表述为真实的 product-of-experts；在本项目中它是 weighted reverse-KL 对应的 consensus reference，具体语义必须与实验假设一致。

## Git 与协作

- 保留用户已有修改；提交前检查 `git diff` 和 `git status`。
- 一个提交只包含一个逻辑主题，避免把依赖升级、算法改动和无关格式化混在一起。
- 不要自动提交、推送、创建 PR 或修改远程分支，除非用户明确要求。
- 完成任务时说明改动文件、验证命令、已知失败及其是否与本次修改相关。
