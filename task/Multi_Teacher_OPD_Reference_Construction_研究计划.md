# Multi-Teacher OPD 的 Reference Construction 研究计划

## 1. 研究定位

### 暂定题目

**Routing, Mixing, or Consensus? Reference Construction for Multi-Teacher On-Policy Distillation**

中文题目可暂定为：

**多教师 On-Policy Distillation 中的 Reference Construction：路由、混合与共识**

### 一句话摘要

本项目研究：当多个专门化教师同时面对同一个 student-generated prefix 时，学生究竟应该逼近哪一个 reference distribution，以及 hard routing、arithmetic mixture 和 geometric consensus 分别在什么教师关系下合理。

项目不把重点放在新增一个简单的 confidence-aware 或 gap-aware 权重，而是先建立一个可解释的 reference-construction taxonomy，再验证教师分歧和教师可靠性是否能够预测最佳 aggregation operator，最后提出轻量的自适应 operator selector。

---

## 2. 背景与研究动机

### 2.1 MOPD 的基本问题

On-Policy Distillation 的核心优势是：学生在自己的 rollout 上接受教师监督，从而减少 off-policy exposure bias，并在学生实际访问到的状态上获得 token-level 反馈。

对某一个 student rollout prefix，记为

$$
s_t=(x,y_{<t}),
$$

学生和第 $k$ 个教师的 next-token distributions 分别为

$$
q_\theta(v\mid s_t)=\pi_\theta(v\mid s_t),
\qquad
p_k(v\mid s_t)=\pi_{T_k}(v\mid s_t).
$$

单教师 OPD 可以写成

$$
\min_\theta \;D_{\mathrm{KL}}\bigl(q_\theta(\cdot\mid s_t)\,\|\,p_k(\cdot\mid s_t)\bigr).
$$

此时 reference policy 明确就是 $p_k$。

多教师 MOPD 的基础问题则是：当存在

$$
p_1, p_2, \ldots, p_K
$$

时，学生的 target 到底应该是其中一个教师、教师分布的 arithmetic mixture，还是教师分布的 geometric consensus？

### 2.2 现有路线解决了哪些问题

已有 MOPD 工作将多个领域 RL teacher 的能力整合到一个学生中；Open-MOPD 进一步说明，在 oracle routing 下，token budget、序列长度差异、动态收敛和 reward staleness 都会显著影响能力整合结果。

近期工作还分别研究了：

- sample-level answer-verified teacher eligibility；
- token-level confidence-aware teacher arbitration；
- gap-aware sample selection；
- dynamic domain scheduling；
- token-level multi-teacher weighted distillation。

因此，单独提出“动态 teacher weighting”或“confidence-aware routing”已经不够形成清晰的研究空白。本项目将问题前移到 reference construction：

> 在多教师 OPD 中，教师集合究竟代表多个候选答案、多个必须同时满足的约束，还是一个需要被路由的专家集合？

### 2.3 研究动机

目前常见的多教师损失形式容易被误解为“平均听取多个老师”。然而，若使用加权 reverse-KL，实际优化的并不是 arithmetic average，而是 normalized geometric pool。这会导致完全不同的行为：

- arithmetic mixture 保留教师支持的并集，更适合 alternative hypotheses；
- geometric pool 强调教师支持的交集，更适合 consensus constraints；
- hard route 假设当前只有一个教师是合适的 reference；
- best-teacher selection 假设只要找到一个可靠教师即可。

这些方法不是简单的 engineering knobs，而是不同的概率语义和教师关系假设。

---

## 3. 核心研究问题

### RQ1：多教师 OPD 中常见的 aggregation operator 分别优化了什么？

系统分析以下 reference construction：

$$
p_{\mathrm{route}}=p_j,
$$

$$
p_{\mathrm{mix}}(v)=\sum_{k=1}^{K}\alpha_kp_k(v),
$$

$$
p_{\mathrm{geo}}(v)=
\frac{\prod_{k=1}^{K}p_k(v)^{\alpha_k}}
{\sum_u\prod_{k=1}^{K}p_k(u)^{\alpha_k}}.
$$

研究重点不是宣称存在一个对所有任务都正确的 reference，而是确定：什么教师语义对应什么 operator。

### RQ2：教师分歧是否能够预测不同 operator 的性能差异？

定义加权教师不兼容性：

$$
D_{\mathrm{inc}}(s_t;\alpha)
=
-\log\sum_v\prod_{k=1}^{K}p_k(v\mid s_t)^{\alpha_k}.
$$

需要验证：低分歧时各 operator 是否近似等价；高分歧时，最佳 operator 是否取决于“只有一个教师正确”还是“多个教师都提供 valid alternatives”。

### RQ3：教师可靠性和教师分歧能否共同决定 reference construction？

分歧本身不代表正确性。高分歧可能意味着：

1. 一个教师正确、其他教师错误；
2. 多个教师都正确但给出不同 valid modes；
3. 所有教师都不可靠；
4. 教师分布受校准误差或不同 decoding style 影响。

因此需要联合建模：

$$
\text{teacher reliability}
\quad+\quad
\text{teacher incompatibility}
\quad+\quad
\text{valid-mode structure}.
$$

### RQ4：能否构造一个轻量、可解释的 adaptive operator selector？

最终目标不是再增加一个复杂 router，而是根据已经验证的规律，在以下 operator 之间进行选择：

$$
\mathsf{mode}(s_t)
\in
\{\text{route},\text{mix},\text{consensus},\text{abstain}\}.
$$

---

## 4. 理论基础与需要严谨表述的结论

### 4.1 加权 reverse-KL 等价于 geometric pooling

对固定状态 $s$，令 $q(v)$ 为学生分布，$p_k(v)$ 为第 $k$ 个教师分布，且

$$
\alpha_k\ge 0,
\qquad
\sum_k\alpha_k=1.
$$

定义

$$
Z_\alpha
=
\sum_v\prod_kp_k(v)^{\alpha_k},
$$

$$
p_{\mathrm{geo}}(v)
=
\frac{\prod_kp_k(v)^{\alpha_k}}{Z_\alpha}.
$$

则有

$$
\sum_k\alpha_kD_{\mathrm{KL}}(q\|p_k)
=
D_{\mathrm{KL}}(q\|p_{\mathrm{geo}})-\log Z_\alpha.
$$

因此，在 $q$ 可以自由选择的单状态问题上，最优解为

$$
q^*=p_{\mathrm{geo}},
$$

且最小值为

$$
\min_q\sum_k\alpha_kD_{\mathrm{KL}}(q\|p_k)
=
-\log Z_\alpha.
$$

上述等式默认教师分布在共同 token support 上为正；对标准 softmax policy，该条件通常成立。若存在严格的零概率且共同 support 为空，则 $Z_\alpha=0$，相应的 reverse-KL 可能为无穷大。

### 4.2 Disagreement floor

由于加权 AM-GM 不等式，通常有

$$
Z_\alpha\le 1,
$$

于是

$$
D_{\mathrm{inc}}=-\log Z_\alpha\ge 0.
$$

这表示：如果教师分布不一致，即使学生拥有无限 capacity，weighted reverse-KL 也存在一个无法消除的单状态 loss floor。

需要谨慎表述的是：该结论是“无约束单状态分布”的精确结论。真实模型存在参数共享、有限容量、采样估计和 on-policy state distribution，因此全局训练 loss 还会受到其他误差影响。

### 4.3 Loss weighting 与 mixture objective 的差异

weighted reverse-KL 为

$$
L_{\mathrm{geo}}
=
\sum_k\alpha_kD_{\mathrm{KL}}(q\|p_k),
$$

而 arithmetic mixture objective 为

$$
L_{\mathrm{mix}}
=
D_{\mathrm{KL}}\left(q\middle\|\sum_k\alpha_kp_k\right).
$$

二者之差为

$$
L_{\mathrm{geo}}-L_{\mathrm{mix}}
=
\mathbb E_{v\sim q}
\left[
\log\left(\sum_k\alpha_kp_k(v)\right)
-
\sum_k\alpha_k\log p_k(v)
\right]
\ge 0.
$$

差值是由教师之间的分歧产生的 consensus penalty。

### 4.4 Latent teacher 的语义边界

如果 teacher identity 是一个潜变量，并且教师表示多个 alternative hypotheses，那么 arithmetic mixture 有合理的概率解释：

$$
p_{\mathrm{mix}}(v\mid s)
=
\sum_k\rho_k(s)p_k(v\mid s).
$$

但必须区分三种情况：

1. 整个 sample 共享一个 teacher identity；
2. 整条 trajectory 共享一个 teacher identity；
3. 每个 token 都可以重新选择 teacher。

逐 token 的 mixture

$$
\prod_t\sum_k\rho_k(s_t)p_k(y_t\mid s_t)
$$

通常不等于 sequence-level mixture

$$
\sum_k\rho_kP_k(y\mid x).
$$

因此本项目默认研究的是 **local token-level reference construction**，而不是声称它完整实现了 sequence-level latent teacher model。

### 4.5 不把 geometric pool 直接等同于真实 PoE

$$
p_{\mathrm{geo}}(v)
\propto
\prod_kp_k(v)^{\alpha_k}
$$

可以被解释为 consensus pooling 或 logarithmic opinion pool，但只有在教师分布确实能够被视为兼容的 likelihood factors 时，才适合使用严格的 Product-of-Experts 解释。

在 LLM 中，它也可能导致过度尖锐、错误交集和 mode suppression。因此实验需要单独测量 calibration、entropy 和 valid-mode coverage。

---

## 5. 研究假设

### H1：低教师分歧时，三种 reference construction 的表现接近

当

$$
D_{\mathrm{inc}}(s_t;\alpha)\approx 0
$$

时，有

$$
p_{\mathrm{mix}}\approx p_{\mathrm{geo}}\approx p_k
$$

的局部近似，因此 route、mixture 和 consensus 的性能差异应较小。

### H2：单一可靠教师场景中，hard route 或 best-teacher 优于 consensus

当一个教师在 sample 上可靠，而其他教师明显不可靠时，geometric pool 会因为其他教师的近零概率压低正确 token，从而产生错误的 consensus penalty。

### H3：多个可靠但互斥的 valid modes 场景中，arithmetic mixture 优于 geometric pool

当多个教师均能生成正确答案，但在词汇、推理路径或表达方式上存在明显差异时，arithmetic mixture 应比 geometric pool 更能保留多个候选 mode。

### H4：仅依赖 disagreement 不能确定最佳 operator

相同的高分歧可能对应不同的正确 aggregation。必须加入 teacher correctness、verifier feedback 或 calibration information。

### H5：基于语义选择 operator 的方法应优于固定 aggregation

如果 H1 至 H4 成立，则 adaptive operator selector 应在不同分桶上分别接近对应的 oracle operator，并整体优于单一固定方法。

---

## 6. 研究基座与实现基座

### 6.1 总体决策

本项目将“研究基座”和“实现基座”分开选择：

- **研究基座：Open-MOPD。** 使用其公开的 MixSFT student、Math/Code/IF 三个领域 teacher、训练数据、评测数据和 MOPD pipeline，尽可能固定 teacher quality、student initialization、domain composition 和 evaluation protocol。
- **实现基座：优先 fork Open-MOPD 自带的 patched verl。** 先用最小改动跑通官方 Naive M-OPD / Open-MOPD，再在其 teacher-output 接口上增加 fan-out 和 reference construction。
- **备用实现基座：upstream verl。** 只有当 patched verl 的 teacher-output schema、top-$k$ 输出或并发机制明显阻碍实验时，才将 reference operator 迁移到最新版 upstream verl。

Open-MOPD 适合作为研究基座，是因为它公开了完整的端到端复现链路：MixSFT 初始化、三个领域 RL teacher、最终 student、训练与评测数据以及本地启动脚本。其官方设置使用已知 domain label，将 capability integration 与 teacher routing 分开，因此可以作为 matched hard-route baseline 和受控实验环境。详见 [Open-MOPD repository](https://github.com/BytedTsinghua-SIA/Open-MOPD)。

upstream verl 的价值主要在实现抽象：它已经包含 MultiTeacherModelManager、异步 teacher manager、按样本 teacher dispatch、teacher log-probability 收集以及 forward_kl_topk 等 distillation plumbing。需要明确的是，当前文档中的 multi-teacher 示例仍然是由 teacher_key 按样本路由到某一个 teacher；它不是本项目所需的 all-teacher fan-out，也不自动实现 route/mix/geometric reference construction。详见 [verl OPD documentation](https://github.com/verl-project/verl/blob/main/docs/algo/opd.md)。

### 6.2 为什么不能直接把普通 MOPD 当作本项目实现

普通 MOPD 的典型流程是

$$
x\in\text{Math}
\;\Longrightarrow\;
\text{只 query Math teacher}.
$$

本项目需要改成

$$
x\in\text{Math}
\;\Longrightarrow\;
\text{Math、Code、IF 三个 teacher 都读取同一个 student prefix}.
$$

只有这样才能得到

$$
\{p_1(\cdot\mid s_t),p_2(\cdot\mid s_t),p_3(\cdot\mid s_t)\}
$$

并进一步构造

$$
\{p_1,p_2,p_3\}
\;\longrightarrow\;
\mathcal A
\;\longrightarrow\;
p_{\mathrm{ref}}.
$$

因此第一处核心改动不是 loss，而是

$$
\boxed{\text{teacher dispatch: route-one}\;\longrightarrow\;\text{fan-out-all}.}
$$

### 6.3 需要新增的 teacher-output schema

建议将现有单 teacher 输出扩展为：

~~~text
teacher_outputs:
  sampled_logprobs: [num_teachers, batch, response_length]
  topk_token_ids:   [num_teachers, batch, response_length, topk]   # optional
  topk_logprobs:    [num_teachers, batch, response_length, topk]   # optional
  topk_mass:        [num_teachers, batch, response_length]         # optional
  teacher_mask:     [num_teachers, batch]
~~~

最低版本只需要 sampled_logprobs，用于第一阶段的 route、arithmetic mixture 和 weighted reverse-KL 对比。需要完整分布几何时，再打开 top-$k$ 或 full-vocabulary 输出。

### 6.4 sampled-token 阶段与 distribution-level 阶段必须分开

设 student rollout 中实际采样的 token 为 $y_t$，并令

$$
\ell_{k,t}
=
\log p_k(y_t\mid s_t).
$$

对 arithmetic mixture，单个 sampled token 的 reference log-probability 可以直接由

$$
\log p_{\mathrm{mix}}(y_t\mid s_t)
=
\log\sum_k\alpha_k\exp(\ell_{k,t})
$$

得到。因此，第一阶段不需要整个 vocabulary 的 teacher logits。

对 weighted reverse-KL，其 sampled-token 估计只需要

$$
\sum_k\alpha_k\ell_{k,t}.
$$

由于 $Z_\alpha$ 对固定 teacher 和 prefix 而言不依赖 student 参数，优化 weighted reverse-KL 的 sampled-token signal 不必显式计算 $Z_\alpha$。但是，如果要报告或优化归一化的 geometric target，则需要

$$
\log p_{\mathrm{geo}}(y_t\mid s_t)
=
\sum_k\alpha_k\ell_{k,t}
-
\log Z_\alpha.
$$

以下指标不能仅凭 sampled token log-probability 精确得到：

- target entropy；
- teacher top-$k$ overlap；
- exact arithmetic/geometric target distribution；
- exact disagreement floor；
- exact $D_{\mathrm{inc}}$。

因此研究应分成两个阶段：

1. **sampled-token training stage：** 快速比较 route、sampled mixture 和 weighted reverse-KL；
2. **distribution profiling stage：** 使用 top-$k$ 或 full-vocabulary teacher outputs，验证 target geometry 和 disagreement law。

### 6.5 推荐的最小实现改动

第一版只新增两个模块：

~~~text
training/verl/.../reference/
|-- operators.py       # route / sampled mixture / geometric objective
\-- diagnostics.py     # disagreement and bucket statistics
~~~

训练流程由

~~~text
sample.domain
      -> matched teacher
      -> teacher logprob
      -> OPD loss
~~~

改为

~~~text
student prefix
      -> Math teacher
      -> Code teacher
      -> IF teacher
      -> ReferenceConstructor
           -> route / mix / geometric
           -> OPD estimator
~~~

第一版暂不实现 adaptive selector。先保证所有 operator 在相同 prefix、相同 teacher queries、相同 optimizer 和相同 token budget 下可复现比较。

### 6.6 研究基座选择表

| 基座 | 本项目定位 | 优点 | 需要注意的问题 |
|---|---|---|---|
| Open-MOPD | 首选研究基座 | 三领域 teacher/student/data/eval 完整公开，适合 controlled experiments | 原实现主要是 routed MOPD，需要改为 fan-out-all |
| patched verl（Open-MOPD 自带） | 首选第一阶段实现基座 | 与官方 checkpoints、脚本和数据直接兼容 | teacher-output 接口可能只暴露 sampled-token 或 routed output |
| upstream verl | 第二阶段实现基座 | teacher manager、top-$k$ plumbing 和 distillation abstraction 更通用 | 仍需自行实现 all-teacher fan-out 和 reference constructor |
| slime | 对照或辅助实现 | sampled-token OPD 路径清楚 | 更偏 sampled-token logprob，不适合作为 distribution geometry 主基座 |
| NeMo RL | 不作为第一选择 | async multi-teacher worker 和 routing 能力完整 | 工程栈较重，默认 MOPD signal 仍主要是 sampled-token teacher logprob |
| D³-MOPD 实验栈 | 不作为第一选择 | 适合未来比较 domain scheduling | 主实验规模和问题设定会把本项目绑定到 scheduling 方向 |

### 6.7 推荐实验顺序

先在 Open-MOPD 官方 teacher 和 MixSFT student 上做 fixed-rollout profiling：

$$
\text{MixSFT student rollout}
\;\longrightarrow\;
\text{three-teacher fan-out}
\;\longrightarrow\;
\text{reference statistics}.
$$

第一批只使用几百到几千条 prompt，回答：

1. cross-domain teacher disagreement 是否足够大；
2. matched teacher 是否在 sample level 通常最好；
3. arithmetic 与 geometric target 的差异是否集中在少数 token/sample；
4. 是否存在值得训练的 high-disagreement subset。

若观察到

$$
p_{\mathrm{mix}}\approx p_{\mathrm{geo}}
$$

在绝大多数 prefix 上成立，则不应立即进行大规模训练；若 high-disagreement subset 明显，再进入 controlled operator training。

### 6.8 资源与可行性提醒

fan-out-all 会使每个 prefix 至少访问 $K$ 个 teacher，teacher inference 成本和通信量会随教师数近似增加。第一阶段应使用小 profiling subset、teacher-output cache 和 top-$k$ 输出，避免在研究问题尚未验证前消耗完整训练预算。

另外，teacher 和 student 必须共享 tokenizer 与 vocabulary，或者额外实现 token 对齐；upstream verl 的文档明确以同一 tokenizer/vocabulary 作为常规假设。本项目首选 Open-MOPD 的同源模型族，暂不引入异构 tokenizer 对齐问题。

### 6.9 当前复现路线：Qwen3 基线、Open-MOPD 与 D³-MOPD

当前阶段先解决“训练链路和比较基线是否可信”，再进入 reference-construction 算法创新。推荐顺序为：

~~~text
Qwen3 普通 MOPD
    -> Open-MOPD 精确复现
    -> D³-MOPD 风格动态 domain scheduling
    -> 自定义 reference construction / adaptive operator
~~~

这里的三类实验不能混为一个实验：

| 实验 | student / teacher | 训练数据与目标 | 复现含义 |
|---|---|---|---|
| Qwen3 普通 MOPD | Qwen3-4B student；现有 Qwen3 Math/Code teacher，补充同源 IF teacher | G-OPD 风格 Math/Code/IF prompt pool；固定 domain ratio 或 matched route | 验证本地 Qwen3 训练、reward、tokenizer 和多教师接口 |
| Open-MOPD 精确复现 | SmolLM3-3B MixSFT student；官方 SmolLM3-3B Math/Code/IF teacher | `Open-MOPD-Data` 与官方评测协议 | 复现 Open-MOPD 的模型、数据和报告结果；不能替换成 Qwen3 后仍称为精确复现 |
| D³-MOPD 风格复现 | Qwen3-4B 或 Qwen3.5-4B student；同 backbone 的 Math/Code/IF teacher，可选 Tool-use teacher | 每域规模接近、带 domain tag 的 prompt pool；固定 mixture 对比动态 mixture | 验证 D³ 的调度思想能否迁移到当前 verl/Open-MOPD 栈 |
| D³-MOPD 规模复现 | Qwen3.6-35B-A3B student；四个同 backbone domain expert | Math、Code、IF、Tool-use 四域，约 4k prompts/domain | 需要重新准备同源 teachers 和大规模基础设施，不是当前两张 GPU 的首轮实验 |

Qwen3-4B 应作为第一主线，因为现有 Math/Code teacher 可以复用，当前只需要补充 IF teacher。Qwen3.6-35B-A3B 只能作为后续规模实验：它要求重新训练或取得 Qwen3.6 同源 teachers，且论文主实验使用 7 个 GPU 节点。D³-MOPD 的 4B 附录支持小模型迁移，但不能证明 Qwen3-4B 与 Qwen3.5-4B 完全等价。

Open-MOPD 是主开发库，G-OPD 只作为 Qwen3 baseline 的数据、teacher 和 launcher 参考，D³-MOPD 作为调度算法的论文规范。不要把 D³ 的 slime/Megatron/SGLang 训练栈整体移植进来：D³ 的核心变化是读取每域 reverse-KL 的 remaining gap 与 descent velocity，然后更新下一批的 domain quota；当前 Open-MOPD 已有固定 domain sampler 和 `AbstractCurriculumSampler` 更新接口，足以承载这一变化。

复现难度应分开记录：

- **Qwen3 普通 MOPD：中等偏低。** G-OPD 已有 Qwen3 launcher 和数据，但必须核对 `only_reverse_kl_advantages`、`lambda_vals`、thinking mode、reward manager 以及多 teacher 的实际路由语义。
- **Open-MOPD 精确复现：中等。** 官方模型和数据链路完整，但它是 SmolLM3 同源配置，不能直接用 Qwen3 teacher 替代；本地 patched verl 仍需通过 dry-run 和最小 smoke test 验证。
- **D³-MOPD 算法复现：中等。** 算法层面主要新增 scheduler、KL history、domain quota 和日志；精确论文复现则较难，因为论文使用 slime、异步 teacher prefill、7 节点基础设施，公开原文主要提供 recipe 而非可直接运行的完整代码。

第一版 D³-style scheduler 必须保持以下变量不变：student、teacher、tokenizer、thinking mode、OPD loss、response budget、总 prompt/rollout budget、验证集和随机种子。只改变 fixed mixture 与 online-updated mixture，才能把收益归因于调度。

### 6.10 当前 IF 数据与评测资产

截至 2026-09-10，`scripts/local/if_grpo_32gpu.sh` 所需的 IF 数据已经下载并完成 Open-MOPD schema 转换。默认数据根目录为：

```text
/nfs/dataset-ofs-nmgvoyagermodel-prediction/rl_lab/liushiqi/llm/dataset
```

原始下载源和当前 launcher 使用的路径如下：

| 用途 | 数据源 | launcher 默认路径 | 当前规模 |
|---|---|---|---:|
| IF-GRPO 训练 | `nvidia/Nemotron-RL-instruction_following` | `train/instruction_following.jsonl` | 46,391 条 |
| IFEval 对齐评测 | `BytedTsinghua-SIA/Open-MOPD-Data` 的 `eval/if/ifeval_aligned.parquet` | `eval/ifeval_aligned.parquet` | 541 条 |
| IFBench 对齐评测 | `BytedTsinghua-SIA/Open-MOPD-Data` 的 `eval/if/ifbench_test_aligned.parquet` | `eval/ifbench_test_aligned.parquet` | 300 条 |

数据构建脚本 `training/scripts/rl/build_if_grpo_data.py` 已生成：

```text
if_grpo/nemotron_if_rl_train.parquet   # 46,391 条训练数据
if_grpo/ifeval_only.parquet             # 541 条验证数据
if_grpo/if_eval.parquet                 # 841 条验证数据：IFEval 541 + IFBench 300
```

当前 launcher 的默认行为是 `INCLUDE_IFBENCH_VAL=auto`：当当前环境没有 `OPENOPD_IFBENCH_REPO` 时，只使用 `ifeval_only.parquet`；即使 IFBench 对齐 parquet 已经下载，也不能据此完成官方 IFBench reward 评测。要使用合并验证集，需在具备 IFBench evaluator checkout 的环境中设置 `OPENOPD_IFBENCH_REPO` 并传入 `--include-ifbench`。本机当前 shell 未设置该变量，因此后续默认 smoke test 应按 IFEval-only 记录，不能把它写成 IFEval+IFBench 结果。

IF reward 依赖已随仓库准备在 `training/third_party/verifiable-instructions`，launcher 的 preflight 会检查 instruction registry 和 `<think>\n</think>` 输出前缀；训练使用 patched verl 的 instruction-following reward、`dapo` reward manager 和 `tiered_think_format` scoring mode。数据和派生 summary 中应保留 seed、输入路径、验证集组成和行数，作为 Phase 0 manifest 的一部分。

需要区分实验含义：`if_grpo_32gpu.sh` 调用的是 `verl.trainer.main_ppo` 的 GRPO 配置（无 teacher query、无 teacher logits、无 OPD reference），因此它可以作为 IF domain 的数据/reward/训练链路 baseline 和 smoke test，但不能作为 multi-teacher reference-construction 的实验结果。进入 Phase 1--3 前仍需准备或确认同源 Qwen3 IF teacher，并在固定 student rollout 上执行 Math/Code/IF teacher fan-out。

---

## 7. 方法设计

### 7.1 统一 reference-construction 接口

在每个 student prefix $s_t$ 上，所有教师读取同一个 prefix，得到

$$
\mathcal P(s_t)=\{p_1(\cdot\mid s_t),\ldots,p_K(\cdot\mid s_t)\}.
$$

然后由 reference operator 构造

$$
p_{\mathrm{ref}}(\cdot\mid s_t)
=
\mathcal A\bigl(\mathcal P(s_t),\rho(s_t),m(s_t)\bigr),
$$

其中：

- $\rho(s_t)$ 表示教师权重或可靠性 posterior；
- $m(s_t)$ 表示 aggregation mode；
- $\mathcal A$ 表示 route、mix 或 consensus operator。

学生统一优化

$$
\mathcal L_{\mathrm{OPD}}
=
\mathbb E_{s_t\sim d_{\pi_\theta}}
\left[
D_{\mathrm{KL}}
\left(
\pi_\theta(\cdot\mid s_t)
\middle\|
p_{\mathrm{ref}}(\cdot\mid s_t)
\right)
\right].
$$

实际代码中需要明确 student distribution 是否 stop-gradient、teacher logits 是否只保留 top-$k$、以及 rollout state distribution 是否固定或随训练更新。

### 7.2 Baseline A：matched hard route

使用原始 domain label 或预定义 domain router：

$$
p_{\mathrm{ref}}=p_{d(x)}.
$$

这是原始 MOPD-style baseline。

### 7.3 Baseline B：uniform arithmetic mixture

$$
p_{\mathrm{ref}}=\frac{1}{K}\sum_{k=1}^{K}p_k.
$$

该方法代表“教师是多个可能的 alternative reference”。

### 7.4 Baseline C：uniform geometric consensus

$$
p_{\mathrm{ref}}(v)
\propto
\prod_{k=1}^{K}p_k(v)^{1/K}.
$$

该方法等价于均匀加权 reverse-KL。

### 7.5 Baseline D：oracle best teacher

利用 answer verifier 或人工标注，在每个 sample 上选择结果最好的教师。该方法不是可部署方法，而是用于估计 reference construction 的上限。

### 7.6 Baseline E：verified mixture

只保留经过 answer verifier 的教师，然后在剩余教师之间进行 arithmetic mixture：

$$
p_{\mathrm{ref}}(v)
=
\sum_{k\in\mathcal V(s)}
\tilde\rho_k(s)p_k(v\mid s),
$$

其中 $\mathcal V(s)$ 是通过验证的教师集合，$\tilde\rho_k$ 为归一化权重。

### 7.7 最终方法：adaptive operator selector

第一版不要直接学习复杂的连续权重，而是先实现离散、可解释的 selector：

$$
m(s_t)
\in
\{\text{route},\text{mix},\text{consensus},\text{abstain}\}.
$$

输入特征可以包括：

- $D_{\mathrm{inc}}$；
- 教师 entropy；
- top-1 / top-$k$ overlap；
- teacher answer correctness 或 verifier score；
- reliability concentration，例如 $\max_k\rho_k$；
- student-teacher log-probability gap；
- token position 和 reasoning stage。

第一版 selector 采用 calibration set 上训练的浅层模型或规则表，不引入新的大型网络。这样可以直接回答：性能提升来自 operator selection，还是来自额外参数和额外训练能力。

---

## 8. 具体实验计划

### Phase 0：分布级 toy verification

目标是先验证理论和实现没有错误，不依赖大型模型。

实验内容：

1. 构造二 token、三 token 和多 token 的教师分布；
2. 比较 arithmetic mixture、geometric pool 和 hard route；
3. 验证 weighted reverse-KL 的最优解；
4. 画出 $D_{\mathrm{inc}}$、target entropy、support overlap 和 operator gap 的关系；
5. 构造“单一可靠教师”和“多个 valid modes”两类 synthetic cases。

验收标准：

- 数值误差与理论公式一致；
- 能观察到 geometric pool 的交集效应；
- 能观察到 mixture 的并集效应；
- 能明确展示相同 disagreement 不一定对应同一个最佳 operator。

### Phase 1：教师能力与 cross-domain competence matrix

对每个 domain 的 held-out prompts，分别运行所有教师，得到：

$$
\mathrm{CompetenceMatrix}_{i,k}
=
\mathrm{Performance}(T_k,\mathcal D_i).
$$

需要回答：

- matched teacher 是否通常最好；
- 非 matched teacher 是否在一部分样本上更好；
- 教师 correctness 是否具有明显 sample-level variation；
- domain label 与真实 teacher competence 的相关性有多高。

输出：

- domain-level matrix；
- sample-level best-teacher frequency；
- verifier agreement；
- matched-route regret。

### Phase 2：固定 rollout 的 token distribution profiling

从固定 student checkpoint 采样 student trajectories。对同一个 prefix $s_t$，让所有教师都进行 prefill，记录：

- 完整或 top-$k$ token logits；
- entropy；
- pairwise KL / JS；
- $D_{\mathrm{inc}}$；
- top-$k$ overlap；
- student-to-teacher log-probability gap；
- 最终答案 correctness。

按以下维度分桶：

- domain；
- sample difficulty；
- token position；
- reasoning stage；
- teacher disagreement；
- number of verified correct teachers。

输出核心图表：

1. disagreement distribution；
2. disagreement 随 token position 的变化；
3. disagreement 与 correctness 的二维分布；
4. 教师 answer correctness 与 token-level agreement 的关系；
5. geometric target entropy 与 arithmetic target entropy 的比较。

### Phase 3：reference operator controlled comparison

固定：

- student rollout；
- teacher checkpoint；
- teacher queries；
- batch size；
- token budget；
- reward refresh；
- sequence length balancing；
- optimizer 和 learning-rate schedule。

只改变 reference construction：

1. matched hard route；
2. random route；
3. oracle best teacher；
4. uniform arithmetic mixture；
5. uniform geometric consensus；
6. verified mixture；
7. adaptive operator selector。

每种设置至少运行多个随机种子，并报告平均值、标准差和分领域结果。

### Phase 4：按结构分桶的 operator evaluation

实验不能只报告 overall average。必须报告以下四类：

| 分桶 | 判定条件 | 预期最佳方法 |
|---|---|---|
| 一致且可靠 | 教师分歧低，多数教师正确 | 三者接近 |
| 单一可靠教师 | 仅一个教师通过验证 | route / best teacher |
| 多个 valid alternatives | 多个教师正确但分布差异高 | arithmetic mixture |
| 高分歧且均不可靠 | 教师均未通过验证 | abstain / base student |

核心不是证明某一种方法永远最好，而是证明上述结构能够解释 operator 的相对表现。

### Phase 5：adaptive selector

先在 calibration split 上训练 selector，再在严格 held-out split 上评估。比较：

- 固定 route；
- 固定 mixture；
- 固定 consensus；
- oracle operator；
- learned selector。

关键指标：

$$
\mathrm{SelectorRegret}
=
\mathrm{Score}(\text{oracle operator})
-
\mathrm{Score}(\text{selected operator}).
$$

如果 selector 能显著降低 regret，并在不同 domain 和随机种子下稳定，则说明 operator-selection 具有实际价值。

---

## 9. 评价指标

### 8.1 最终任务性能

- 各 domain accuracy / reward；
- overall average；
- weakest-domain performance；
- domain gap；
- 是否超过 matched teacher ensemble；
- 是否超过 best single teacher。

### 8.2 Reference fidelity

- 学生与 reference 的 token-level KL；
- student target cross-entropy；
- target entropy；
- top-$k$ overlap；
- valid-token coverage；
- sequence-level answer correctness。

### 8.3 结构性指标

- $D_{\mathrm{inc}}$；
- pairwise JS divergence；
- teacher correctness count；
- best-teacher margin；
- selector regret；
- calibration error；
- teacher query cost。

### 8.4 资源指标

- 每个 training step 的 teacher forward 次数；
- 每个 token 的额外显存；
- wall-clock time；
- logits cache 大小；
- top-$k$ 近似造成的误差。

---

## 10. 工程实施方案

### 9.1 建议代码模块

建议在现有 Open-MOPD 或 MOPD 复现代码上增加独立模块：

```text
reference_construction/
|-- operators.py          # route / mixture / geometric pool
|-- disagreement.py       # KL, JS, incompatibility score
|-- verifier.py           # answer-level correctness
|-- selector.py           # adaptive operator selection
|-- cache.py              # teacher logits and prefix cache
\-- diagnostics.py        # plots and bucket analysis
```

主训练循环只接收统一接口：

```text
teacher_outputs -> reference_operator -> p_ref -> OPD_loss
```

这样可以保证不同实验只改变 reference construction，不改变训练框架。

### 9.2 数值稳定性

Geometric pool 不应直接计算概率乘积，而应使用 log-space：

$$
\log p_{\mathrm{geo}}(v)
=
\sum_k\alpha_k\log p_k(v)
-
\log Z_\alpha.
$$

实现时需要：

- 对 logits 使用 log-softmax；
- 使用 log-sum-exp；
- 对极小概率设置数值下限；
- 明确 full-vocabulary 与 top-$k$ 近似的差异；
- 单独报告 truncation error。

### 9.3 计算成本控制

完整的 $K$ 教师 token distribution profiling 成本较高。建议分三阶段：

1. Phase 0 使用小模型和 full vocabulary；
2. Phase 1 使用固定小规模 profiling subset；
3. Phase 3 使用 top-$k$ logits cache，并定期抽样 full-vocabulary 校验。

不要在第一版同时研究 teacher compression、KV cache sharing 和新的并行系统，否则会削弱论文主线。

---

## 11. 主要风险与应对策略

### 风险 1：理论结论不够新

应对：不要把 weighted reverse-KL 的恒等式作为唯一贡献。将核心贡献转向：

- OPD reference construction 的统一形式化；
- operator 的教师语义解释；
- disagreement × competence 的预测规律；
- adaptive operator selector。

### 风险 2：所有 operator 最终性能差异很小

这并不一定失败，可能说明真实教师在大多数 prefix 上高度一致。此时应转向：

- 寻找高分歧 subset；
- 分析特定 reasoning stage；
- 报告 operator equivalence region；
- 研究什么时候简单 route 已经足够。

### 风险 3：arithmetic mixture 看似合理但实际性能下降

可能原因包括 token-level mixture 破坏 trajectory coherence、教师权重不可靠或教师分布校准差。应增加：

- sequence-level consistency analysis；
- verifier-filtered mixture；
- entropy and calibration control；
- per-token versus per-sample routing ablation。

### 风险 4：Open-MOPD 的优化问题掩盖 reference effect

应先在 token balancing、reward refresh 和 sequence length 处理稳定后做 operator comparison。所有 operator 使用完全相同的训练预算和 rollout pipeline。

### 风险 5：teacher correctness 无法获得

可分三档：

1. 数学、代码等任务使用 verifier；
2. 一般任务使用 judge model，并进行人工抽样校验；
3. 无 verifier 场景只做 disagreement analysis，不声称识别了真正正确教师。

---

## 12. 论文预期贡献

若实验验证假设，论文可以形成以下贡献：

### Contribution 1：统一视角

将 hard routing、loss weighting、arithmetic mixture、geometric consensus 统一为不同的 reference-construction operators。

### Contribution 2：理论澄清

证明在 reverse-KL 下，weighted multi-teacher objective 对应 geometric pooling，并给出不可消除的 disagreement floor。

### Contribution 3：经验规律

证明 teacher disagreement 本身不足以决定 aggregation，需要结合 teacher competence 和 valid-mode structure。

### Contribution 4：自适应方法

提出轻量、可解释的 adaptive operator selector，并在固定 compute budget 下降低相对于 oracle operator 的 regret。

### Contribution 5：诊断工具

提供 teacher disagreement map、cross-domain competence matrix 和 operator-conditioned evaluation protocol。

---

## 13. 暂不研究的内容

为了保持主线清晰，第一版暂不同时研究：

- 将 domain sampling schedule 作为最终 reference-construction 主创新；但在基线复现阶段，必须单独实现和评估 D³-style dynamic scheduling，作为受控的外部调度变量；
- token budget allocation；
- reward refresh strategy；
- gradient surgery；
- teacher parameter merging；
- 新的 verifier 训练方法；
- 多模态 teacher 的输入对齐；
- 大规模系统并行优化。

这些内容可以作为后续工作，或作为固定控制变量。

---

## 14. 时间安排与阶段产出

当前执行计划优先采用“先基线、再复现、后创新”的顺序。下面的阶段优先级高于后续 operator-first 的扩展计划；每个阶段都必须先完成最小 smoke test 和配置记录，再考虑扩大规模。

### 阶段 0：环境、数据和配置清点

目标：保证不同代码库的结果可以比较。

产出：

- 模型、teacher、数据和评测文件的 manifest；
- IF 数据 manifest：46,391 条训练样本、541 条 IFEval、300 条 IFBench，以及对应的 46,391/541/841 条派生 parquet；
- 明确记录当前 launcher 默认采用 IFEval-only，IFBench 仅在 evaluator checkout 可用时纳入；
- tokenizer、chat template、thinking mode 和 vocabulary 检查；
- G-OPD、Open-MOPD launcher 的 dry-run 输出；
- Python、patched verl、Torch、vLLM、SGLang、Ray 版本记录；
- 每个实验的 git commit、seed、batch、sequence length、teacher 顺序和输出目录。

### 阶段 1：Qwen3 普通 MOPD baseline

目标：先确认 Qwen3 训练链路本身可靠，而不是先引入新的 scheduler 或 reference operator。

推荐设置：

```text
Qwen3-4B student
Qwen3-4B Math/Code teachers
Qwen3-4B IF teacher（checkpoint 待确认；IF 数据与 reward 已就绪）
non-thinking mode
固定 domain ratio，先 1:1 或 1:1:1
标准 per-token reverse-KL / G-OPD loss
```

产出：

- Math、Code、IF 分域验证结果；
- IF 域优先报告 IFEval 541 条；IFBench 300 条只有在 evaluator checkout 可用并显式启用后才报告；
- per-domain prompt share、response-token share 和 loss share；
- teacher-student reverse-KL 曲线；
- 单 teacher、固定 route 和固定三域 MOPD 的可复现 checkpoint。

这一阶段不使用 D³ scheduler，也不同时比较 arithmetic/geometric reference，避免把基础训练问题误判为算法收益。

### 阶段 2：Open-MOPD 精确复现

目标：在官方 SmolLM3 配置上验证 Open-MOPD 的公开结果链路。

设置必须尽量遵循官方 recipe：MixSFT student、官方 Math/Code/IF RL teacher、`Open-MOPD-Data`、官方评测集和 thinking mode。Qwen3 实验可以作为本地工程 baseline，但不能替代该阶段的精确复现。

产出：

- 官方模型和数据的版本记录；
- MixSFT、三个 domain teacher 和最终 MOPD 的配置表；
- math/code/IF 分域结果及 overall macro average；
- 与官方报告的差异、失败原因和可接受误差范围。

### 阶段 3：D³-MOPD 风格复现

目标：在 Open-MOPD/patched verl 中只加入动态 domain scheduling，复现 D³ 的算法效应。

先使用 Qwen3-4B 三域设置，不强行复现 Qwen3.6-35B-A3B 的模型规模。每一步同时记录每个 domain 的 reverse-KL，并实现：

1. fixed mixture baseline；
2. remaining-KL gap；
3. smoothed descent velocity；
4. gap 与 velocity 的 composite signal；
5. softmax temperature、domain floor 和 batch-level jitter；
6. 动态 quota 的实际 prompt/token share。

验收标准：固定 mixture 与动态 mixture 的 student、teacher、loss、总 rollout budget 和评测协议一致；D³ scheduler 的变化只体现在 domain sampling ratio。

### 阶段 4：Qwen3.6-35B-A3B 可行性与规模实验

只有阶段 1--3 的小规模结果稳定后才启动。该阶段需要重新准备 Qwen3.6 同源的 Math、Code、IF teacher；若目标是完整 D³ 对齐，还需要 Tool-use teacher 及对应数据。当前两张 GPU 只做模型加载、tokenizer、单 batch 或短时 profiling，不启动论文规模训练。

### 阶段 5：提出自己的 reference-construction 算法

只有在普通 MOPD、Open-MOPD 和 D³-style scheduling 都有可比较结果后，才进入本项目原定的 operator 研究：

```text
固定 scheduler
    -> hard route / arithmetic mixture / geometric consensus
    -> disagreement + reliability profiling
    -> 自定义 adaptive reference construction
```

新算法的第一版必须与 D³ scheduler 解耦。默认先固定 domain ratio，再研究 reference operator；之后再做“固定 operator 下的 scheduler”与“固定 scheduler 下的 operator”双向消融。

### 阶段 6：长期 operator 研究

在上述复现链路通过后，继续执行 Phase 0--5 的 toy verification、competence matrix、distribution profiling、controlled comparison、bucket evaluation 和 adaptive selector。最终报告必须区分：

- G-OPD / 普通 MOPD 的 loss 与 teacher routing；
- Open-MOPD 的模型、数据和 domain balancing；
- D³-MOPD 的动态 sampling；
- 本项目自己的 reference-construction operator。

---

## 15. 最低可行版本与成功标准

### 最低可行版本

即使不实现最终 selector，只要完成以下结果，也可以形成一篇分析型工作：

1. 理论上确认 weighted reverse-KL 对应 geometric pool；
2. 在真实 MOPD prefix 上测量 teacher disagreement；
3. controlled comparison route、mixture 和 consensus；
4. 证明性能差异能够按 teacher competence structure 分桶解释。

### 强版本成功标准

满足以下条件时，方向具有较强论文潜力：

- 低 disagreement 区域内各 operator 表现近似；
- 高 disagreement 区域出现稳定的结构化差异；
- 单一可靠教师场景中 route 明显占优；
- 多个 valid alternatives 场景中 mixture 明显占优；
- geometric consensus 只在教师相互兼容时表现稳定；
- adaptive selector 接近 oracle operator；
- 结论跨 domain、student checkpoint 和随机种子成立。

---

## 16. 最终推荐的论文主张

建议最终不要主张：

> 我们提出了一个更好的多教师权重方法。

而应主张：

> Multi-teacher OPD is under-specified until the reference-construction operator is defined. Routing, arithmetic mixing, and geometric consensus encode different assumptions about teacher competence and teacher semantics. We show that teacher disagreement alone is insufficient: the appropriate operator is determined jointly by distributional incompatibility and teacher reliability.

中文表述为：

> 多教师 OPD 在 reference construction 未被明确之前其实是不完整的。路由、算术混合和几何共识分别编码了不同的教师能力与教师语义假设；真正决定合适 operator 的，不是教师分歧单独一个信号，而是教师分歧与教师可靠性的联合结构。

这会比“再提出一种 XXX-aware MOPD”更有解释力，也更容易形成清晰的理论—实验闭环。

---

## 17. 参考工作

- [MOPD: Multi-Teacher On-Policy Distillation for Capability Integration in LLM Post-Training](https://arxiv.org/abs/2606.30406)
- [Open-MOPD: Diagnosing and Fixing Capability Imbalance in Multi-Teacher On-Policy Distillation](https://arxiv.org/abs/2608.19098)
- [Learn from Whoever Is Right: Answer-Verified Multi-Teacher Distillation for Multi-Domain LLMs](https://arxiv.org/abs/2609.02548)
- [H-OPD: Confidence Aware Heterogeneous Multi-Teacher Multimodal On-policy Distillation](https://arxiv.org/abs/2607.02592)
- [Language-Specialized Multi-Teacher On-Policy Distillation for Multilingual LLM-Based ASR](https://arxiv.org/abs/2608.03610)
- [Counteraction-Aware Multi-Teacher On-Policy Distillation for General Capability Recovery with Domain Preservation](https://arxiv.org/abs/2605.27115)
- [D³-MOPD: Adaptive Dynamic Domain ScheDuling for Efficient Multi-Teacher Distillation](https://arxiv.org/abs/2608.24987)
- [Open-MOPD official implementation](https://github.com/BytedTsinghua-SIA/Open-MOPD)
- [verl OPD implementation and configuration](https://github.com/verl-project/verl/blob/main/docs/algo/opd.md)
- [slime on-policy distillation implementation](https://github.com/THUDM/slime/blob/main/slime/rollout/on_policy_distillation.py)
- [NVIDIA NeMo RL MOPD documentation](https://github.com/NVIDIA-NeMo/RL/blob/main/docs/about/algorithms/mopd.md)
