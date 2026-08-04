# SMCO-EVO 高维论文结果合同

日期：2026-07-28
状态：与 `result/smco-evo-paper-highdim-2026/` 战役绑定；冻结后字段语义只能通过
`schema_version` 升版修改，不静默改列含义。
配套：[实验方案](smco-evo-highdim-paper-experiment-plan-2026-07-28.md) /
[实现计划](smco-evo-highdim-implementation-plan-2026-07-28.md)

本文档是 Python runner、R runner、merge 脚本和 CSV header 的**唯一字段权威**。
任何枚举值、字段名或拼接规则的变更必须同时改这里、`src/smco/` 常量、
`vendor/SMCO_R/main/` 常量，并升 `schema_version`。

## 1. 枚举字段

```text
language        = python | r
state_semantics = state_preserving | restart        # 仅 evolutionary=true 时有意义
family          = smco | smco_refine | smco_boost_refine
evolutionary    = true | false
strategy        = rand1bin | current-to-best1bin | best1bin | sobol   # 仅 EVO；base 行写 "none"
objective_sense = minimize | maximize
status          = success | algorithm_failure | infra_failure | timeout
is_confirmatory = true | false
```

`family` 与公共 API 的对应（命名不得再用 `SMCO_R` 同时表示语言和 family）：

```text
family              python API      R entry
smco                smco            SMCO_single / SMCO
smco_refine         smco_r          SMCO_single_refine
smco_boost_refine   smco_br         SMCO_single_boost_refine
```

## 2. algorithm_id 拼接

唯一稳定格式：

```text
algorithm_id = f"{LANG}-{SLOT}-{FAMILY_TOK}" + ("-EVO" if evolutionary else "")
```

- `LANG` ∈ {`PY`, `R`}
- `SLOT` ∈ {`SP`, `RS`, `BASE`}
  - `SP` = state_preserving（仅 EVO）
  - `RS` = restart（仅 EVO）
  - `BASE` = 非 EVO 基线（`evolutionary=false`，不占用语义槽）
- `FAMILY_TOK`：`smco`→`SMCO`，`smco_refine`→`SMCO-REFINE`，
  `smco_boost_refine`→`SMCO-BOOST-REFINE`
- EVO 后缀：`evolutionary=true` 加 `-EVO`，否则无后缀

示例：

```text
PY-SP-SMCO-EVO              # Python, state-preserving, smco family, EVO
PY-RS-SMCO-BOOST-REFINE-EVO # Python, restart, boost_refine family, EVO
R-SP-SMCO-REFINE-EVO        # R, state-preserving, refine family, EVO
R-RS-SMCO-EVO               # R, restart（当前 R EVO 的历史语义）
PY-BASE-SMCO                # Python 非 EVO smco 基线
R-BASE-SMCO-BOOST-REFINE    # R 非 EVO boost_refine 基线
```

## 3. run_id 派生

```text
run_id = "r" + sha256(canonical_json(task))[:16]
```

`canonical_json(task)` 是键按字典序排列、无空格、UTF-8 的 JSON，必须且仅包含下列
键（缺失键按空串/0 参与哈希，因此 Python/R/merge 三处构造必须逐键一致）：

```text
stage
suite
function
dimension
instance
replication
algorithm_id
evolution_strategy          # base 行填 "none"
seed                        # 主种子，十进制无前导零
fe_budget                   # 十进制整数
n_starts                    # 十进制整数
configuration_hash          # 见第 6 节
```

并行调度、执行顺序、重试不得改变 `run_id`。同一 run key 在 Python、R 和 merge
脚本中必须生成同一 `run_id`。

## 4. configuration_hash

```text
configuration_hash = sha256(canonical_json({
    "algorithm_id": ...,
    "evolution_strategy": ...,
    "n_starts": ...,
    "evolution_points": ...,        # 如 "0.50,0.75"（两位小数、逗号无空格）
    "elimination_rate": ...,        # 如 "0.25"（两位小数）
    "de_factor": ...,
    "de_crossover": ...,
    "fe_budget_per_d": ...,         # 如 1000，B_max = fe_budget_per_d * d
    "state_semantics": ...,         # base 行填 "none"
    "refine_ratio": ...,            # 仅 boost_refine family；其余填 "none"
    "objective_sense": ...,
})).hexdigest()[:16]
```

所有浮点按两位小数字符串化（与语言无关），整数按十进制。这保证 Python 与 R 对同一
配置算出同一 hash。

## 5. 结果行必填字段

每个 run 的每个 FE checkpoint 写一行（长格式）。字段（`schema_version = "1"`）：

```text
schema_version              # "1"
manifest_id                 # manifest 文件的 sha256[:16]
stage                       # e0_contract | e1_development | e1b_baseline_selection
                            # | e2_factorial_highdim | e3_baselines_highdim
                            # | e4_bbob_largescale | e5_lowdim_check | e6_ablations
suite                       # synthetic_highdim | bbob_largescale | bbob | contract
function
dimension
instance
replication
seed
language
state_semantics             # base 行填 "none"
family
evolutionary
evolution_strategy          # base 行填 "none"
algorithm_id
n_starts
fe_budget
fe_used
checkpoint_fe               # 本行对应的 FE checkpoint；终行 = fe_used
best_value                  # 该 checkpoint 的 best-so-far（objective_sense 原值）
known_optimum
normalized_gap              # max(best-f*,eps)/max(initial_reference-f*,eps)，eps=1e-12
objective_sense
target_hit_fe_1e-1          # 命中目标所需 FE，未命中留空（右删失）
target_hit_fe_1e-2
target_hit_fe_1e-3
target_hit_fe_1e-5
wall_time_sec               # 到该 checkpoint 的累计墙钟
peak_memory_mb
status
failure_reason              # status=success 时为空
is_confirmatory             # e1_development/e0_contract/e6_ablations=false；其余=true
supersedes_run_id           # infra 重跑时指向被取代的旧 run_id；否则为空
machine_id
git_commit
environment_hash
start_points_hash
instance_hash
configuration_hash
run_id
termination_reason          # iteration_limit | evaluation_budget | boundary_budget
                            # | convergence | clip_stopped | error
fe_counts_by_event          # JSON 文本：{"initialization":i,"finite_difference":f,...}
```

事件名（与 `evaluation.py` / `evaluation_budget.R` 严格一致）：

```text
initialization
finite_difference
iterate
replacement_initialization
restart_initialization      # RS 边界重启（SP 不产生）
refine
boost                       # boost_refine family 的 boosted branch
clip_recheck
```

`fe_used = sum(fe_counts_by_event.values()) <= fe_budget` 必须在每行成立。

## 6. 数值与方向约定

- SMCO 内部最大化；对已知最优问题统一转 minimization gap 上报：
  `normalized_gap = max(best_so_far - f*, eps) / max(initial_reference - f*, eps)`，
  `initial_reference` 为共同起点集合的中位目标值，`eps = 1e-12`。
- `best_value` 仍按 `objective_sense` 原值记录（maximize 问题存内部最大值），
  分析脚本据此与 `known_optimum` 计算 gap，**不得在 runner 内部取负**（历史方向 bug
  见 `docs/direction-bug-audit-2026-07-20.md`）。
- 浮点写出统一 `repr` 或 17 位有效数字，避免 Py/R 序列化差异。

## 7. 验收（Task 0）

- 同一 run key 在 Python、R、merge 三处生成同一 `run_id`（单测覆盖）；
- 字段枚举在本文档、`src/smco/` 常量、`vendor/SMCO_R/main/` 常量、CSV header 一致；
- `algorithm_id` 可从 `(language, state_semantics, family, evolutionary)` 唯一重建，
  也可反向解析；
- `schema_version` 字段存在且为 `"1"`。
