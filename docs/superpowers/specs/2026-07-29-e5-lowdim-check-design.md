# SMCO-EVO E5 低维非退化检查设计

- 日期：2026-07-29
- 分支：`feat/smco-evo-highdim-paper-2026`
- 关联：实现计划 E5（low-dim non-degradation check）；补 Task 10 标注的 gap（`run_smco_evo_lowdim_check.py` 当前是 cocoex 骨架 + `NotImplementedError`）
- 状态：设计已与用户对齐（自算 CSV、Py winner、不改 SMCO 核心）

## 1. 背景与动机

E5 检查高维 E1 winner 在低维（COCO `bbob`，d∈{5,20}）是否系统退化：winner vs matched
non-EVO base，B_max = 2000·d FE（计划 E5）。结果进 supplement；除非严重退化，不推翻高维 winner。

当前 `scripts/run_smco_evo_lowdim_check.py` 仅是契约骨架（`_have_cocoex` + `NotImplementedError`）。
本机现已装 `coco-experiment` 2.8.2（`import cocoex`）+ `cocopp` 2.8.8，E5 可本机完整 TDD。

## 2. 目标

- 在 COCO `bbob` suite（24 函数 × instances 1-5 × d∈{5,20} = 240 problems）上跑 winner (EVO) +
  matched base (non-EVO)，FE = 2000·d。
- 产出 supplement 表（自算 CSV）：per (func×dim×instance) 的 winner/base 指标 + winner-vs-base 对比，
  回答"低维是否退化"。

## 3. 非目标

- 不改 SMCO 核心（用现有 `optimizer.smco`/`smco_evo` API；不动 `_run_evolutionary_states`）。
- 不跑强基线（E5 只 winner vs matched base；baselines 是 E3/E4）。
- 不要求 Py/R 跨语言逐轨迹一致（E5 是 Python supplement；Task 5 仍是可选诊断）。
- cocopp 标准 figure 不在本范围（自算 CSV 足够；cocopp 后处理可后续叠加，不阻塞）。

## 4. 设计

### 4.1 数据流

```
cocoex.Suite('bbob','instances:1-5','dimensions:5,20') → 240 problems
两趟独立遍历（problem 状态不互相污染）：
  winner pass:  for p in suite: p.observe_with(observer_w); run EVO; 记录指标
  base pass:    for p in suite: p.observe_with(observer_b); run BASE; 记录指标
→ supplement CSV（per func×dim×instance×algorithm 指标 + winner-vs-base diff）
```

每趟用独立 `Observer`（`result_folder` 区分 winner/base），problem 在 suite 迭代中天然独立。

### 4.2 coco_runner（新 `src/smco/coco_runner.py`）

`run_on_problem(problem, *, algorithm_id, fe_budget, n_starts, seed, observer=None) -> dict`：
- **objective 包装**：SMCO maximize → `g(x) = -problem(x)`（cocoex 是 minimization）。每次 `problem(x)`
  同时被 cocoex observer（若 observed）与 SMCO 的 FE 计数记录，两者一致。
- **bounds**：`problem.lower_bounds` / `problem.upper_bounds`。
- **starts**：bounds 内 `n_starts=8` 个点，`rng = np.random.default_rng(seed)`，`seed` 派生自
  `problem.id`（`int(sha256(problem.id)[:8],16)`，确定性、与 run 顺序无关）。
- **FE hard stop**：`EvaluationContext`（Task 1，`max_evals=fe_budget`）传给 SMCO，触发
  `evaluation_budget` termination。
- **算法 dispatch**：`parse_algorithm_id(algorithm_id)` 选 `smco`/`smco_evo`（Py only；R 拒绝），
  复用 highdim_worker 的 control 构造逻辑（iter_max = fe_budget//(2d+1)、evolution_points 等由
  algorithm_id 的 family/semantics + 冻结默认）。
- **返回指标**：`{"best_observed_fvalue1": float(problem.best_observed_fvalue1), 
  "final_target_hit": bool(problem.final_target_hit), "evaluations": int(problem.evaluations)}`。
  这些是 cocoex 在 run 内累计的（minimization 语义：best_observed_fvalue1 越小越优，
  final_target_hit 表示达到 COCO final target）。

### 4.3 lowdim runner（改 `scripts/run_smco_evo_lowdim_check.py`）

- 遍历 `Suite('bbob','instances:1-5','dimensions:5,20')`，对每个 problem 跑 winner + matched base
  （两趟独立遍历，独立 observer/result_folder）。
- winner 的 matched base：同 family 的 non-EVO（如 winner=`PY-SP-SMCO-EVO` → base=`PY-BASE-SMCO`；
  winner 含 refine/boost → 对应 base family）。`parse_algorithm_id` 派生。
- **winner 语言归一**：cocoex 是 Python；若 `--winner` 是 `R-*`，runner 内部转同 family/semantics 的
  `PY-*`（E5 用 Py implementation）。
- **supplement CSV**（`result/e5_lowdim/lowdim_degradation.csv`）：列
  `function,dim,instance,algorithm,best_observed_fvalue1,final_target_hit,evaluations`；外加聚合
  `lowdim_summary.csv`（per func×dim：winner/base 的 final_target_hit rate + median best + ERT）。
- CLI：`--winner`（E1 winner algorithm_id）、`--dims`（默认 5 20）、`--instances`（默认 1-5）、
  `--fe-budget-per-d`（默认 2000）、`--result-dir`。

### 4.4 关键约定

- **指标方向**：`best_observed_fvalue1` 是 minimization（越小越优）；winner 优于 base ⇔ winner 的
  best ≤ base 的 best（或 target_hit rate 更高）。
- **FE 一致**：`problem.evaluations` 必须 ≤ `fe_budget`（audit 项）。
- **确定性**：starts seed 派生自 `problem.id` + `algorithm_id`（不同算法不同 starts？或同 problem 同
  starts 以公平对比）。**采用**：seed = hash(problem.id)（winner/base 用同 starts 公平对比）。

### 4.5 测试策略（TDD，本机可测）

- `coco_runner.run_on_problem`：在 1 个 bbob problem（d=5）上跑 base SMCO，断言
  `evaluations ≤ fe_budget`、`best_observed_fvalue1` 合理（≤ 初始随机点）、`final_target_hit` 是 bool。
- FE hard stop：小 budget（如 50）触发 `evaluations ≤ 50`。
- starts 确定性：同 problem.id 同 seed。
- 端到端冒烟：小子集（1 函数 × 1 instance × d=5）跑 winner+base，产出 CSV 行数正确。
- cocopp 不在测试关键路径（自算 CSV）。

## 5. 影响与兼容

- 新 `src/smco/coco_runner.py` + 改 `scripts/run_smco_evo_lowdim_check.py`（替换骨架）。
- 新 `tests/test_coco_runner.py`。
- 纯本机可测（cocoex+cocopp 已装）；不改 SMCO 核心、不影响 Gate A/B。
- `coco-experiment`/`cocopp` 加入 `pyproject.toml` 的 `paper` optional deps（Task 12 一并处理）。

## 6. 决策记录

- **自算 CSV（非 cocopp）**：supplement 只需回答"低维是否退化"，target-hit rate / median best / ERT
  自算足够，与 `merged/` 同风格；cocopp 标准 figure 可后续叠加，不阻塞。
- **Py winner（R→Py 归一）**：cocoex 是 Python；E5 是 supplement，Py implementation 足够；不要求 Py/R
  逐轨迹一致（Gate A/B 不要求，Task 5 可选）。
- **不改 SMCO 核心**：用现有 `optimizer` API 包装 cocoex problem，风险最低。
- **同 starts 公平对比**：winner/base 用同 problem.id 派生的 starts，消除 starts 差异干扰。

## 7. 实施顺序（概要，详见 writing-plans）

1. `coco_runner.run_on_problem`（objective 包装 + starts + FE hard stop + 指标）— TDD
2. lowdim runner 遍历 + winner/base + supplement CSV
3. 端到端冒烟 + 全量 pytest
