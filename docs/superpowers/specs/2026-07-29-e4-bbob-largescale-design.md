# SMCO-EVO E4 bbob-largescale 外部基准设计

- 日期：2026-07-29
- 分支：`feat/smco-evo-highdim-paper-2026`
- 关联：实现计划 E4（bbob-largescale 外部基准）；补 Task 10 标注的 gap（`run_smco_evo_bbob_largescale.py` 当前是 cocoex 骨架 + `NotImplementedError`）
- 状态：设计已与用户对齐（winner+base+5 baselines，自算 CSV，不改 SMCO 核心）

## 1. 背景与动机

E4 在 COCO `bbob-largescale`（24 函数, d∈{160,320,640}, instances 1-5）上跑 E1 winner +
matched non-EVO base + 5 强基线（DE/GA/PSO/SA/GenSA，与 E3 高维 baseline 集一致），B_max =
1000·d FE，验证 SMCO-EVO 高维优势在外部标准基准上复现（计划 E4，figure 5 数据）。

当前 `scripts/run_smco_evo_bbob_largescale.py` 仅骨架。本机已装 `coco-experiment`/`cocopp`，
`bbob-largescale` suite 已验证可用（d160, 24 problems）。

## 2. 目标

- 在 `bbob-largescale`（24 func × inst 1-5 × d∈{160,320,640} = 360 problems）上跑 winner (EVO)
  + matched base (non-EVO) + 5 baselines，FE = 1000·d。
- 产出 figure-5 数据 CSV（自算）：per (func×dim×instance×algorithm) 指标 + 算法间对比 summary。

## 3. 非目标

- 不改 SMCO 核心（复用 E5 的 `coco_runner.run_on_problem` + 新增 baseline 路径）。
- 不要求 Py/R 跨语言逐轨迹一致（E4 是 Python 外部基准）。
- cocopp 标准 figure 不在本范围（自算 CSV 足够 figure 5 数据；cocopp 可后续叠加）。

## 4. 设计

### 4.1 数据流

```
cocoex.Suite('bbob-largescale','instances:1-5','dimensions:160,320,640') → 360 problems
7 趟独立遍历（每算法独立 observer/result_folder，problem 互不污染）：
  winner (SMCO EVO)  → coco_runner.run_on_problem
  matched base (SMCO non-EVO) → coco_runner.run_on_problem
  DE / GA / PSO / SA / GenSA  → coco_runner.run_baseline_on_problem (新)
→ figure-5 CSV（per func×dim×instance×algorithm 指标 + summary）
```

### 4.2 coco_runner 扩展（`src/smco/coco_runner.py`）

新增 `run_baseline_on_problem(problem, *, algorithm_name, fe_budget, n_starts=8, seed=None, observer=None) -> dict`：
- **objective**：`_CocoMinObserver(problem, fe_budget)`——minimization（直接 `problem(x)`，不取负），
  FE hard stop（`EvaluationBudgetExceeded` at `fe_budget`，复用 `smco.evaluation`），**clip 到 bounds + non-finite 惩罚**（与 SMCO 路径一致的数值保护）。
- **dispatch**：`_BASELINE_DISPATCH = {"DE": differential_evo, "GA": genetic_algorithm, "PSO": particle_swarm, "SA": simulated_annealing, "GenSA": gensa}`（复用 `comparison.methods`，与 `baseline_worker` 同源）。
- **调用**：`algorithm(observer, lower, upper, start_points=starts, maximize=False, max_iter=fe_budget, seed=seed)`；`try/except EvaluationBudgetExceeded: pass`（预期 hard stop）。
- **starts**：bounds 内 n_starts=8，seed = `problem_seed(problem)`（与 winner/base 同 starts 公平对比）。
- **返回**：与 `run_on_problem` 同结构（`best_observed_fvalue1`/`final_target_hit`/`evaluations` + `algorithm_id`=baseline 名）。

### 4.3 bbob-largescale runner（改 `scripts/run_smco_evo_bbob_largescale.py`）

- 遍历 `Suite('bbob-largescale','instances:1-5','dimensions:160,320,640')`，7 算法各一趟独立遍历。
- winner 用 `to_py(winner)` 归一（R→Py）；matched base 同 E5（同 family non-EVO）。
- 5 baselines：`BASELINE_NAMES = ("DE","GA","PSO","SA","GenSA")`（与 `baseline_worker.BASELINE_NAMES` 一致）。
- **figure-5 CSV**（`result/e4_bbob_largescale/bbob_largescale.csv`）：列
  `function,dim,instance,algorithm_id,best_observed_fvalue1,final_target_hit,evaluations`；
  外加 `bbob_largescale_summary.csv`（per func×dim：7 算法的 target_hit rate + median best）。
- CLI：`--winner`、`--dims`（默认 160 320 640）、`--instances`（默认 1-5）、`--fe-budget-per-d`（默认 1000）、`--baselines`（默认全 5）、`--result-dir`。

### 4.4 关键约定

- **指标方向**：`best_observed_fvalue1` minimization（越小越优）；bbob-largescale 是 transformed objective（最优可负，E5 已验证 f2 类）。
- **FE 一致**：SMCO 用 `max_evals` hard stop；baseline 用 `_CocoMinObserver` hard stop；`problem.evaluations ≤ fe_budget`（audit 项）。
- **公平对比**：所有 7 算法同 `problem_seed` 派生 starts、同 FE budget。
- **数值保护**：SMCO + baseline 路径都 clip 到 bounds + non-finite 惩罚（防界外外推 + nan 伪 best）。

### 4.5 测试策略（TDD，本机可测）

- `run_baseline_on_problem`：1 个 bbob-largescale problem（d160 太慢，测试用 d20 bbob 或 bbob-largescale 最小 d），跑 GenSA，断言 `evaluations ≤ fe_budget`、`best_observed_fvalue1` 合理、`final_target_hit` bool。
- FE hard stop：小 budget 触发 `evaluations ≤ budget`。
- 拒绝未知 baseline。
- 端到端冒烟：小子集（bbob d20 代替 largescale 加速？或 bbob-largescale d160 1 func 1 inst）跑 winner+base+5 baselines，CSV 行数正确。
- 注：bbob-largescale d160 单 SMCO run ~160k FE 较慢；测试用小 budget（如 5·d）+ 1 函数 1 instance 冒烟。

## 5. 影响与兼容

- 改 `src/smco/coco_runner.py`（加 `run_baseline_on_problem` + `_CocoMinObserver`）、`scripts/run_smco_evo_bbob_largescale.py`（替换骨架）、`tests/test_coco_runner.py`（追加）。
- 纯本机可测；不改 SMCO 核心、不影响 Gate A/B。
- `comparison.methods` 已是 baseline_worker 依赖，E4 复用。

## 6. 决策记录

- **5 baselines（与 E3 一致）**：DE/GA/PSO/SA/GenSA 全上，与高维 baseline 集一致，便于跨基准比较。
- **自算 CSV（非 cocopp figure）**：与 E5 一致，figure 5 数据用自算指标足够；cocopp 标准图可后续叠加。
- **`_CocoMinObserver`（不复用 baseline_worker._MinObserver）**：baseline_worker 的 observer 绑 synthetic HighDimInstance；cocoex 需独立 observer（problem + clip/nan）。
- **不改 SMCO 核心**：复用 E5 `run_on_problem` + 新增 baseline 路径，风险最低。

## 7. 实施顺序（概要，详见 writing-plans）

1. `coco_runner.run_baseline_on_problem` + `_CocoMinObserver`（TDD）
2. bbob-largescale runner + figure-5 CSV（TDD）
3. 端到端冒烟 + 全量 pytest
