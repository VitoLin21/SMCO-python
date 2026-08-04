# SMCO-EVO E6.1 start-count 消融设计

- 日期：2026-07-29
- 分支：`feat/smco-evo-highdim-paper-2026`
- 关联：实现计划 E6.1（start-count 消融）；补 Task 10 标注的 gap（`ablations.py` 已列 `start_count` 维度但未实现）
- 状态：设计已与用户对齐（方案 A：同 instance dir 多 starts 文件）

## 1. 背景与动机

E6.1 消融研究 SMCO-EVO winner 的起点数 `n_starts` 对质量与 FE 效率的影响，三档
**{8, 16, ceil(sqrt(d))}**（计划 E6.1）：8 是 control（winner 默认），16 是固定加倍，
`ceil(sqrt(d))` 是维度自适应档（计划 8："`n_starts=8` 对 `ceil(sqrt(d))` 的质量与 FE 效率"）。

当前每个 instance dir 只有一份 `starts.csv.gz`（`n_starts=8`），`load_starts(dir)` 固定读它；
worker 的 `run_task` 接收 starts 数组，`n_starts = starts.shape[0]`（由 artifact 决定）。
因此无法支持多档 n_starts。

`ceil(sqrt(d))` **随维度变**（d=1000→32, d=3000→55, d=5000→71），所以 E6.1 不能复用 E6.2/E6.3
的固定 config grid，需要 per-dimension expand。

## 2. 目标

- 支持三档 n_starts∈{8, 16, ceil(sqrt(d))}，winner EVO，**dev instances**（与 E6.2/E6.3 一致）。
- 向后兼容：现有 n8 artifact 与 E1/E2/E3/E6.2/E6.3 manifest 全部不受影响。

## 3. 非目标

- 不改 E1/E2/E3/E6.2/E6.3 的任何路径（它们都用 n8）。
- 不碰 confirmatory instances（E6 全程 dev）。
- 不引入新的 n_starts 档（只 {8, 16, ceil(sqrt(d))}）。

## 4. 设计（方案 A：同 instance dir 多 starts 文件）

### 4.1 数据流

```
generate --stage instances --extra-n-starts 16,sqrt
  每个 instance dir：
    transform (1 份) + starts.csv.gz[n8] + starts_n16.csv.gz + starts_n{ceil√d}.csv.gz
    metadata.json 记录每档 hash；instances_index.json entry 带 extra_starts
E6.1 manifest 生成（per-dim，因 ceil√d 随 d 变）
  每个 (func, dim, instance) × n_starts∈{8,16,ceil√dim} → task 携带 n_starts + 对应档 start_points_hash
factorial runner
  load_starts(inst_dir, n_starts=task["n_starts"]) 选对应档 → run_task(starts)
merge（Task 11）正常处理 outcome
```

### 4.2 starts artifact

- 文件命名：n8 → `starts.csv.gz`（向后兼容）；n≠8 → `starts_n{N}.csv.gz`（N 为具体整数，如
  `starts_n16.csv.gz`、`starts_n32.csv.gz`）。**不用 "sqrt" 字面量**——sqrt 档按 dim 算出具体 N。
- `metadata.json`：
  - `n_starts: 8`、`file_hashes.starts`、`transform_sha256` 不变（向后兼容）。
  - 新增 `extra_starts: {"16": {"file": "starts_n16.csv.gz", "hash": ..., "n_starts": 16}, ...}`。
- starts seed：`_starts_seed(function, dim, instance_id, stage, n_starts)`——加 n_starts 维度，
  各档确定且互不相关（不同 n 不会共享同一批随机点的前 k 行）。

### 4.3 组件改动（逐文件）

| 文件 | 改动 |
|---|---|
| `src/smco/highdim_instances.py` | `write_instance_artifacts(instance, starts, out_dir, *, extra_starts=None)`：`extra_starts` 是 `{n: matrix}`，为每档写 `starts_n{N}.csv.gz` + 填 metadata `extra_starts`。`load_starts(artifact_dir, n_starts=8)`：n8 读 `starts.csv.gz`，其他读 `starts_n{N}.csv.gz`；缺档 `FileNotFoundError`。 |
| `scripts/generate_smco_evo_manifests.py` | `_starts_seed(..., n_starts)` 加 n_starts 维度；`build_instance_set` 增参 `extra_n_starts`（如 `("16","sqrt")`，sqrt→按 dim 算 `ceil(sqrt(d))`），为每 instance 生成对应 starts 并经 `extra_starts` 传入 `write_instance_artifacts`；index entry 记 `extra_starts`。CLI 增 `--extra-n-starts`。 |
| `src/smco/experiment_manifests.py` | `expand_tasks` 按 `config["n_starts"]` 选 start_points_hash：n8（或 entry 无 `extra_starts`）用 `entry["start_points_hash"]`；其他用 `entry["extra_starts"][str(n)]["hash"]`。E1/E2 仍 n8，行为不变。 |
| `src/smco/ablations.py` | 新增 `start_count_configs(winner_algorithm_id, dim) -> list[(label, config)]`：`n_list = sorted({8, 16, math.ceil(math.sqrt(dim))})`，对每个 n 调 `build_algorithm_config(language, family, True, semantics, evolution_strategy=winner_strategy, evolution_points=(0.5,0.75), elimination_rate=0.25, de_factor=0.8, de_crossover=0.7, n_starts=n)`；label=`f"n{n}"`。不同 n → 不同 configuration_hash。 |
| `scripts/run_smco_evo_ablations.py` | 新增 `--dimension start_count` 路径：per-dim 循环（ceil√d 随 d），每 dim 用 `start_count_configs(winner, dim)` + `expand_tasks` 生成该 dim 的 tasks（`--instances-index` 关联 extra_starts hash），合并成 frozen `e6_ablations` manifest。 |
| `scripts/run_smco_evo_highdim_factorial.py` | `run_task_file`：`starts = load_starts(inst_dir, n_starts=int(task["n_starts"]))`；`_verify_provenance` 校验对应档 hash（n8 用 `starts.csv.gz`，其他用 `starts_n{N}.csv.gz` 的 sha）。 |

### 4.4 R 侧（`vendor/SMCO_R/main/highdim_instances.R` + `run_smco_evo_highdim_r.R`）

- `load_highdim_instance(inst_dir, ..., n_starts=8)`：n8 读 `starts.csv.gz`，其他读 `starts_n{N}.csv.gz`。
- R worker `run_smco_evo_highdim_r.R`：`n_starts <- as.integer(.task$n_starts)`（task 经统一 outcome 契约已嵌入，含 `n_starts`），传给 `load_highdim_instance`。
- 本机 R 端到端重验一档 n≠8（如 n16）确认选对文件。

### 4.5 关键约定

- **configuration_hash 含 n_starts**（`build_algorithm_config` 已有）：不同档 → 不同 hash → 不同 run_id，互不混淆；audit 的"非 EVO 伪重复"检查不会误判（algorithm_id 相同但 n_starts 不同 → configuration_hash 不同 → identity_key 含 configuration_hash? 否——见下）。

  > 注：`_identity_key`（merge_results）目前是 (function,dim,instance,algorithm_id,language,state_semantics,evolution_strategy,seed)。E6.1 不同 n_starts 档的 task 有**不同 seed**（`derive_seed` 不含 n_starts，但 configuration_hash 不同 → run_id 不同；而 seed 由 derive_seed(stage,suite,func,dim,instance,replication,algorithm_id) 决定，algorithm_id 相同 → seed 相同！）。因此同 (func,dim,instance,algorithm_id) 的 n8/n16/n_sqrt 三档会有**相同 identity_key** → 会被 audit 误判为 duplicate。

  **修正**：`_identity_key`（merge_results）增 `n_starts` 维度。**不动 `derive_seed`**——改 derive_seed 会让现有 E1/E2 manifest 的 seed 失效。identity_key 增 n_starts 即可让三档（同 algorithm_id、同 seed、不同 n_starts）区分开，audit 不误判。

### 4.6 测试策略（TDD，本机可测）

- `write_instance_artifacts(extra_starts={16: M})` → 写 `starts_n16.csv.gz` + metadata `extra_starts["16"]` hash 正确。
- `load_starts(dir, 16)` 读对文件；`load_starts(dir, 99)` 缺档报错；`load_starts(dir)`/`load_starts(dir,8)` 读 `starts.csv.gz`（兼容）。
- `_starts_seed(..., n_starts)` 含 n_starts → 不同 n 不同 seed。
- `start_count_configs(winner, 1000)` → 三档 n∈{8,16,32}，label `n8/n16/n32`，不同 n 不同 configuration_hash。
- `expand_tasks`：config.n_starts=16 → task.start_points_hash == entry.extra_starts["16"].hash；config.n_starts=8 → 不变。
- `derive_seed(..., n_starts)` → 不同 n 不同 seed（修正 4.5）。
- E6.1 manifest 生成端到端（per-dim expand + frozen）。
- factorial runner 按 `task.n_starts` 加载对应档 starts（端到端冒烟，d=4 小实例）。

## 5. 影响与兼容

- 改 6 个 Python 文件 + 2 个 R 文件；向后兼容（n8 artifact/manifest 不变）。
- 纯代码，本机可测，无 cocoex 依赖。
- **不动 `derive_seed`**：现有 E1/E2/E3/E6.2/E6.3 manifest 的 seed 字段保持有效（audit seed 检查不受影响）。E6.1 三档通过 `_identity_key` 增 n_starts 维度区分（见 4.5），不依赖 seed 区分。
- E6.1 task 携带 n_starts（config），worker/runner 据此选 starts 文件；现有 n8 路径完全不变。

## 6. 决策记录

- **方案 A（同 dir 多 starts 文件）**：transform 只存一次（d=5000 的 rotation_blocks 大），向后兼容 n8 的 `starts.csv.gz`。方案 B 重复 transform，方案 C 改动过大。
- **sqrt 档用具体 n 命名**（`starts_n32.csv.gz`）：文件名稳定、可读、与 metadata 的 n_starts 一致；不用 "sqrt" 字面量避免歧义。
- **dev instances**：E6 全程 dev（不碰 confirmatory），与 E6.2/E6.3 一致。
- **identity_key 增 n_starts**（不动 derive_seed）：避免破坏现有 manifest 的 seed，最小改动解决 audit 误判。

## 7. 实施顺序（概要，详见 writing-plans）

1. `write_instance_artifacts` extra_starts + `load_starts(n_starts)`（TDD）
2. `_starts_seed` 加 n_starts + generate `--extra-n-starts` + index entry
3. `expand_tasks` 按 config.n_starts 选 hash（TDD）
4. `start_count_configs(winner, dim)`（TDD）
5. E6.1 manifest 路径（run_smco_evo_ablations.py per-dim expand）
6. `_identity_key` 增 n_starts（merge_results，TDD）
7. factorial runner `load_starts(n_starts)` + provenance
8. R 侧 load_highdim_instance(n_starts) + worker 传 n_starts + 本机重验
9. 全量 pytest
