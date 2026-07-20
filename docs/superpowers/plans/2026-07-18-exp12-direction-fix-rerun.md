# 实验1/2 方向修正与最小化重跑 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **项目约束（覆盖 skill 默认）：** 不用 worktree（依赖 NFS 共享路径）；commit 节点执行前需用户确认（按惯例不自动提交）；实验脚本无 pytest 基础设施，用 `--quick` 冒烟验证方向。

**Goal:** 纠正实验1/2 的方向 bug（最小化），全量重跑 ~1050 task，每 task 实时输出 CSV，分布式部署到本机+251+215+213+217（~216 workers）。

**Architecture:** 本机编辑两处脚本（删 sense 取负分支 → `f=config.f`；`_run_parallel` 改 per-task append；实验1 加 `full` 档）。冒烟验证方向后，按 (dim,func) 切 shard 分发 5 机后台跑，断点续跑，NFS 即时可见 + 251 rsync 回传，合并出总 CSV。

**Tech Stack:** Python（smco 包，`/amax/math/code/SMCO/.venv`）、ProcessPoolExecutor、NFS 同构挂载（本机/217/213/215）、ssh 分发、rsync。

**Spec:** `docs/superpowers/specs/2026-07-18-exp12-direction-fix-rerun-design.md`

---

## File Structure

| 文件 | 责任 | 改动 |
|---|---|---|
| `scripts/run_evo_startsweep_comparison.py` | 实验1 起点缩减 | 删 sense 分支 + 加 full 档 + per-task flush |
| `scripts/run_evo_dimsplit_comparison.py` | 实验2 维度分离 | 删 sense 分支 + per-task flush |
| `result/evo-startsweep-2026-07-18/` | 实验1 产物（新） | 新建 |
| `result/evo-dimsplit-2026-07-18/` | 实验2 产物（新） | 新建 |
| 不变 | `src/smco/optimizer.py`、`src/smco/test_functions.py`、`vendor/SMCO_R/` | 不动（bug 在调用方） |

---

## Task 1: 实验1 脚本 — 方向修正 + full 基线 + per-task flush

**Files:**
- Modify: `scripts/run_evo_startsweep_comparison.py`

- [ ] **Step 1: FRACTIONS 增列 full（约 61 行）**

```python
# 旧
FRACTIONS = [0.5, 2.0 / 3.0, 0.75, 0.8]  # 1/2, 2/3, 3/4, 4/5
# 新
FRACTIONS = [1.0, 0.5, 2.0 / 3.0, 0.75, 0.8]  # full + 1/2, 2/3, 3/4, 4/5
```

- [ ] **Step 2: `_fraction_label` 处理 full（约 107-112 行）**

```python
# 新（在 for 循环前加 full 分支）
def _fraction_label(frac: float) -> str:
    # Stable human-readable tag: 1.0->full, 0.5->1/2, 0.667->2/3, 0.75->3/4, 0.8->4/5.
    if abs(frac - 1.0) < 1e-9:
        return "full"
    for num, den in ((1, 2), (2, 3), (3, 4), (4, 5)):
        if abs(frac - num / den) < 1e-9:
            return f"{num}/{den}"
    return f"{frac:.4f}"
```

- [ ] **Step 3: `_reduced_start_count` full 返回满（约 115-119 行）**

```python
# 新
def _reduced_start_count(base: int, frac: float) -> int:
    # full budget for frac==1.0; otherwise truncate so the reduced count is
    # strictly less than the full budget (frac < 1); never drop below 1.
    if abs(frac - 1.0) < 1e-9:
        return base
    k = int(base * frac)
    return max(1, k)
```

- [ ] **Step 4: 删 sense 取负分支（`_run_task` 约 207-210 行）**

```python
# 旧
        config = assign_config(func_name, dim)
        f = config.f
        if task["sense"] == "min":
            f = lambda x, _f=config.f: -_f(x)
# 新
        config = assign_config(func_name, dim)
        f = config.f  # config.f 已是 -raw（最大化目标）；最大化它 = 最小化 raw（正确方向）
```

- [ ] **Step 5: `_run_parallel` 改 per-task flush（约 248-261 行）**

```python
# 新（签名加 out_csv；每完成一个 task 立即 append + 打印）
def _run_parallel(tasks, workers, label, out_csv):
    total = len(tasks)
    done = 0
    print(f"[{label}] {total} tasks, {workers} workers", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_task, t) for t in tasks]
        for fut in as_completed(futures):
            r = fut.result()
            _append_csv([r], out_csv)  # 每 task 立即落盘
            done += 1
            print(f"[{label}] {r['func']}/{r['dim']}D rep{r['rep']} {r['algo']} "
                  f"fopt={r['fopt']:.6g} {r['time']:.0f}s ({done}/{total})", flush=True)
    print(f"[{label}] DONE {total} tasks in {time.perf_counter() - t0:.0f}s", flush=True)
```

- [ ] **Step 6: `run` 调用处同步改（约 302-307 行）**

```python
# 旧
        tasks = _generate_tasks(seed, dim, dim_funcs, n_reps)
        results = _run_parallel(tasks, workers, label)
        _append_csv(results, out_csv)
        for r in results:
            done_keys.add((r["strategy"], r["start_fraction"], r["algo"], r["func"], r["dim"], r["rep"]))
        print(f"[{label}] SAVED {len(results)} rows", flush=True)
# 新
        tasks = _generate_tasks(seed, dim, dim_funcs, n_reps)
        _run_parallel(tasks, workers, label, out_csv)
        done_keys = _existing_keys(out_csv)  # per-task 已写盘，从磁盘刷新
        print(f"[{label}] DONE (flushed per-task)", flush=True)
```

- [ ] **Step 7: 移除 BASELINE_CSV 依赖（约 71 行定义 + 约 326 行 main print）**

删除 `BASELINE_CSV = Path("result/highdim-full-comparison-2026-06-04/all_results.csv")` 这行；删除 `main()` 里 `print(f"Baseline (reused, not re-run): {BASELINE_CSV}", flush=True)` 这行。（保留 `from run_highdim_full_comparison import ...` 的 RNG plumbing。）

---

## Task 2: 实验2 脚本 — 方向修正 + per-task flush

**Files:**
- Modify: `scripts/run_evo_dimsplit_comparison.py`

- [ ] **Step 1: 删 sense 取负分支（`_run_task` 约 191-194 行）**

```python
# 旧
        config = assign_config(func_name, dim)
        f = config.f
        if task["sense"] == "min":
            f = lambda x, _f=config.f: -_f(x)
# 新
        config = assign_config(func_name, dim)
        f = config.f  # config.f 已是 -raw；最大化它 = 最小化 raw（正确方向）
```

- [ ] **Step 2: `_run_parallel` 改 per-task flush（约 230-243 行）**

```python
# 新
def _run_parallel(tasks, workers, label, out_csv):
    total = len(tasks)
    done = 0
    print(f"[{label}] {total} tasks, {workers} workers", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_task, t) for t in tasks]
        for fut in as_completed(futures):
            r = fut.result()
            _append_csv([r], out_csv)
            done += 1
            print(f"[{label}] {r['func']}/{r['dim']}D rep{r['rep']} {r['algo']} "
                  f"g{r['dim_groups']}/{r['strategy']} fopt={r['fopt']:.6g} "
                  f"{r['time']:.0f}s ({done}/{total})", flush=True)
    print(f"[{label}] DONE {total} tasks in {time.perf_counter() - t0:.0f}s", flush=True)
```

- [ ] **Step 3: `run` 调用处同步改（约 278-283 行）**

```python
# 旧
        tasks = _generate_tasks(seed, dim, dim_funcs, n_reps, groups, strategies)
        results = _run_parallel(tasks, workers, label)
        _append_csv(results, out_csv)
        for r in results:
            done_keys.add((r["strategy"], r["dim_groups"], r["algo"], r["func"], r["dim"], r["rep"]))
        print(f"[{label}] SAVED {len(results)} rows", flush=True)
# 新
        tasks = _generate_tasks(seed, dim, dim_funcs, n_reps, groups, strategies)
        _run_parallel(tasks, workers, label, out_csv)
        done_keys = _existing_keys(out_csv)
        print(f"[{label}] DONE (flushed per-task)", flush=True)
```

- [ ] **Step 4: 移除 BASELINE_CSV 依赖（约 72 行定义 + 约 304 行 main print）**

同 Task 1 Step 7，删除 `BASELINE_CSV` 定义行与 main 的 print 行。

---

## Task 3: 本机冒烟验证方向（改对了吗？）

**Files:** 无改动，仅验证。本机现 ~61 核空闲，1000D 秒级，直接本机跑。

- [ ] **Step 1: 实验1 冒烟（1000D Rastrigin，应得负 fopt）**

```bash
cd /amax/math/code/SMCO
.venv/bin/python scripts/run_evo_startsweep_comparison.py --quick --dims 1000 --funcs Rastrigin --workers 8 --output-dir /tmp/smoke-startsweep
```
Expected: CSV 里 fopt 为**负**且 |fopt| 小（如 -1 ~ -50，找到谷底）；对照旧反向结果同配置 fopt≈+2.9 万（推边界）。`full` 档 fopt 应 ≤ 缩减档。

- [ ] **Step 2: 实验2 冒烟（1000D Rastrigin，G=1 应负）**

```bash
.venv/bin/python scripts/run_evo_dimsplit_comparison.py --quick --dims 1000 --funcs Rastrigin --groups 1 2 --strategies rand1bin --workers 8 --output-dir /tmp/smoke-dimsplit
```
Expected: fopt 为负、|fopt| 小；G=1 与 G=2 都收敛到谷底。

- [ ] **Step 3: 确认方向铁证**

```bash
echo "新(最小化,应为负小):"; tail -3 /tmp/smoke-startsweep/startsweep_results.csv
echo "旧(反向,应为正大):"; awk -F',' 'NR>1 && $4=="Rastrigin" && $5==1000 {print $7}' /amax/math/code/SMCO/result/evo-startsweep-2026-06-15/startsweep_results.csv | head -3
```
Expected: 新 fopt 负且绝对值远小于旧的正值。若新仍为正大值 → 方向没改对，回 Task 1/2 检查。

- [ ] **Step 4: commit（待用户确认）**

改动两脚本 + spec + 本计划。等用户点头再 commit。

---

## Task 4: 251 Python 环境准备（251 此前只跑 R，无 venv/smco）

**Files:** 无项目文件改动，仅在 251 建环境。

- [ ] **Step 1: 探测 251 Python 现状**

```bash
ssh 10.25.40.251 'which python3; python3 --version; ls ~/code/SMCO/.venv 2>/dev/null || echo "no venv"; ls ~/code/SMCO/src/smco 2>/dev/null || echo "no smco src"'
```
Expected: 有 python3；无 .venv、无 src/smco（需建）。

- [ ] **Step 2: rsync 代码到 251（home=/data/math，非 NFS）**

```bash
rsync -av --exclude=.git --exclude=result --exclude=.venv --exclude=__pycache__ \
  /amax/math/code/SMCO/ 10.25.40.251:~/code/SMCO/
```

- [ ] **Step 3: 251 建 venv + 装 smco**

```bash
ssh 10.25.40.251 'cd ~/code/SMCO && python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install numpy && .venv/bin/pip install -e .'
```
Expected: `Successfully installed smco-...`。验证 `ssh 10.25.40.251 '~/code/SMCO/.venv/bin/python -c "from smco import smco_evo; print(\"ok\"))"'` 输出 ok。

---

## Task 5: 部署启动 5 机（每机实验1+2 各一 nohup，按核分 workers）

**分配表（workers 之和 ≤ 各机空闲核）：**

| 机器 | 实验1 shard | 实验2 shard | w1 | w2 |
|---|---|---|---|---|
| 本机(64) | 5000D Rosenbrock | 5000D Rosenbrock | 28 | 28 |
| 251(48) | 5000D Rastrigin | 5000D Rastrigin | 22 | 22 |
| 215(46) | 5000D Ackley | 5000D Ackley | 20 | 20 |
| 213(46) | 3000D 全3函数 | 3000D 全3函数 | 20 | 20 |
| 217(40) | 1000D 全3函数 | 1000D 全3函数 | 18 | 18 |

- [ ] **Step 1: 本机启动（5000D Rosenbrock，实验1+2）**

```bash
cd /amax/math/code/SMCO
mkdir -p result/evo-startsweep-2026-07-18/shard-local result/evo-dimsplit-2026-07-18/shard-local
nohup .venv/bin/python scripts/run_evo_startsweep_comparison.py --dims 5000 --funcs Rosenbrock --workers 28 --output-dir result/evo-startsweep-2026-07-18/shard-local > result/evo-startsweep-2026-07-18/shard-local/run.log 2>&1 &
nohup .venv/bin/python scripts/run_evo_dimsplit_comparison.py --dims 5000 --funcs Rosenbrock --workers 28 --output-dir result/evo-dimsplit-2026-07-18/shard-local > result/evo-dimsplit-2026-07-18/shard-local/run.log 2>&1 &
```

- [ ] **Step 2: 251 启动（5000D Rastrigin）**

```bash
ssh 10.25.40.251 'cd ~/code/SMCO && mkdir -p result/evo-startsweep-2026-07-18/shard-251 result/evo-dimsplit-2026-07-18/shard-251 && nohup .venv/bin/python scripts/run_evo_startsweep_comparison.py --dims 5000 --funcs Rastrigin --workers 22 --output-dir result/evo-startsweep-2026-07-18/shard-251 > result/evo-startsweep-2026-07-18/shard-251/run.log 2>&1 & nohup .venv/bin/python scripts/run_evo_dimsplit_comparison.py --dims 5000 --funcs Rastrigin --workers 22 --output-dir result/evo-dimsplit-2026-07-18/shard-251 > result/evo-dimsplit-2026-07-18/shard-251/run.log 2>&1 &'
```

- [ ] **Step 3: 215 启动（5000D Ackley）**

```bash
ssh 10.16.144.215 'cd ~/code/SMCO && mkdir -p result/evo-startsweep-2026-07-18/shard-215 result/evo-dimsplit-2026-07-18/shard-215 && nohup .venv/bin/python scripts/run_evo_startsweep_comparison.py --dims 5000 --funcs Ackley --workers 20 --output-dir result/evo-startsweep-2026-07-18/shard-215 > result/evo-startsweep-2026-07-18/shard-215/run.log 2>&1 & nohup .venv/bin/python scripts/run_evo_dimsplit_comparison.py --dims 5000 --funcs Ackley --workers 20 --output-dir result/evo-dimsplit-2026-07-18/shard-215 > result/evo-dimsplit-2026-07-18/shard-215/run.log 2>&1 &'
```

- [ ] **Step 4: 213 启动（3000D 全函数）**

```bash
ssh 10.16.144.213 'cd ~/code/SMCO && mkdir -p result/evo-startsweep-2026-07-18/shard-213 result/evo-dimsplit-2026-07-18/shard-213 && nohup .venv/bin/python scripts/run_evo_startsweep_comparison.py --dims 3000 --workers 20 --output-dir result/evo-startsweep-2026-07-18/shard-213 > result/evo-startsweep-2026-07-18/shard-213/run.log 2>&1 & nohup .venv/bin/python scripts/run_evo_dimsplit_comparison.py --dims 3000 --workers 20 --output-dir result/evo-dimsplit-2026-07-18/shard-213 > result/evo-dimsplit-2026-07-18/shard-213/run.log 2>&1 &'
```

- [ ] **Step 5: 217 启动（1000D 全函数）**

```bash
ssh 10.16.144.217 'cd ~/code/SMCO && mkdir -p result/evo-startsweep-2026-07-18/shard-217 result/evo-dimsplit-2026-07-18/shard-217 && nohup .venv/bin/python scripts/run_evo_startsweep_comparison.py --dims 1000 --workers 18 --output-dir result/evo-startsweep-2026-07-18/shard-217 > result/evo-startsweep-2026-07-18/shard-217/run.log 2>&1 & nohup .venv/bin/python scripts/run_evo_dimsplit_comparison.py --dims 1000 --workers 18 --output-dir result/evo-dimsplit-2026-07-18/shard-217 > result/evo-dimsplit-2026-07-18/shard-217/run.log 2>&1 &'
```

- [ ] **Step 6: 启动后 5 分钟核查所有进程在跑**

```bash
for ip in 10.25.40.251 10.16.144.215 10.16.144.213 10.16.144.217; do echo "=== $ip ==="; ssh $ip 'ps -eo stat,%cpu,cmd --no-headers | grep run_evo | grep -v grep | head'; done
```
Expected: 每机 2 个 python 进程（实验1+2），stat=R，%cpu 累加 ≈ workers。本机同理 `ps -eo stat,%cpu,cmd --no-headers | grep run_evo | grep -v grep`。

---

## Task 6: 监控 + 回传合并

- [ ] **Step 1: 监控进度（随时看 CSV 增长，无需翻日志）**

```bash
echo "=== 实验1 各 shard 行数 ==="; wc -l /amax/math/code/SMCO/result/evo-startsweep-2026-07-18/shard-*/startsweep_results.csv 2>/dev/null
echo "=== 实验2 各 shard 行数 ==="; wc -l /amax/math/code/SMCO/result/evo-dimsplit-2026-07-18/shard-*/dimsplit_results.csv 2>/dev/null
```
（本机/217/213/215 的 shard 在 NFS，本机直接可见；251 的需 ssh。）

- [ ] **Step 2: 251 shard rsync 回传（251 非 NFS）**

```bash
rsync -av 10.25.40.251:~/code/SMCO/result/evo-startsweep-2026-07-18/shard-251/ /amax/math/code/SMCO/result/evo-startsweep-2026-07-18/shard-251/
rsync -av 10.25.40.251:~/code/SMCO/result/evo-dimsplit-2026-07-18/shard-251/ /amax/math/code/SMCO/result/evo-dimsplit-2026-07-18/shard-251/
```

- [ ] **Step 3: 合并各 shard 为总 CSV（去重 header）**

```bash
cd /amax/math/code/SMCO/result/evo-startsweep-2026-07-18
head -1 $(ls shard-*/startsweep_results.csv | head -1) > startsweep_results.csv
for f in shard-*/startsweep_results.csv; do tail -n +2 "$f"; done >> startsweep_results.csv
wc -l startsweep_results.csv  # 期望 ~406 (1 header + 405)
# 实验2 同理（期望 ~649）
cd ../evo-dimsplit-2026-07-18
head -1 $(ls shard-*/dimsplit_results.csv | head -1) > dimsplit_results.csv
for f in shard-*/dimsplit_results.csv; do tail -n +2 "$f"; done >> dimsplit_results.csv
wc -l dimsplit_results.csv
```

---

## Task 7: 验证 + 分析报告

- [ ] **Step 1: 方向铁证（全量）**

实验1 总 CSV 所有 fopt 应为负（最小化）；对照旧 `-2026-06-15` 同配置正值。

- [ ] **Step 2: 实验1 full 基线检查**

`full` 档 fopt（按 func/dim 均值）应 ≤ 所有缩减档（满起点不弱于缩减）。

- [ ] **Step 3: 实验2 G=1 锚点自洽**

G=1 同 seed 可复现；画 G∈{1,2,4,8} 的 fopt 对比，看维度分离是否增益。

- [ ] **Step 4: 生成分析报告**

参照旧 `evo-startsweep-2026-06-15/analysis.md`、`evo-dimsplit-*-2026-06-15/analysis.md` 风格，用新最小化数据生成 `result/evo-startsweep-2026-07-18/analysis.md`、`result/evo-dimsplit-2026-07-18/analysis.md`（fopt 越小越优）。

- [ ] **Step 5: commit 全部产物（待用户确认）**

代码 + 产物 + 报告。等用户点头。

---

## Self-Review

- **Spec 覆盖**：spec §3→Task 1/2；§4→Task 1 Step5/Task 2 Step2；§5→Task 4/5/6；§6→Task 3/7；§7 工作量预估隐含于 Task 5 分配；§8 YAGNI 体现在"不变"清单。✅ 无遗漏。
- **占位符**：无 TBD/TODO，所有改动有完整代码，所有命令可执行。✅
- **类型一致**：`_run_parallel` 新签名 `(tasks, workers, label, out_csv)` 在 Task 1/2 定义、Task 1 Step6/Task 2 Step3 调用一致；`_append_csv([r], out_csv)` 单行 append 与现有 `_append_csv` 签名（`rows: list[dict]`）兼容。✅
