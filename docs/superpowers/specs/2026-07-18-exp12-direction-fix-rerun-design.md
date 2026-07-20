# 实验1/2 方向修正与最小化重跑设计

- **日期**：2026-07-18
- **状态**：已批准（用户确认 2026-07-18），待实现
- **背景**：实验1/2 因方向 bug（`_run_task` 对 `sense=="min"` 多取一次负），实际在做"最大化 raw = 推向边界"，与真正的最小化相反。详见 `docs/direction-bug-2026-06-15.md`。现纠正方向并全量重跑最小化版。

---

## 1. 目标

1. **纠正方向**：实验1/2 改为真正的最小化（`fopt = -raw`，越小越接近真解）。
2. **实时输出**：每完成一个 task 立即写 CSV 一行，长尾期也能随时查看结果、无需翻日志。
3. **分布式重跑**：按服务器性能分配，~240 核（去 216、加本机）。
4. **自带基线**：实验1 加 `full` 满起点档、实验2 用 `G=1` 作锚点，摆脱对反向 `all_results.csv` 的依赖。

## 2. 范围（全量重跑）

方向反了不只是符号问题——最小化找谷底 vs 最大化推边界，收敛点与路径完全不同，旧的反向 fopt 无法复用。故全量重跑：

| 实验 | 网格 | task 数 |
|---|---|---|
| 实验1 startsweep | 3 evo 变体 × {full,1/2,2/3,3/4,4/5} × {1000,3000,5000}D × {Rastrigin,Ackley,Rosenbrock} × reps{5,2,2} | ≈ 405 |
| 实验2 dimsplit | 3 evo 变体 × {1,2,4,8} 组 × {rand1bin,best1bin} × {1000,3000,5000}D × {Rastrigin,Ackley,Rosenbrock} × reps{5,2,2} | ≈ 648 |
| **合计** | | **~1050** |

- seed=21，维度/函数/reps 配置不变。
- **不含** GenSA/SA/DEoptim/GA/PSO。最慢是 5000D BR_EVO ~22-30h（非实验3 的 28 天）。

## 3. 设计 A：方向修正（本机编辑，零 CPU）

### A.1 实验1 `scripts/run_evo_startsweep_comparison.py`
- **`_run_task`（约 208-210 行）**：删除
  ```python
  f = config.f
  if task["sense"] == "min":
      f = lambda x, _f=config.f: -_f(x)
  ```
  改为单行 `f = config.f`（`config.f` 已是 `-raw` 最大化目标）。
- **`FRACTIONS`（约 61 行）**：增列 `1.0`。
- **`_fraction_label`（约 107 行）**：1.0 → `"full"`。
- **`_reduced_start_count`（约 115 行）**：`frac==1.0` 时返回 `base`（满起点 `n_starts`）。
- **移除 `BASELINE_CSV` 依赖**（保留 `_make_batch_rng`/`_n_starts` 等 RNG plumbing，保证 starts 可复现）。`full` 行用 `starts_full` 全部行，与缩减档同矩阵前缀配对。

### A.2 实验2 `scripts/run_evo_dimsplit_comparison.py`
- **`_run_task`（约 191-194 行）**：同 A.1，删 sense 分支，`f = config.f`。
- **`G=1` 已是满起点 evo 锚点**（dim_groups=1 = 整体进化 = 基准 evo），无需加 full。
- 移除 `BASELINE_CSV` 依赖。
- **不改** `optimizer.py`、`test_functions.py`、`SMCO_evo.R`（bug 在调用方）。

## 4. 设计 B：每 task 实时 flush

`_run_parallel` 改为每完成一个 task 立即落盘：

```python
for fut in as_completed(futures):
    r = fut.result()
    _append_csv([r], out_csv)                       # 每完成一个立即写
    done_keys.add((r["strategy"], r.get("start_fraction", r.get("dim_groups")),
                   r["algo"], r["func"], r["dim"], r["rep"]))
    done += 1
    print(f"[{label}] {r['func']}/{r['dim']}D rep{r['rep']} {r['algo']} "
          f"fopt={r['fopt']:.6g} {r['time']:.0f}s ({done}/{total})", flush=True)
```

- 主进程单线程收集 `as_completed` → append 天然串行，**无需文件锁**。
- 删除旧的"整个 dim 批量 append"逻辑。
- CSV 随时最新 + 日志每 task 一行，双保险。

## 5. 设计 C：服务器编排（去 216、加本机，~240 核）

### C.1 资源与分配
| 服务器 | 可用核 | workers | 分配（实验1+2 合计） |
|---|---|---|---|
| **本机** | ~61（64 核） | 56 | 5000D **Rosenbrock**（最慢，强耦合） |
| **251** | 48 全空 | 44 | 5000D **Rastrigin** |
| **215** | ~46（2 核跑 Rosenbrock 复现） | 40 | 5000D **Ackley** |
| **213** | ~46 | 40 | 3000D 全部 3 函数 |
| **217** | ~40 | 36 | 1000D 全部 + 3000D 分担（动态） |

- 216 移除：ssh key 未配，省去配置。
- 本机最强且代码/venv 现成（`/amax/math/code/SMCO/.venv`），**免 rsync 分发、结果直接落本机磁盘免回传**。
- 切分原则：5000D 三函数 → 本机/251/215 各扛一个；3000D → 213 主力、217 先冒烟再分担；1000D → 217。允许按实际进度动态再分配。

### C.2 shard 切分与启动
按 (dim, func) 切，每台一个 shard，独立 output 子目录（避免多机写同一 CSV 冲突）：
- `result/evo-startsweep-2026-07-18/shard-<host>/startsweep_results.csv`
- `result/evo-dimsplit-2026-07-18/shard-<host>/dimsplit_results.csv`

启动（各机 cwd = `~/code/SMCO`，本机 = `/amax/math/code/SMCO`，NFS 同构）：
```bash
nohup /amax/math/code/SMCO/.venv/bin/python scripts/run_evo_startsweep_comparison.py \
  --dims 5000 --funcs Rosenbrock --workers 56 \
  --output-dir result/evo-startsweep-2026-07-18/shard-local \
  > result/evo-startsweep-2026-07-18/shard-local/run.log 2>&1 &
```
实验2 同理（`--dims/--funcs/--groups/--strategies`）。各机经 ssh 启动，本机直接启动。

### C.3 断点续跑
`_existing_keys` 已支持：重启自动跳过已完成 task（key = strategy / fraction-or-groups / algo / func / dim / rep）。

### C.4 回传合并
各 shard 完成后 rsync 回本机（本机 shard 免传），合并为总 CSV：
- `result/evo-startsweep-2026-07-18/startsweep_results.csv`（~405 行）
- `result/evo-dimsplit-2026-07-18/dimsplit_results.csv`（~648 行）

## 6. 设计 D：产物 & 验证

### D.1 产物（新目录，旧反向结果保留并存）
- `result/evo-startsweep-2026-07-18/`：实验1 最小化结果
- `result/evo-dimsplit-2026-07-18/`：实验2 最小化结果
- 旧 `-2026-06-15`/`-2026-06-18` 目录不动。

### D.2 验证（成功标准）
1. **冒烟**（217，改代码后先跑）：`--quick`（1000D Rastrigin 1 rep）→ fopt 为**负**且 |fopt| 小（找到谷底 raw≈0 附近，而非推边界的 +2.9 万）。
2. **方向铁证**：同配置新 fopt（负、小）vs 旧反向 fopt（正、~万）符号相反、量级悬殊。
3. **实验2 G=1 锚点**：G=1 行同 seed 可复现、逐点自洽。
4. **实验1 full 行**：full（满起点）fopt ≤ 任何缩减档（满起点不弱于缩减）。
5. **实时性**：跑期间 `tail -f` CSV 可见逐行增长，无需等整 dim。

## 7. 工作量预估

瓶颈 = 5000D BR_EVO 单 task ~22-30h。三个 5000D 函数由本机/251/215 独立并行 → 大部分 **~1-1.5 天**完成，长尾 BR_EVO 收官约 **~1.5-2 天**。

## 8. 不做（YAGNI）

- 不修 `optimizer.py` / `test_functions.py` / `SMCO_evo.R`（bug 在调用方）。
- 不重跑 `all_results.csv` 基准（脚本自带基线）。
- 不含 GenSA/SA 等慢算法。
- 不覆盖旧反向结果（并存）。
- 不动实验3（R，已用正确方向）。

## 9. 关键文件

- 改：`scripts/run_evo_startsweep_comparison.py`（A.1 + B）
- 改：`scripts/run_evo_dimsplit_comparison.py`（A.2 + B）
- 新增产物：`result/evo-startsweep-2026-07-18/`、`result/evo-dimsplit-2026-07-18/`
- 参考：`docs/direction-bug-2026-06-15.md`（bug 根因）
- 不变：`src/smco/optimizer.py`、`src/smco/test_functions.py`、`vendor/SMCO_R/`
