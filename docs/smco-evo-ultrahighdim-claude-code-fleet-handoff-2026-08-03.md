# SMCO-EVO 超高维实验 Claude Code / fleet 接手手册

日期：2026-08-03

范围：E3-F 420 runs + E7 1736 个 physically-new runs；本手册不包含旧 E3 的重跑。
原则：72 小时是 operational deadline，不是 kill timeout；超过 72 小时的任务必须继续到 FE budget、正常终止或明确基础设施失败。

## 1. 接手状态与硬 gate

代码已完成本地审查、真实 `d=10000` worker smoke 和全量测试。正式派跑前，Claude Code 必须在 coordinator 与每个执行节点记录同一个完整 40-hex `FROZEN_SHA`，不得从 dirty worktree 直接运行。

已验证：

- 全量测试：`633 passed`；
- governing selection `selection_hash=bcf87965006220a0` 可生成 420-task E3-F manifest；
- 实际 `d=10000` HighConditionedEllipsoid artifact + SignGD worker：成功，严格停在 32 FE；
- 缺少 R/rpy2 时 R-DEoptim 保存为明确的 `algorithm_failure / unsupported_dependency`，不再丢失 outcome；
- logical composite 会拒绝伪造数值、未通过 audit 的新来源、manifest/run-id 不匹配以及新旧算法 instance/starts 不一致。

正式 campaign 的硬 gate：

1. 每节点通过 Python/R 依赖预检；缺依赖时不得派正式 task；
2. E3-F/E7 instance index composer 验证所有 artifact 的实际 SHA-256；
3. manifest dry-run 的 errors 必须为空，任务数必须分别为 420/1736；
4. shard validator 必须通过，且 shard 内 task payload 与 frozen manifest 逐字节语义一致；
5. `--git-commit` 必须是正式冻结提交的完整 SHA，否则 dispatch 在派发前 fail-fast；
6. E3-F audit 未通过前不得构建 E3-F composite；E3-F composite 未通过前不得开始 E7 正式分析链。

## 2. Fleet 与目录约定

已知节点：

- `10.16.144.213`
- `10.16.144.214`
- `10.16.144.215`
- `10.16.144.217`
- `10.25.40.251`

用户名和认证由 operator 在外部配置；不要把密码写入脚本、日志、manifest 或本文档。

统一目录：

```bash
REPO=/amax/math/code/SMCO
EXT_ROOT=$REPO/result/smco-evo-ultrahighdim-2026
SELECTION=$REPO/result/e1-2026-07-30/selection/selection.json
FROZEN_SHA=$(git -C "$REPO" rev-parse HEAD)
```

每个节点的 checkout、instance artifact 和 shard 文件必须位于相同绝对路径。结果目录按节点隔离，例如 `e7/evidence_213`；禁止多个节点写同一 evidence root。

## 3. 依赖预检

```bash
cd "$REPO"
.venv/bin/python - <<'PY'
import numpy, scipy
import rpy2
print('numpy', numpy.__version__)
print('scipy', scipy.__version__)
print('rpy2', rpy2.__version__)
PY
Rscript -e 'cat(R.version$major,".",R.version$minor,"\n",sep=""); print(packageVersion("DEoptim")); print(packageVersion("nloptr"))'
```

冻结版本为 SciPy `1.17.1`、R `4.3.2`、rpy2 `3.6.4`、DEoptim `2.2-8`、nloptr `2.2.1`。若节点不满足，先安装并重做 smoke；不得用同名 Python 算法替代 R-DEoptim/STOGO。

建议同时记录：CPU、物理核、内存、BLAS、`pip freeze`、`R sessionInfo()` 和环境 hash。正式进程设置单线程 BLAS：

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

## 4. 生成并组合实例

只在 coordinator 生成一次，然后把相同 artifact 同步到 fleet。

```bash
cd "$REPO"
mkdir -p "$EXT_ROOT/e3f_instances" "$EXT_ROOT/e7_high_instances" "$EXT_ROOT/indexes"

.venv/bin/python scripts/generate_smco_evo_manifests.py \
  --stage instances --suite-stage extension_confirmatory \
  --out-dir "$EXT_ROOT/e3f_instances" \
  --functions Rosenbrock Levy Schwefel226 HighConditionedEllipsoid \
  --dims 200 500 1000 --n-instances 5 --n-starts 8

.venv/bin/python scripts/generate_smco_evo_manifests.py \
  --stage instances --suite-stage extension_confirmatory \
  --out-dir "$EXT_ROOT/e7_high_instances" \
  --functions Rastrigin Ackley Griewank Zakharov Rosenbrock Levy Schwefel226 HighConditionedEllipsoid \
  --dims 2000 3000 5000 10000 --n-instances 4 --n-starts 8

.venv/bin/python scripts/compose_smco_evo_ultrahighdim_instances.py \
  --campaign e3f --repo-root "$REPO" \
  --source "$EXT_ROOT/e3f_instances/instances_index.json" \
  --out "$EXT_ROOT/indexes/e3f_instances_index.json"

.venv/bin/python scripts/compose_smco_evo_ultrahighdim_instances.py \
  --campaign e7 --repo-root "$REPO" \
  --source "$REPO/instances_index_confirmatory.json" \
  --source "$EXT_ROOT/e3f_instances/instances_index.json" \
  --source "$EXT_ROOT/e7_high_instances/instances_index.json" \
  --out "$EXT_ROOT/indexes/e7_instances_index.json"
```

E7 composer 的语义不可改变：原四函数 d=1000 来自旧 E3 confirmatory index；新增四函数 d=1000 来自 E3-F；d=2000--10000 来自新的 extension-confirmatory index。

## 5. 冻结 manifest 与五节点 shard

```bash
mkdir -p "$EXT_ROOT/e3f" "$EXT_ROOT/e7"

.venv/bin/python scripts/run_smco_evo_ultrahighdim_extension.py manifest \
  --campaign e3f --selection "$SELECTION" \
  --instances-index "$EXT_ROOT/indexes/e3f_instances_index.json" \
  --out "$EXT_ROOT/e3f/manifest.json" --dry-run
.venv/bin/python scripts/run_smco_evo_ultrahighdim_extension.py manifest \
  --campaign e3f --selection "$SELECTION" \
  --instances-index "$EXT_ROOT/indexes/e3f_instances_index.json" \
  --out "$EXT_ROOT/e3f/manifest.json"
.venv/bin/python scripts/run_smco_evo_ultrahighdim_extension.py shard \
  --manifest "$EXT_ROOT/e3f/manifest.json" --n-shards 5 \
  --out "$EXT_ROOT/e3f/shards.json"

.venv/bin/python scripts/run_smco_evo_ultrahighdim_extension.py manifest \
  --campaign e7 --selection "$SELECTION" \
  --instances-index "$EXT_ROOT/indexes/e7_instances_index.json" \
  --out "$EXT_ROOT/e7/manifest.json" --dry-run
.venv/bin/python scripts/run_smco_evo_ultrahighdim_extension.py manifest \
  --campaign e7 --selection "$SELECTION" \
  --instances-index "$EXT_ROOT/indexes/e7_instances_index.json" \
  --out "$EXT_ROOT/e7/manifest.json"
.venv/bin/python scripts/run_smco_evo_ultrahighdim_extension.py shard \
  --manifest "$EXT_ROOT/e7/manifest.json" --n-shards 5 \
  --out "$EXT_ROOT/e7/shards.json"
```

检查 stdout：E3-F=`420`，E7=`1736`，两者 `errors=[]`。保留 manifest/shard SHA。先为每个 shard 执行一次 `dispatch --dry-run`，确认五 shard union 等于 manifest 且互不重叠。

## 6. Fleet 派发模板

建议固定映射：213→`shard-000`、214→`shard-001`、215→`shard-002`、217→`shard-003`、251→`shard-004`。E3-F 全部 audit pass 后再正式派 E7。每节点并发数先根据内存 smoke 决定；不要盲目按逻辑核全部拉满，尤其 R-DEoptim 的 512×d 种群与多份 R bridge 拷贝会叠加内存。

节点模板（以 E7/213 为例）：

```bash
cd "$REPO"
ENV_HASH=$(.venv/bin/python -c 'from smco.provenance import default_environment_hash; print(default_environment_hash())')
MACHINE_ID=$(hostname)

nohup .venv/bin/python scripts/run_smco_evo_ultrahighdim_extension.py dispatch \
  --manifest "$EXT_ROOT/e7/manifest.json" \
  --instance-root "$REPO" \
  --evidence-root "$EXT_ROOT/e7/evidence_213" \
  --shards "$EXT_ROOT/e7/shards.json" --shard-id shard-000 \
  --workers 4 --machine-id "$MACHINE_ID" \
  --git-commit "$FROZEN_SHA" --environment-hash "$ENV_HASH" \
  > "$EXT_ROOT/e7/dispatch_213.log" 2>&1 &
```

不要使用 shell `timeout`、scheduler walltime kill 或 72h 后 `pkill`。基础设施失败才允许同 run_id 新 attempt；恢复命令默认 resume，不要加 `--no-resume`。logical run 的 72h 时钟跨 attempt 累计，不会因 retry 重置。

## 7. 监控清单

每小时记录：

- 每节点 planned/running/success/algorithm_failure/infra_failure/stalled；
- 最新 heartbeat 时间与 FE 增长；
- 1h/6h/24h/72h sidecar 数；
- 超过 72h 仍运行的 run-id、算法、函数、维度、当前 FE/best；
- CPU/RSS、磁盘余量；
- retry 的原因、旧/new attempt id 和 supersedes 链。

可重复运行 `dispatch --dry-run` 查看 pending/retryable/stalled，但只有 heartbeat stale 且 FE 在规定窗口无增长时才允许自动恢复。算法慢、目标值不改善或超过 72h 都不是 infrastructure failure。

## 8. Merge、composite 与最终 index

E3-F 五节点 evidence root 不能用最后一个参数覆盖前四个。当前 merge 接受单 evidence root，因此先把五个互斥 run-id 目录汇集到 coordinator 的 `evidence_all/`（复制/rsync 时拒绝同名冲突），再执行：

```bash
.venv/bin/python scripts/run_smco_evo_ultrahighdim_extension.py merge \
  --manifest "$EXT_ROOT/e3f/manifest.json" \
  --evidence-root "$EXT_ROOT/e3f/evidence_all" \
  --out-dir "$EXT_ROOT/e3f/merged"

.venv/bin/python scripts/run_smco_evo_ultrahighdim_extension.py composite \
  --campaign e3f --selection-hash bcf87965006220a0 \
  --original-e3-valid "$REPO/result/e3-2026-07-31/merged_composite/valid_runs.csv" \
  --e3f-valid "$EXT_ROOT/e3f/merged/valid_runs.csv" \
  --e3f-manifest "$EXT_ROOT/e3f/manifest.json" \
  --e3f-audit "$EXT_ROOT/e3f/merged/provenance_audit.json" \
  --source-document "$REPO/result/e3-2026-07-31/e3_comparative_composite.json" \
  --materialized-out "$EXT_ROOT/e3f/e3_combined_valid_runs.csv" \
  --out "$EXT_ROOT/e3f/composite.json"

.venv/bin/python scripts/run_smco_evo_ultrahighdim_extension.py index \
  --campaign e3f --manifest "$EXT_ROOT/e3f/manifest.json" \
  --merged-dir "$EXT_ROOT/e3f/merged" --composite "$EXT_ROOT/e3f/composite.json" \
  --root "$REPO" --out "$EXT_ROOT/e3f/canonical_extension_index.json"
```

E7 收口：

```bash
.venv/bin/python scripts/run_smco_evo_ultrahighdim_extension.py merge \
  --manifest "$EXT_ROOT/e7/manifest.json" \
  --evidence-root "$EXT_ROOT/e7/evidence_all" \
  --out-dir "$EXT_ROOT/e7/merged"

.venv/bin/python scripts/run_smco_evo_ultrahighdim_extension.py composite \
  --campaign e7 --selection-hash bcf87965006220a0 \
  --e3-combined-valid "$EXT_ROOT/e3f/e3_combined_valid_runs.csv" \
  --e7-new-valid "$EXT_ROOT/e7/merged/valid_runs.csv" \
  --e7-manifest "$EXT_ROOT/e7/manifest.json" \
  --e7-audit "$EXT_ROOT/e7/merged/provenance_audit.json" \
  --source-document "$EXT_ROOT/e3f/composite.json" \
  --materialized-out "$EXT_ROOT/e7/e7_logical_valid_runs.csv" \
  --out "$EXT_ROOT/e7/composite.json"

.venv/bin/python scripts/run_smco_evo_ultrahighdim_extension.py index \
  --campaign e7 --manifest "$EXT_ROOT/e7/manifest.json" \
  --merged-dir "$EXT_ROOT/e7/merged" --composite "$EXT_ROOT/e7/composite.json" \
  --root "$REPO" --out "$EXT_ROOT/e7/canonical_extension_index.json"
```

最终必须满足：E3-F physical 420、E3 combined 840、E7 physical-new 1736、E7 logical 2016；所有 12 main checks 与 deadline-specific checks 全 PASS。未达到时不得进入统计、图表或论文 package。

## 9. 科学记录要求

- Schwefel 是标准盒内 2.26 + 盒外二次延拓，不能简称为未限定域的原始 Schwefel；
- R-DEoptim 的 NP 上限 512 和 STOGO 的剩余 FE 平衡分摊必须写入方法与 supplement；
- 72h 后完成的数据同时进入 eventual 表，deadline 表中保留其 72h snapshot；
- unsupported dependency 是部署失败，必须先修环境再重跑，不能当作算法失败参与排名；
- `algorithm_failure`、nonfinite 和超时均保留在分母，不得筛掉慢算法；
- 不同 CPU 的绝对 wall-time 不直接混排；主时间比较使用同 bundle/同节点 paired 结果，并报告节点信息。
