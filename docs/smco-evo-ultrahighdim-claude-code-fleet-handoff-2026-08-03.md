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

权威服务器清单为 `docs/smco-fleet-servers.md`。本 campaign 使用十一台机器：

- 本机 coordinator/worker：`/amax/math/code/SMCO`，24 核；
- `m213/m214/m215/m217`：`~/code/SMCO`；
- `10.25.40.251`：`/data/math/code/SMCO`，48 核，高维主力；
- `m253`：`~/code/SMCO`，48 核，Python/cocoex 已部署，正式 E7 前仍需补齐 R 合同；
- `zf129/zf132/zf133/zf852`：`~/code/SMCO`，每台 128 逻辑核、64 物理核、503 GB 内存。

用户名和认证由 operator 在外部配置；不要把密码写入脚本、日志、manifest 或本文档。

统一目录：

```bash
REPO=/amax/math/code/SMCO     # 内网远端/zftest 用 $HOME/code/SMCO；251 用 /data/math/code/SMCO
EXT_ROOT=$REPO/result/smco-evo-ultrahighdim-2026
SELECTION=$REPO/result/e1-2026-07-30/selection/selection.json
FROZEN_SHA=$(git -C "$REPO" rev-parse HEAD)
```

十一台机器不共享 NFS，不能用本机文件是否存在判断远端完成情况。必须以 rsync 分发并回收 `src/`、`scripts/`、`vendor/`、manifest、shards 和 instances。各节点绝对路径可以不同，但 repo-relative artifact 布局和文件 SHA-256 必须相同。结果目录按节点和 shard 隔离，例如 `e7/evidence_213_s002`；禁止多个 dispatch 写同一 evidence root。

zftest 的 evidence 不放在接近满载的 root 盘：zf129/zf133/zf852 使用 `/data/zftest/result/smco-evo-ultrahighdim-2026`，无 `/data` 的 zf132 使用 `~/result/smco-evo-ultrahighdim-2026`。实例仍同步到 repo-relative 路径，以便 manifest 中的 artifact 路由保持一致。

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

当前代码冻结 SciPy `1.17.1`、R `4.3.2`、rpy2 `3.6.4`、DEoptim `2.2-8`、nloptr `2.2.1`，但权威 fleet 文档显示 9 台远端 R 节点已经统一为 R `4.5.2`；本机仍是 R 4.3.2 且缺 R 包/rpy2，253 尚未具备 R 环境。**推荐在生成 manifest 前把 E7 R adapter 合同统一改冻到 R 4.5.2**，补测试并形成新的 40-hex `FROZEN_SHA`，再给本机和 253 部署同版本。若坚持 4.3.2，则需在全部十一台节点部署隔离环境。两种方案只能选一种，不能在同一 manifest 下混跑。

> **更新（code/environment gate 执行，2026-08-03）**：E7 R adapter 合同已统一改冻到 R `4.5.2`——`src/smco/e7_algorithm_adapters.py` 的 `_Rpy2Backend._R_VERSION`、`R-DEoptim`/`STOGO` 两条 `rng` metadata 与 ImportError 错误消息均已改为 `R 4.5.2`，并由 `tests/test_e7_algorithm_adapters.py::test_r_contract_freezes_r_version_4_5_2` 守护；同时保持 SciPy `1.17.1`、rpy2 `3.6.4`、DEoptim `2.2-8`、nloptr `2.2.1`，全量 pytest `635 passed`。本次 gate 提交的完整 HEAD SHA 即为正式 campaign 的 `FROZEN_SHA`。本机与 253 仍须在 preflight 阶段部署 R 4.5.2 + DEoptim + nloptr + rpy2 才能跑 R-DEoptim/STOGO；E3-F 的七个算法为纯 Python，不依赖 R。

> **更新（scipy 合同 gate，2026-08-03）**：preflight 实测发现 10 台远端 scipy 已是 `1.18.0`（非原冻结合同的 `1.17.1`），且 E3-F 的 DE/SA/GenSA/SMCO 经 `scipy.optimize` / `scipy.stats.qmc`，跨节点版本不一致会破坏数值可比性。经决策，scipy 合同统一改冻到 `1.18.0`（`L-BFGS` `package_version` + 新增 `tests/test_e7_algorithm_adapters.py::test_python_contract_freezes_scipy_1_18_0` 守护）；本机 numpy 升级 `2.4.6→2.5.1`、scipy `1.17.1→1.18.0`，全量 pytest `636 passed`。**本次提交的新 HEAD SHA 取代之前的 gate SHA，成为正式 campaign 的 `FROZEN_SHA`**。其它合同（R 4.5.2 / rpy2 3.6.4 / DEoptim 2.2-8 / nloptr 2.2.1）不变。

服务器文档所称“5 个 R 包”不包含 STOGO 所需的 `nloptr`；每节点还必须实测安装 `nloptr==2.2.1` 和 Python `rpy2==3.6.4`。任何一个 full-bundle 节点缺少 R/DEoptim/nloptr/rpy2 时不得领取正式 shard，也不得把 `unsupported_dependency` 当作算法性能结果。不得用同名 Python算法替代 R-DEoptim/STOGO。

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

## 5. 冻结 manifest 与十一节点 shard

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
  --manifest "$EXT_ROOT/e3f/manifest.json" --n-shards 42 \
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
  --manifest "$EXT_ROOT/e7/manifest.json" --n-shards 42 \
  --out "$EXT_ROOT/e7/shards.json"
```

检查 stdout：E3-F=`420`，E7=`1736`，两者 `errors=[]`。保留 manifest/shard SHA。先为每个 shard 执行一次 `dispatch --dry-run`，确认 42 shard union 等于 manifest 且互不重叠。

42 shards 在十一台机器间按能力加权：四台 zftest 各 6 份，251/253 各 4 份，其余节点各 2 份。四台 zftest 因而承担 E7 的 1104/1736 tasks（约 64%），与其 512 个逻辑核的地位相符，同时保留足够小的迁移单位。Shard 一旦开始运行不得拆分；只能整体迁移尚未启动的 shard。若 pilot 产生 bundle cost 文件，生成 E7 shards 时增加 `--cost-estimates <pilot-costs.json>`；否则默认 cost 仍保持 problem bundle 完整并按维度 FE 近似平衡。

## 6. Fleet 派发模板

E3-F 与 E7 使用相同的初始映射，但各自具有独立的 `shards.json`：

| 节点 | 初始 shards | E3-F tasks | E7 tasks | dispatch 数 × 每 dispatch workers | 初始总并发 | 硬限制/说明 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 本机 | `000–001` | 14 | 72 | `2×5` | 10 | 用户硬上限为 24 核的 1/2，即最多 12；预留 2 核额度给 coordinator/merge |
| 213 | `002–003` | 14 | 72 | `2×3` | 6 | 非共享 NFS，单独同步/回收 |
| 214 | `004–005` | 14 | 68 | `2×3` | 6 | key 免密已就绪，仍须做真实 R smoke |
| 215 | `006–007` | 14 | 68 | `2×5` | 10 | R 高维经验节点 |
| 217 | `008–009` | 14 | 58 | `2×4` | 8 | 先检查当前负载 |
| 251 | `010–013` | 28 | 130 | `4×6` | 24 | 48 核高维主力；`/data/math/code/SMCO` |
| 253 | `014–017` | 28 | 164 | `4×6` | 24 | 48 核；补齐 R/nloptr/rpy2 后启动 |
| zf129 | `018–023` | 70 | 246 | `6×10` | 60 | 用户硬上限为 128 逻辑核的 3/5，即最多 76；同时不超过 64 物理核 |
| zf132 | `024–029` | 84 | 246 | `6×10` | 60 | 用户硬上限同为最多 76；无 `/data`，输出放 `~/result` |
| zf133 | `030–035` | 70 | 294 | `6×10` | 60 | 64 物理核，503 GB |
| zf852 | `036–041` | 70 | 318 | `6×10` | 60 | 64 物理核，503 GB |

上述行精确合计 E3-F `420` tasks、E7 `1736` tasks。E7 每 shard 默认 cost 约 `1.85--2.04×10^8 FE`，四台 zftest 各承担约 `1.11×10^9 FE`。初始总并发为 328；其中本机 10≤12，zf129/zf132 各 60≤76，满足用户给定上限。上限按整台机器所有 SMCO 进程合计计算，不是每个 dispatch 单独计算。真实 d=10000 R-DEoptim RSS smoke 后可以下调；任何时候不得上调越过表中硬限制。E3-F 全部 audit pass 后再正式派 E7。

节点模板（以 E7/213 的 `shard-002` 为例；`shard-003` 使用独立 evidence/log 并另起一个 `--workers 3` dispatch）：

```bash
cd "$REPO"
ENV_HASH=$(.venv/bin/python -c 'from smco.provenance import default_environment_hash; print(default_environment_hash())')
MACHINE_ID=$(hostname)

nohup .venv/bin/python scripts/run_smco_evo_ultrahighdim_extension.py dispatch \
  --manifest "$EXT_ROOT/e7/manifest.json" \
  --instance-root "$REPO" \
  --evidence-root "$EXT_ROOT/e7/evidence_213_s002" \
  --shards "$EXT_ROOT/e7/shards.json" --shard-id shard-002 \
  --workers 3 --machine-id "$MACHINE_ID" \
  --git-commit "$FROZEN_SHA" --environment-hash "$ENV_HASH" \
  > "$EXT_ROOT/e7/dispatch_213_s002.log" 2>&1 &
```

不要使用 shell `timeout`、scheduler walltime kill 或 72h 后 `pkill`。基础设施失败才允许同 run_id 新 attempt；恢复命令默认 resume，不要加 `--no-resume`。logical run 的 72h 时钟跨 attempt 累计，不会因 retry 重置。

动态接管规则：只允许接管 `dispatch --dry-run` 显示全部 pending、且原节点确认从未启动的整个 shard。记录 `old_assignment/new_assignment/reason/timestamp`。已经产生 attempt ledger 的 run-id 只能按 retry 合同迁移，不能当作全新 task 重新派发。

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

E3-F 十一节点、42 个 evidence root 不能用最后一个参数覆盖前面的目录。当前 merge 接受单 evidence root，因此先把 42 个互斥 shard 的 run-id 目录汇集到 coordinator 的 `evidence_all/`（复制/rsync 时拒绝同名冲突；`_task_cache` 不作为结果合并依据），再执行：

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
