# SMCO-EVO 高维论文：fleet 正式实验运行手册（2026-08-02）

## 1. 目的、范围与结论

本文把当前空闲的 213/215/217 等机器转化为**有边界的、可审计的工作**。它只规定
已冻结问题的复跑/正式化，不允许根据当前结果新增算法、预算、函数、维度、实例、种子或
选择赢家。CC 应按本文的 gate 和停止条件调度；没有通过前置 gate 的项目不得“先跑再说”。

当前检查结果：

| 项目 | 当前状态 | 是否可立刻派跑 |
| --- | --- | --- |
| E6.3 schedule | 旧 manifest 216 task、189 unique `run_id`，现存 raw 61 条 | **否**：先冻结去重 manifest，并补齐/确认允许的实例索引 |
| E4 BBOB-largescale | 有 2,520-task 冻结 manifest；只有 development CSV | **否**：现 runner 不是 task-level outcome runner，不能分片 merge/audit |
| E5 low-dim | 有 480-task 冻结 manifest；只有 development CSV | **否**：同 E4；且论文优先级最低 |

所以空闲机器的第一批工作不是盲目计算，而是跑每台机器的环境/COCO预检和冻结输入
验证（不产生论文结果）；在 E6 gate 关闭后运行 E6.3；E4/E5 在专用接口合入并冻结后
再运行。不得用 E4/E5 现有 development CSV 替代正式结果。

## 2. 全局不可变规则

1. 所有正式结果使用单独的新目录，绝不写入或覆盖
   `result/e6-2026-07-31/schedule/raw/`、`result/e4-2026-07-31/`、
   `result/e5-2026-07-31/` 的旧结果。
2. 每台机器使用本地工作目录，例如
   `/local/smco-evo-20260802/<campaign>/raw_<machine>/`。不要假定 `/amax/math`
   是共享 NFS；传回 coordinator 后才 merge。
3. 每次派发前固定完整 40-hex `GIT_SHA`，每机独立计算 `ENV_HASH`，并把
   `machine_id`、`git_commit`、`environment_hash` 传给 worker。`--confirmatory`
   的 fail-fast 若拒绝，停止而不是删除字段重试。
4. 所有分片只能以 frozen manifest 的 `run_id` 集合切分。禁止以“缺少某算法”或
   “表现差”做条件筛选。每片开始均先 `--dry-run`、再 `--validate-only`。
5. 失败、超时、NaN 都保留为 outcome，不能静默重抽种子。重试仅使用同一 run-id，
   把新 attempt 留在新 raw 目录，由 merge 的 supersede 规则处理。
6. 停止进程时不得用会匹配自身 shell 的 `pkill -f run_smco...`。先
   `pgrep -af '[r]un_smco_evo_highdim_factorial.py'` 人工核对 PID；只对已核对的
   单个 PID 执行 `kill <pid>`，并保存该动作到节点日志。

建议每机预检（只读/不派发）：

```bash
cd /path/to/SMCO
GIT_SHA=$(git rev-parse HEAD) || exit 1
.venv/bin/python -m pytest tests/test_confirmatory.py tests/test_merge_results.py -q || exit 1
.venv/bin/python -c 'import cocoex; print(cocoex.__file__)'  # 仅 E4/E5 节点
.venv/bin/python -m pip freeze > /local/smco-evo-20260802/environment-$(hostname).txt
```

CC 应为每机保存 `node_preflight.json`：hostname、CPU/内存、`GIT_SHA`、Python/Numpy、
`pip freeze` SHA-256、cocoex 版本/路径、开始时间和本地结果根。任一节点版本与冻结环境
合同不兼容时，该节点不接正式任务。

## 3. 推荐顺序与资源分配

| 顺序 | 工作 | 计算量 | 何时可启动 | 停止条件 |
| --- | ---: | ---: | --- | --- |
| P0 | 全节点预检、COCO smoke（非正式） | 每节点 0 个正式 task | 立即 | 任一预检失败，不派该节点 |
| P1 | E6.3 去重/冻结/复用审计 | 0 task | 立即 | 189 unique 的 frozen manifest 或 61 条复用证明不成立 |
| P2 | E6.3 剩余正式任务 | 128（仅 gate 后） | P1 通过 | 任一 shard 与 manifest/实例/provenance 不一致 |
| P3 | E4 formal runner 实现、测试、冻结 | 0 task | 与 P1/P2 并行 | 接口测试或 schema/audit 未通过 |
| P4 | E4 正式 BBOB-largescale | 2,520 | P3 通过 | 任何完整性/audit gate 失败 |
| P5 | E5 formal runner 与运行 | 480 | E4 通过且仍需要补充证据 | 资源紧张、E4 未完成，或接口未审计 |

优先让 213/215/217 做 P0；P1/P3 是 coordinator 上的代码/验证工作。P2 通过后按
run-id 清单均分为 43/43/42；P4 通过后按 COCO dimension 分片：213=d160（840）、
215=d320（840）、217=d640（840）。这种切分公平、透明，且不依赖 CPU 时间作为论文比较。

## 4. E6.3 schedule：189 条正式去重、复用与补跑

### 4.1 先决条件（任一失败即停止）

旧文件是
`result/e6-2026-07-31/schedule/e6_ablations__synthetic_highdim.json`，其 216 条 task
只有189个唯一 `run_id`（27条重复）；旧 raw 有61个唯一 JSON。旧 raw 绝不移动或覆盖。

在运行前，CC/实现者必须完成下列可测试的冻结步骤：

1. 新增/使用纯函数按 `run_id` 去重，首次出现顺序固定；断言
   `len(tasks) == len({run_id}) == 189`。如果重复 run-id 对应的 task content 不完全相同，
   报错，不能任选其一。
2. 新 manifest 使用新的 `manifest_id`（推荐
   `e6_schedule_dedup__synthetic_highdim`）、`frozen=true`、新 hash，写至：
   `result/e6-2026-08-02/e6_schedule_dedup/manifest/`。旧 manifest 保留只读。
3. **实例 gate**：新 manifest 的每一 task 必须引用由规范指定、可验证的实例 index；
   index 中全部条目须是允许的 stage，且 transform/starts hash 与 task 一致。当前工作树
   未发现 `instances_index_confirmatory.json`；因此不能把旧 development E6 静默改称
   confirmatory。必须由论文方案明确以下二选一并写入 manifest/provenance：
   - E6 仍是 `development/mechanistic_ablation`，不进入 confirmatory 主结论；或
   - 新建并冻结适用的 confirmatory/spec instances，并用它们**重新生成整个189任务**，
     旧61条不能复用。

   在没有这个明确决定和对应 index 前，E6.3 不得运行。本手册下文的“复用61条”仅在
   新 manifest task 与旧 raw 的完整 task content、instance hash、starts hash、config hash、
   seed、FE budget 和 `GIT_SHA` 合同均一致，且 stage 语义保持不变时成立。
4. 对61条候选 raw 逐条比较内嵌 task/结果与新 manifest task，输出
   `reuse_decision.csv`（run_id、旧路径、content_sha256、可复用布尔值、拒绝原因）。
   只有61条全部通过才可复用；任何一条不通过则该条进入补跑清单。因此任务数是
   `189 - 可复用数`，现在的预期上界为128，不是承诺值。
5. 生成不可变的三份 run-id 文件：`shards/213.txt` 43 条、`215.txt` 43 条、`217.txt`
   42 条，只含**补跑** run-id；文件 SHA-256 写入 dispatch ledger。

### 4.2 派发模板（仅 P1 gate PASS 后）

以下是模板，`<...>` 必须由 CC 解析成实际绝对路径；不允许手改任务参数。

```bash
cd /path/to/SMCO
GIT_SHA=<frozen_40_hex_sha>
ENV_HASH=<node_environment_hash>
NODE=$(hostname -s)
MANIFEST=/coordinator/result/e6-2026-08-02/e6_schedule_dedup/manifest/e6_schedule_dedup__synthetic_highdim.json
SELECTION=/coordinator/result/e1-2026-07-30/selection_v2/selection.json
INSTANCE_ROOT=<frozen_instance_root>
LOCAL=/local/smco-evo-20260802/e6_schedule_dedup

.venv/bin/python scripts/run_smco_evo_highdim_factorial.py \
  --manifest "$MANIFEST" --instance-root "$INSTANCE_ROOT" \
  --result-dir "$LOCAL/raw_$NODE" --log-dir "$LOCAL/log_$NODE" \
  --only-run-ids $(tr '\n' ' ' < "$LOCAL/run_ids.txt") \
  --confirmatory --selection "$SELECTION" --git-commit "$GIT_SHA" \
  --environment-hash "$ENV_HASH" --machine-id "$NODE" --dry-run

.venv/bin/python scripts/run_smco_evo_highdim_factorial.py \
  --manifest "$MANIFEST" --instance-root "$INSTANCE_ROOT" \
  --result-dir "$LOCAL/raw_$NODE" --only-run-ids $(tr '\n' ' ' < "$LOCAL/run_ids.txt") \
  --validate-only
```

只有 dry-run 的计划数严格等于该 shard 文件行数，且 validate-only 没有意外已有/孤儿结果，
才移除 `--dry-run`。并发数从每机 `--workers 1` 起；完成首个 task 后根据内存、超时和
CPU 利用率最多增至该机物理核数的一半。wall-clock 只记录为敏感性/成本元数据，不能用于
跨机器算法排名。

### 4.3 回传、合并与验收

节点完成后将**整个** `raw_$NODE/`、`log_$NODE/`、`node_preflight.json` 和 shell 日志
回传到 coordinator 的新目录（例如
`result/e6-2026-08-02/e6_schedule_dedup/incoming/<node>/`）。回传后先对文件数和 SHA-256
清单核验，随后在 coordinator 创建新的 `raw_assembled/`（复制，不覆盖旧 raw）。把通过
复用检查的61条也复制进该目录。

```bash
.venv/bin/python scripts/merge_smco_evo_highdim_results.py \
  --manifest result/e6-2026-08-02/e6_schedule_dedup/manifest/e6_schedule_dedup__synthetic_highdim.json \
  --raw-dir result/e6-2026-08-02/e6_schedule_dedup/raw_assembled \
  --merged-dir result/e6-2026-08-02/e6_schedule_dedup/merged_v1
```

验收必须同时满足：189 个物理 `valid_runs.csv` 行、189 unique run-id、0 missing、0 orphan、
0 unexpected duplicate、每行 FE 不超 task budget、configuration/seed/instance/starts hash 全匹配、
`provenance_complete` 通过，且 `provenance_audit.json` 为 **12 checks 全 PASS**。若 E6 保持
development，12-check 中 confirmatory 项的适用语义应由代码/合同明示，不能人工修改 audit JSON。

## 5. E4 BBOB-largescale：正式外部验证的实现与运行

### 5.1 已有冻结输入和公平合同

已有 manifest：
`result/e4-2026-07-31/e4_bbob_largescale__bbob-largescale.json`，hash
`1a8f215431099f0de5e8f944de9c1a33cf3b6101a195b5f70dca43f91760e838`。
它冻结了 Python winner `PY-SP-SMCO-EVO`、matched base `PY-BASE-SMCO`、五个 baseline
`DE/GA/PSO/SA/GenSA`，24 functions × 3 dims (160/320/640) × 5 instances，共2,520 task，
每 task `1000*d` FE。该矩阵是公平预算合同；不能因节点性能更快而增加某算法 FE。

当前 `scripts/run_smco_evo_bbob_largescale.py` 虽校验 manifest/selection 并能生成 CSV，
但它：

- 迭代 COCO suite，而不是以 manifest task 执行；
- 没有 `--only-run-ids`、`--workers`、`--resume`、`--dry-run`、`--validate-only`；
- 只写 CSV/provenance，不产生符合 `merge_results.py` 的 per-task raw outcome；
- 没有进入 12-check merge/audit 的明确适配。

因此 CC 不能直接把这支脚本丢到三台机器跑；这样会重复整套2,520任务且无法证明 manifest
覆盖率。

### 5.2 先实现的精确任务（先 TDD、独立提交）

实现者须先完成以下接口，全部通过后才解除 E4 运行 gate：

1. 新建 `coco_task_worker`/扩展 `coco_runner`：输入一个冻结 manifest task，在指定
   `function/dimension/instance` 上只运行该 task 的 algorithm；COCO problem 选择必须与 task
   严格匹配；记录 FE、best value、target hit、status/failure、COCO version、observer 路径和
   全部 provenance。
2. 新建 E4 batch runner：支持 `--manifest --selection --result-dir --only-run-ids --workers`
   `--resume --dry-run --validate-only --machine-id --git-commit --environment-hash`；正式模式
   强制 `enforce_confirmatory`、完整40位 SHA 和 locked 7-algorithm matrix。CLI 不得保留能在
   confirmatory 模式改变 baselines/dims/instances/budget 的开关。
3. 制定 COCO outcome adapter：每个 task 的 JSON 必须能由统一 merge 读取，或为 COCO 建立
   等价的 merge 函数，输出 `all_attempts.csv`、`valid_runs.csv`、`missing_runs.csv`、
   `duplicate_runs.csv`、`provenance_audit.json`。Audit 至少覆盖 frozen manifest hash、task
   coverage、unique run-id、FE cap、algorithm/config/seed 一致性、COCO suite/version、三项
   provenance、failure 保留；目标是沿用项目的12-check 合同，若 COCO 与 synthetic 字段不同，
   必须写明等价检查并有测试，不能伪造 instance-artifact hash。
4. 在 canonical artifact contract 增加 `e4_formal_manifest`、`e4_formal_merged`，并将现有
   `e4_dev` 保留为 `development_only`，不得覆写。Task 12/13 只允许通过 canonical index
   读取正式 E4 artifact。
5. 测试最低集：单 task 对应 COCO problem；错误 function/dim/instance 拒绝；FE 严格停止；
   manifest过滤不改变 task；resume 不重复；缺 provenance 拒绝；2,520 矩阵的三分片并集
   恰好覆盖；重复/缺失 run-id audit 失败；R winner 标记 `python_port_external`。完成后全量 pytest。

### 5.3 COCO 环境 gate 与三机运行

冻结前，三机执行同一小 smoke：各自运行 manifest 中预先固定的**一个 smoke task**，该输出
置于 `smoke_not_for_analysis/`，不得 merge/canonical。三个节点必须使用同一 cocoex/coco-experiment
版本（记录 package version、`cocoex.__file__`、suite 名和 observer version）。若不同，先用锁定
环境安装/容器统一，不能混合进正式结果。

正式输出根建议：
`result/e4-2026-08-02/formal/`，节点本地仍写
`/local/smco-evo-20260802/e4_formal/raw_<node>/`。固定 shard：d160→213，d320→215，d640→217，
各840 task；每节点从 manifest 提取 run-id 文件并先 dry-run/validate-only。回传、组装、merge、
12-check audit 的规则完全遵循第4.3节。全部2,520条合格前不得挑选子集写结论；任何 COCO
安装/observer 故障的 task 仍在 missing/failure 分母中，停止后先修基础设施再同 run-id 重试。

E4 的定位仅为 Python frozen winner 的外部验证。若未来 winner 是 R，E4 必须明确标
`python_port_external_check=true`，不能作为 R winner 的外部确认。

## 6. E5 low-dim support：低优先级、同样不能直接运行

E5 manifest
`result/e5-2026-07-31/e5_lowdim_check__bbob.json`（hash
`001ac85ee3debff155b0349e358aa1a648dad52e142a5878e243cba5bea1a2af`）冻结 winner/base
两算法、24 functions × d={5,20} × 5 instances，共480 task，FE cap=`2000*d`。

当前 `scripts/run_smco_evo_lowdim_check.py` 与 E4 有同一结构缺口：它直接遍历 suite 并写
`lowdim_degradation.csv`，没有 manifest task 级分片、raw outcome 或统一 merge/audit。因此**不要
现在运行它来占用 fleet**。E5 的科学定位是“低维不严重退化”的补充，不能反转高维 winner；
其优先级低于 E4 和 E6.3，也不能用它事后替换主结论。

E5 正式化复用 E4 的 COCO task worker/adapter，额外只需：

1. E5 canonical runner 固定为 winner/base、`bbob`、d={5,20}、instances 1--5、`2000*d`；
   从 E5 manifest 而非 CLI 读所有合同字段。
2. 新目录 `result/e5-2026-08-02/formal/`，新的 canonical index key；保留 e5_dev。
3. 分片建议：213 跑 d=5 的240 task；215 跑 d=20 的240 task；217 只做审计/重试备用。
4. 完整480条、12-check audit 通过后，才以“supporting / low-dimensional non-degradation”
   标记进入补充材料；若资源或 E4 gate 未完成，E5 停止，不影响高维主论文包。

## 7. CC 的调度 ledger、回传与停止规则

每次 campaign 在 coordinator 创建只追加的 `dispatch_ledger.jsonl`。一行记录：campaign、
manifest SHA、selection SHA、instance/index SHA（如适用）、shard SHA、node、local path、
GIT_SHA、ENV_HASH、COCO version、开始/结束时间、退出码、回传 SHA、验证结果。CC 不应把
remote 目录存在当作成功；只以回传文件清单、merge 和 audit 为完成判定。

统一停止条件：

- frozen/hash/selection/instance/provenance preflight 任一失败：不派发该 shard；
- dry-run 数量不等于 run-id 文件数量：停止，修 manifest/shard；
- 一个 task 发生代码错误：停止该节点该 shard，保留 raw/log，先修复并提交，再从相同 run-id
  恢复；
- merge 出现 orphan、missing、pseudo duplicate、FE cap 或 provenance failure：不生成图、不更新
  canonical index；
- 只有完整 audit PASS 后，才由 `freeze_canonical_artifacts.py` 生成新的 index，并允许 Task 12/13
  引用该 artifact。

## 8. 给 CC 的立即行动清单

1. 在213/215/217做 P0 预检并回传 node ledger；对 E4/E5 仅做 import/version smoke，不运行
   formal COCO matrix。
2. 让实现者先完成 E6.3 的去重+实例语义 gate；在明确 E6 是 development 还是新 confirmatory/spec
   的书面决定前，不派128条计算。
3. 让实现者完成共享 COCO task-level runner/merge/audit；其测试和 manifest/canonical 合同通过后，
   再派 E4 的840×3。
4. E4 audit PASS 后再判断是否值得运行 E5；若运行，按240×2并保留217作重试/审计节点。

这份顺序确保空闲机器有合规预检和后续明确队列，同时不把 development CSV、跨语言 port 或
事后扩展伪装成确认性论文证据。
