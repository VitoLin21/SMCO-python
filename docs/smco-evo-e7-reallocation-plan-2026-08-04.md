# E7 停机节点回收与迁移计划（2026-08-04）

## 1. 范围与冻结边界

`m213`、`m214`、`m215`、`m217` 已在停机前完成非破坏性 evidence
回收，随后停止 E7 并关机。四台机器不再是可调度资源。回收根目录为：

```text
result/smco-evo-ultrahighdim-2026/e7/collected-2026-08-04-pre-stop/
```

迁移必须使用既有 E7 `manifest.json`、`shards.json`、原始 run_id、seed、
configuration hash、实例 artifact 和 FE budget。禁止重建 manifest、重新分配
seed、删除旧 attempt，或把已成功 run 重新计算。

## 2. 停机时快照

| 原节点 | success | open | retryable | 未完成 |
| --- | ---: | ---: | ---: | ---: |
| m213 | 54 | 18 | 0 | 18 |
| m214 | 50 | 18 | 0 | 18 |
| m215 | 64 | 15 | 30 | 45 |
| m217 | 34 | 8 | 16 | 24 |
| **合计** | **202** | **59** | **46** | **105** |

未完成任务按维度为 `d=10000: 76`、`d=5000: 22`、`d=3000: 7`；其中
`R-DEoptim: 7`、`STOGO: 4`。因而不能把完整 shard 迁移到本机或 `m253`：
它们不具备 E7 R comparator 的正式合同。`m251` 仍有 30 个 E7 worker，
不作为本轮迁移目的地。

## 3. 目标节点与 shard 映射

完整复制一个原 evidence root（包括 success 与未完成 ledger），再在目标节点
以原 shard-id 默认 resume。完整复制是避免成功任务因目的端缺少 ledger 而被
误重跑的必要条件。

| 目标节点 | 迁移来源 evidence root | 原 shard | 总任务 | 未完成 | 建议新 dispatcher workers |
| --- | --- | --- | ---: | ---: | ---: |
| zf129 | m213/evidence_m213_s002 | shard-002 | 36 | 9 | 3 |
| zf129 | m213/evidence_m213_s003 | shard-003 | 36 | 9 | 3 |
| zf132 | m214/evidence_m214_s004 | shard-004 | 34 | 9 | 3 |
| zf132 | m214/evidence_m214_s005 | shard-005 | 34 | 9 | 3 |
| zf852 | m215/evidence_m215_s006 | shard-006 | 34 | 15 | 5 |
| zf852 | m215/evidence_m215_s007 | shard-007 | 34 | 15 | 5 |
| zf852 | m215/evidence_m215_s017 | shard-017 | 41 | 15 | 5 |
| zf133 | m217/evidence_m217_s008 | shard-008 | 29 | 12 | 4 |
| zf133 | m217/evidence_m217_s009 | shard-009 | 29 | 12 | 4 |

这会使 zf129/zf132 各新增 6 workers（低于用户 76 worker 上限）；zf133
新增 8 workers；zf852 新增 15 workers。目标节点必须在启动前复核现有 E7
worker 与磁盘余量，且总并发不得超过既有用户限制。

## 4. 恢复顺序

1. 从 coordinator 回收副本为每个迁移 root 计算文件清单及 SHA-256；复制到目标
   节点的新、唯一 evidence root（例如 `evidence_migrated_m213_s002`），禁止覆盖
   目标节点任何现有 `evidence_*` 目录。
2. 在目标节点验证 E7 manifest/shards hash、实例 artifact hash、Python/R 依赖、
   algorithm-core hash 和 deployment manifest；R-DEoptim/STOGO 必须真实 preflight
   成功。
3. 对 m213/m214 的 one-or-more-heartbeat open attempt，先用已审计的 orphan
   recovery 工具追加 `stalled` 事件；不得手工编辑 JSON。对已有 retryable 的
   m215/m217 ledger 不改写。
4. 确认目标 evidence root 内 success run 与原快照完全一致，再用原 shard-id 执行
   默认 `dispatch`（不得使用 `--no-resume`）。每个 shard 同时只能有一个
   dispatcher；原节点已关机，故不存在源端并发。
5. 启动后验证新 attempt 的 `supersedes_attempt_id`、machine/environment provenance
   和未完成计数；成功数量不得下降。

## 5. 完成条件

- 九个迁移 root 的 run_id 并集与停机前回收副本一致；
- 202 个原 success run 不重跑；
- 105 个未完成 run 都有可审计的新 attempt 或 success outcome；
- 回收时将各目标 evidence root 与 coordinator 原副本合并，按 E7 的 deadline、
  provenance 和 duplicate audit 通过后，才进入 E7 merge/composite/statistics。

## 6. 迁移执行记录（2026-08-04）

九个完整 evidence root 已从 coordinator 的停机前回收副本复制到上述目标节点的
唯一 `evidence_migrated_*` 目录。目标节点 R-DEoptim/STOGO preflight 均通过。
为修正旧 `unsupported_dependency -> a002` 链在 validator 中被误判为 invalid
的遗漏，迁移 dispatcher 使用调度修订 `396c5926894684b665688d51c6ec2e74779d179a`；
所有四个目标节点的 `algorithm_core_sha256` 均为
`14d13155938477fc0b7dfbaa28b8816d6451c266e86c7f76a1fe3bd273dafdfa`，与既有 E7
算法核一致。

首次 resume 后的九个 shard 均满足：原 success 仍为 completed，原 open 已写入
stalled/a002 恢复链。迁移工作总状态为：

| 状态 | logical run 数 |
| --- | ---: |
| 已保留 completed | 202 |
| 新 dispatcher 中 running | 35 |
| retryable queue | 70 |
| 失败或丢失 | 0 |

新增并发严格为 zf129=6、zf132=6、zf133=8、zf852=15 workers；未修改这些节点原有
E7 evidence root 或 dispatcher。
