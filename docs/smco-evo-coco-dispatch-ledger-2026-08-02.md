# COCO formal dispatch ledger (2026-08-02)

冻结代码 SHA：`32ffd03c0f957e32b23c396f24708815f6a48325`。

## E4 formal BBOB-largescale

Manifest：`result/e4-2026-07-31/e4_bbob_largescale__bbob-largescale.json`（2520 tasks）。
Selection：`result/e1-2026-07-30/selection_v2/selection.json`。

| 节点 | 维度 | 任务数 | run-id 清单 | raw | log |
|---|---:|---:|---|---|---|
| m213 | 160 | 840 | `result/e4-2026-08-02/formal/shards/213.txt` | `result/e4-2026-08-02/formal/raw_213/` | `result/e4-2026-08-02/formal/log_213/run.log` |
| m215 | 320 | 840 | `result/e4-2026-08-02/formal/shards/215.txt` | `result/e4-2026-08-02/formal/raw_215/` | `result/e4-2026-08-02/formal/log_215/run.log` |
| m217 | 640 | 840 | `result/e4-2026-08-02/formal/shards/217.txt` | `result/e4-2026-08-02/formal/raw_217/` | `result/e4-2026-08-02/formal/log_217/run.log` |

## E5 formal low-dimensional check

Manifest：`result/e5-2026-07-31/e5_lowdim_check__bbob.json`（480 tasks）。

| 节点 | 任务数 | run-id 清单 | raw | log |
|---|---:|---|---|---|
| m214 | 240 | `result/e5-2026-08-02/formal/shards/214.txt` | `result/e5-2026-08-02/formal/raw_214/` | `result/e5-2026-08-02/formal/log_214/run.log` |
| m251 | 240 | `result/e5-2026-08-02/formal/shards/251.txt` | `result/e5-2026-08-02/formal/raw_251/` | `result/e5-2026-08-02/formal/log_251/run.log` |

## 接手检查

各节点均使用 `--machine-id`、冻结 SHA 和非空 `--environment-hash` 派发。接手时先检查进程、日志和 raw JSON 数量；不要把 smoke 目录合并。完成后把各节点 raw/log 回传到 coordinator，再执行 merge、12-check audit，最后分别用 formal external index freeze 和 COCO-native analysis。

当前仅观察到正式进程已启动；中途失败的任务必须保留在日志/失败分母中，不能删除后声称完成。
