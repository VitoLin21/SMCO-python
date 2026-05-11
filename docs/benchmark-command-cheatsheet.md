# SMCO Benchmark 命令清单

本文档给后续 LLM 代理或人工操作使用，默认工作目录为：

```bash
cd /amax/math/code/SMCO
```

统一解释器：

```bash
.venv/bin/python
```

---

## 1. 基线检查

```bash
.venv/bin/python --version
.venv/bin/python -m pytest -q
```

预期：测试通过（当前基线 70 tests）。

---

## 2. 快速冒烟（单次/低开销）

### 2.1 论文 smoke 配置

```bash
.venv/bin/python - <<'PY'
from smco import run_paper_smoke
runs = run_paper_smoke(iter_max=40, n_starts=5, seed=123)
for name, run in runs.items():
    print(name, run.best_value, run.best_x)
PY
```

### 2.2 单函数单次（Rastrigin 2D）

```bash
.venv/bin/python - <<'PY'
from smco import run_benchmark
run = run_benchmark(
    name="Rastrigin",
    dim=2,
    variant="smco_r",
    repetitions=1,
    iter_max=60,
    n_starts=5,
    seed=123,
)
print("best_value:", run.best_value)
print("best_x:", run.best_x)
PY
```

---

## 3. 变体对比（smco / smco_r / smco_br / smco_multi）

```bash
.venv/bin/python - <<'PY'
from smco import run_benchmark

for variant in ["smco", "smco_r", "smco_br", "smco_multi"]:
    run = run_benchmark(
        name="Ackley",
        dim=10,
        variant=variant,
        repetitions=3,
        iter_max=120,
        n_starts=8,
        seed=123,
    )
    print(variant, "best_value=", run.best_value, "best_x_norm=", (run.best_x**2).sum()**0.5)
PY
```

---

## 4. 论文常见函数批量跑（低配版）

```bash
.venv/bin/python - <<'PY'
from smco import run_benchmark

configs = [
    ("Rastrigin", 2),
    ("Ackley", 2),
    ("Griewank", 2),
    ("Rosenbrock", 10),
    ("DixonPrice", 10),
    ("Zakharov", 10),
    ("Qing", 10),
    ("Michalewicz", 10),
]

for name, dim in configs:
    run = run_benchmark(
        name=name,
        dim=dim,
        variant="smco_r",
        repetitions=5,
        iter_max=150,
        n_starts=10,
        seed=123,
    )
    print(f"{name:12s} d={dim:4d} best={run.best_value:.6f}")
PY
```

---

## 5. 输出结果到 JSON/CSV（便于后续分析）

### 5.1 JSON 落盘

```bash
.venv/bin/python - <<'PY'
import json
from smco import run_benchmark

configs = [
    ("Rastrigin", 2),
    ("Ackley", 2),
    ("Griewank", 2),
]

rows = []
for name, dim in configs:
    run = run_benchmark(
        name=name,
        dim=dim,
        variant="smco_r",
        repetitions=10,
        iter_max=120,
        n_starts=8,
        seed=123,
    )
    rows.append({
        "name": name,
        "dim": dim,
        "variant": run.variant,
        "best_value": run.best_value,
        "best_x": run.best_x.tolist(),
        "values": [r.best_result.f_optimal for r in run.results],
    })

with open("benchmark-results-smoke.json", "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)

print("wrote benchmark-results-smoke.json")
PY
```

### 5.2 CSV 落盘

```bash
.venv/bin/python - <<'PY'
import csv
from smco import run_benchmark

configs = [("Rastrigin", 2), ("Ackley", 2), ("Griewank", 2)]

with open("benchmark-results-smoke.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["name", "dim", "variant", "rep", "value"])
    writer.writeheader()
    for name, dim in configs:
        run = run_benchmark(
            name=name,
            dim=dim,
            variant="smco_r",
            repetitions=10,
            iter_max=120,
            n_starts=8,
            seed=123,
        )
        for i, r in enumerate(run.results, 1):
            writer.writerow({
                "name": name,
                "dim": dim,
                "variant": run.variant,
                "rep": i,
                "value": r.best_result.f_optimal,
            })

print("wrote benchmark-results-smoke.csv")
PY
```

---

## 6. 论文级扩展建议（逐步放大）

不要一次性上大规模，建议按这条路径增加计算量：

1. `repetitions`: `5 -> 20 -> 100`
2. `iter_max`: `120 -> 300 -> 500`
3. `dim`: `2/10 -> 20/50 -> 100+`
4. `n_starts`: `max(5, sqrt(dim))` 起步，再逐步增大

示例（中等规模）：

```bash
.venv/bin/python - <<'PY'
from smco import run_benchmark

run = run_benchmark(
    name="Rastrigin",
    dim=50,
    variant="smco_br",
    repetitions=20,
    iter_max=400,
    n_starts=12,
    seed=2026,
)
print("best_value:", run.best_value)
print("best_x[:5]:", run.best_x[:5])
PY
```

---

## 7. 兼容命名提示（已支持）

`assign_config` 支持以下上游常用别名：

- `SqNorm`
- `-SqNorm`
- `Mishra06`

也支持常规名字：

- `Rastrigin`, `Ackley`, `Griewank`, `Rosenbrock`, `DixonPrice`, `Zakharov`, `Qing`, `Michalewicz`
- `DropWave`, `Shubert`, `McCormick`, `SixHumpCamel`, `Camel6`, `Eggholder`, `Bukin6`, `Easom`, `Mishra6`

---

## 8. 常见报错处理

### 8.1 `ModuleNotFoundError: smco`

用 `.venv` 解释器运行，不要用系统 Python：

```bash
.venv/bin/python -m pytest -q
```

### 8.2 benchmark 名称错误

先确认名字拼写和维度，尤其 2D-only 函数必须 `dim=2`。

### 8.3 结果波动较大

先固定 `seed`，再增加 `repetitions` 看统计分布，不要只看单次结果。
