
## SMCO-HB 多保真 HPO （效果比较差，已废弃！）

`smco_hb()` 将现有 `SMCO` 变体作为配置建议器，并用同步式 Hyperband / Successive Halving 做训练资源分配。

### 第一版能力边界

- 只支持连续数值型搜索空间
- fidelity 由 `budget` 表示，例如 epoch、training steps、dataset fraction
- 支持固定 worker 池的同步并发
- 支持可选 checkpoint 透传
- 不支持 ASHA 异步晋级或 categorical / conditional space

### 使用示例

```python
from smco import smco_hb
import numpy as np


def evaluate(config, budget, seed=None, checkpoint=None):
    x = np.asarray(config, dtype=float)
    return {
        "score": 1.0 - float(np.sum((x - 0.2) ** 2)) - 0.2 / float(budget),
        "checkpoint": {"budget": budget},
    }


run = smco_hb(
    suggest_variant="smco_r",
    config_space=([-5.0, -5.0], [5.0, 5.0]),
    evaluate=evaluate,
    min_budget=1,
    max_budget=9,
    eta=3,
    n_workers=4,
    seed=123,
    proposer_options={"pool_size": 9, "iter_max": 40},
)

print(run.incumbent_config)
print(run.incumbent_score)
print(run.incumbent_budget)
```

### 接口说明

```python
smco_hb(
    *,
    suggest_variant,
    config_space,
    evaluate,
    min_budget,
    max_budget,
    eta=3,
    seed=123,
    n_workers=1,
    proposer_options=None,
    runner_options=None,
) -> SMCOHBResult
```

| 参数 | 说明 |
|------|------|
| `suggest_variant` | `smco` / `smco_r` / `smco_br`，决定配置建议器 |
| `config_space` | 连续数值空间，可写成 `(bounds_lower, bounds_upper)` |
| `evaluate` | 预算感知评估函数，签名为 `evaluate(config, budget, seed=None, checkpoint=None)` |
| `min_budget`, `max_budget`, `eta` | Hyperband / Successive Halving 调度参数 |
| `n_workers` | 同一 rung 内的固定并发 worker 数 |
| `proposer_options` | 传给建议器的控制参数，如 `pool_size`、`iter_max` |
| `runner_options` | 预留给执行层的附加参数 |

`SMCOHBResult` 包含：

- `incumbent_config`
- `incumbent_score`
- `incumbent_budget`
- `history`
- `brackets`
- `trajectory`
- `metadata`

---