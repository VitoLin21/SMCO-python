# E6.1 start-count 消融 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 E6.1 支持 n_starts∈{8, 16, ceil(sqrt(d))} 三档消融，per-n_starts starts artifact（方案 A，向后兼容 n8），由 worker 按 `task.n_starts` 加载对应档。

**Architecture:** 同 instance dir 写多份 starts（`starts.csv.gz`[n8] + `starts_n{N}.csv.gz`），transform 只存一次；metadata 记每档 hash；`expand_tasks` 按 `config.n_starts` 选 hash；E6.1 专用 per-dim expand（ceil√d 随 d 变）；`_identity_key` 增 n_starts 区分三档（不动 derive_seed）。

**Tech Stack:** Python 3 + numpy + pytest；R 4.3.2。`paper_contract.py` 保持 stdlib-only。

**Spec:** `docs/superpowers/specs/2026-07-29-e6-start-count-ablation-design.md`

**分支：** `feat/smco-evo-highdim-paper-2026`。本机可跑全部测试与 R 端到端。

**全局约定：** TDD（先失败测试→跑→实现→跑→commit）；每 task 结束 commit，message 末尾加 `Co-Authored-By: Claude <noreply@anthropic.com>`；测试用 `.venv/bin/python -m pytest <path> -v`。

---

## 文件结构

| 文件 | 责任 | 动作 |
|---|---|---|
| `src/smco/highdim_instances.py` | `write_instance_artifacts(extra_starts=)`、`load_starts(dir,n_starts)` | 修改 |
| `scripts/generate_smco_evo_manifests.py` | `_starts_seed(...,n_starts)`、`--extra-n-starts`、index entry | 修改 |
| `src/smco/experiment_manifests.py` | `expand_tasks` 按 config.n_starts 选 start_points_hash | 修改 |
| `src/smco/ablations.py` | `start_count_configs(winner,dim)` | 修改 |
| `scripts/run_smco_evo_ablations.py` | E6.1 per-dim manifest 路径（`--dimension start_count`） | 修改 |
| `src/smco/merge_results.py` | `_identity_key` 增 n_starts | 修改 |
| `scripts/run_smco_evo_highdim_factorial.py` | `load_starts(n_starts)` + provenance 校验对应档 | 修改 |
| `vendor/SMCO_R/main/highdim_instances.R` | `load_highdim_instance(n_starts)` | 修改 |
| `scripts/run_smco_evo_highdim_r.R` | 传 `n_starts` 给 loader | 修改 |
| `tests/test_highdim_instances.py` | extra_starts + load_starts(n) | 修改 |
| `tests/test_experiment_manifests.py` | expand_tasks 按 n 选 hash | 修改 |
| `tests/test_ablations.py` | start_count_configs | 修改 |
| `tests/test_merge_results.py` | _identity_key 含 n_starts | 修改 |

---

## Task 1: write_instance_artifacts extra_starts + load_starts(n_starts)

**Files:**
- Modify: `src/smco/highdim_instances.py`
- Test: `tests/test_highdim_instances.py`

- [ ] **Step 1: 写失败测试** — 在 `tests/test_highdim_instances.py` 末尾追加：

```python
def test_write_extra_starts_artifacts(tmp_path):
    from smco.highdim_instances import generate_instance, write_instance_artifacts, load_starts
    import json
    inst = generate_instance("Rastrigin", 4, 0, seed=1)
    import numpy as np
    starts8 = inst.bounds_lower.reshape(1, -1).repeat(8, axis=0)
    extra16 = inst.bounds_upper.reshape(1, -1).repeat(16, axis=0)
    meta = write_instance_artifacts(inst, starts8, tmp_path, extra_starts={16: extra16})
    assert meta["extra_starts"]["16"]["file"] == "starts_n16.csv.gz"
    assert meta["extra_starts"]["16"]["n_starts"] == 16
    assert (tmp_path / "starts_n16.csv.gz").exists()
    # default n8 path unchanged
    assert meta["file_hashes"]["starts"] and meta["n_starts"] == 8


def test_load_starts_by_n(tmp_path):
    from smco.highdim_instances import generate_instance, write_instance_artifacts, load_starts
    import numpy as np
    inst = generate_instance("Rastrigin", 4, 0, seed=1)
    s8 = np.zeros((8, 4)); s16 = np.full((16, 4), 7.0)
    write_instance_artifacts(inst, s8, tmp_path, extra_starts={16: s16})
    assert load_starts(tmp_path).shape == (8, 4)          # default n8
    assert load_starts(tmp_path, 8).shape == (8, 4)       # explicit n8
    assert load_starts(tmp_path, 16).shape == (16, 4)     # extra tier


def test_load_starts_missing_tier_raises(tmp_path):
    from smco.highdim_instances import generate_instance, write_instance_artifacts, load_starts
    import numpy as np
    import pytest
    inst = generate_instance("Rastrigin", 4, 0, seed=1)
    write_instance_artifacts(inst, np.zeros((8, 4)), tmp_path)
    with pytest.raises(FileNotFoundError):
        load_starts(tmp_path, 99)
```

- [ ] **Step 2: 跑确认失败**

Run: `.venv/bin/python -m pytest tests/test_highdim_instances.py -v -k "extra_starts or load_starts_by_n or missing_tier"`
Expected: FAIL — `TypeError: write_instance_artifacts() got an unexpected keyword argument 'extra_starts'`

- [ ] **Step 3: 实现** — 在 `src/smco/highdim_instances.py`：

(a) `write_instance_artifacts` 签名改为加 `extra_starts=None`，并在写完默认 starts 后、构造 metadata 前插入额外档写入：

```python
def write_instance_artifacts(
    instance: HighDimInstance, starts: np.ndarray, out_dir: str | Path,
    *, extra_starts: dict | None = None,
) -> dict:
```

在 `starts_path`/`_write_matrix_gz(starts_path, starts)` 之后、`metadata = {...}` 之前插入：

```python
    extra_starts_meta: dict = {}
    if extra_starts:
        for n_starts_tier, tier_starts in extra_starts.items():
            n_starts_tier = int(n_starts_tier)
            tier_starts = np.asarray(tier_starts, dtype=float)
            if tier_starts.shape[0] != n_starts_tier:
                raise ValueError(
                    f"extra_starts[{n_starts_tier}] has {tier_starts.shape[0]} rows"
                )
            if tier_starts.shape[1] != instance.dimension:
                raise ValueError(
                    f"extra_starts[{n_starts_tier}] has {tier_starts.shape[1]} cols, "
                    f"expected dimension {instance.dimension}"
                )
            tier_path = out_dir / f"starts_n{n_starts_tier}.csv.gz"
            _write_matrix_gz(tier_path, tier_starts)
            extra_starts_meta[str(n_starts_tier)] = {
                "file": tier_path.name,
                "hash": _sha256_file(tier_path),
                "n_starts": n_starts_tier,
            }
```

并在 `metadata` dict 里加一行 `"extra_starts": extra_starts_meta,`（紧挨 `"transform_sha256": spec.sha256(),` 之后）。

(b) `load_starts` 改为按 n_starts 选文件：

```python
def load_starts(artifact_dir: str | Path, n_starts: int = 8) -> np.ndarray:
    artifact_dir = Path(artifact_dir)
    if int(n_starts) == 8:
        return _read_matrix_gz(artifact_dir / "starts.csv.gz")
    path = artifact_dir / f"starts_n{int(n_starts)}.csv.gz"
    if not path.exists():
        raise FileNotFoundError(
            f"no starts artifact for n_starts={n_starts} in {artifact_dir}"
        )
    return _read_matrix_gz(path)
```

- [ ] **Step 4: 跑通过**

Run: `.venv/bin/python -m pytest tests/test_highdim_instances.py -v`
Expected: PASS（全部）

- [ ] **Step 5: Commit**

```bash
git add src/smco/highdim_instances.py tests/test_highdim_instances.py
git commit -m "feat(instances): per-n_starts starts artifacts + load_starts(n_starts)" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: _starts_seed 加 n_starts + generate --extra-n-starts

**Files:**
- Modify: `scripts/generate_smco_evo_manifests.py`
- Test: `tests/test_highdim_instances.py`（或新建轻量测试，放此处）

- [ ] **Step 1: 写失败测试** — 在 `tests/test_highdim_instances.py` 末尾追加（测 generate 的 extra starts 集成）：

```python
def test_build_instance_set_extra_starts(tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "gen", Path("scripts/generate_smco_evo_manifests.py"))
    gen = importlib.util.module_from_spec(spec); spec.loader.exec_module(gen)
    from smco.highdim_instances import load_starts
    idx = gen.build_instance_set(
        ["Rastrigin"], [16], 1, stage="development", out_dir=tmp_path,
        n_starts=8, extra_n_starts=("16", "sqrt"))
    entry = idx["instances"][0]
    assert "16" in entry["extra_starts"]               # explicit 16
    sqrt_n = int(__import__("math").ceil(__import__("math").sqrt(16)))  # =4
    assert str(sqrt_n) in entry["extra_starts"]         # sqrt tier resolved to 4
    # the files exist and load to the right shape
    art = tmp_path / entry["artifact_dir"]
    assert load_starts(art, 16).shape == (16, 16)
    assert load_starts(art, sqrt_n).shape == (sqrt_n, 16)
```

- [ ] **Step 2: 跑确认失败**

Run: `.venv/bin/python -m pytest tests/test_highdim_instances.py::test_build_instance_set_extra_starts -v`
Expected: FAIL — `TypeError: build_instance_set() got an unexpected keyword argument 'extra_n_starts'`

- [ ] **Step 3: 实现** — 在 `scripts/generate_smco_evo_manifests.py`：

(a) 顶部加 `import math`。

(b) `_starts_seed` 加 n_starts 维度：

```python
def _starts_seed(function: str, dim: int, instance_id: int, stage: str, n_starts: int = 8) -> int:
    """Stable seed for a start matrix, decorrelated from the transform seed and across n_starts tiers."""
    key = f"{stage}:starts:{function}:{dim}:{instance_id}:{n_starts}"
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)
```

(c) 加一个解析器 + 改 `build_instance_set`：

```python
def _resolve_n_starts(spec, dim: int) -> int:
    if str(spec) == "sqrt":
        return int(math.ceil(math.sqrt(dim)))
    return int(spec)


def build_instance_set(
    functions, dims, n_instances, *, stage, out_dir, n_starts=8,
    extra_n_starts=(), block_size=None, dry_run=False,
) -> dict:
    """Materialise instance artifacts; ``extra_n_starts`` adds per-tier starts
    artifacts (entries like ``"16"`` or ``"sqrt"`` → ceil(sqrt(dim)))."""
    out_dir = Path(out_dir)
    functions = list(functions)
    dims = [int(d) for d in dims]
    total = len(functions) * len(dims) * n_instances

    if dry_run:
        return {"dry_run": True, "stage": stage, "generator_version": GENERATOR_VERSION,
                "functions": functions, "dims": dims, "n_instances": n_instances,
                "n_starts": n_starts, "extra_n_starts": list(extra_n_starts),
                "instances_planned": total}

    gen_kwargs: dict = {"stage": stage}
    if block_size is not None:
        gen_kwargs["block_size"] = block_size

    entries: list[dict] = []
    for function in functions:
        for dim in dims:
            for instance_id in range(n_instances):
                instance = generate_instance(function, dim, instance_id, **gen_kwargs)
                starts = _shared_starts(
                    instance, n_starts, _starts_seed(function, dim, instance_id, stage, n_starts))
                extra: dict = {}
                for spec in extra_n_starts:
                    n = _resolve_n_starts(spec, dim)
                    if n == n_starts:
                        continue
                    extra[n] = _shared_starts(
                        instance, n, _starts_seed(function, dim, instance_id, stage, n))
                art_dir = _artifact_dir(out_dir, function, dim, instance_id, stage)
                meta = write_instance_artifacts(instance, starts, art_dir, extra_starts=extra)
                entries.append({
                    "function": function, "dimension": dim, "instance_id": instance_id,
                    "stage": stage, "artifact_dir": str(art_dir.relative_to(out_dir)),
                    "n_starts": n_starts, "known_optimum_value": meta["known_optimum_value"],
                    "transform_sha256": meta["transform_sha256"], "file_hashes": meta["file_hashes"],
                    "extra_starts": meta.get("extra_starts", {}),
                })

    index = {"generator_version": GENERATOR_VERSION, "default_block_size": DEFAULT_BLOCK_SIZE,
             "stage": stage, "n_starts": n_starts, "extra_n_starts": list(extra_n_starts),
             "functions": functions, "dims": dims, "n_instances": n_instances, "instances": entries}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "instances_index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False))
    return index
```

(d) CLI 加 `--extra-n-starts`（在 `--n-starts` 参数之后）：

```python
    parser.add_argument(
        "--extra-n-starts", nargs="*", default=[],
        help="Extra start-count tiers (e.g. 16 sqrt). sqrt → ceil(sqrt(dim)).",
    )
```

并把 `build_instance_set(...)` 调用（`if args.stage == "instances":` 块内）加 `extra_n_starts=args.extra_n_starts`。

- [ ] **Step 4: 跑通过**

Run: `.venv/bin/python -m pytest tests/test_highdim_instances.py -v`
Expected: PASS（全部）

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_smco_evo_manifests.py tests/test_highdim_instances.py
git commit -m "feat(generate): --extra-n-starts + n_starts-keyed starts seed" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: expand_tasks 按 config.n_starts 选 start_points_hash

**Files:**
- Modify: `src/smco/experiment_manifests.py`
- Test: `tests/test_experiment_manifests.py`

- [ ] **Step 1: 写失败测试** — 在 `tests/test_experiment_manifests.py` 末尾追加：

```python
def test_expand_tasks_selects_start_points_hash_by_n_starts():
    from smco.experiment_manifests import build_algorithm_config, expand_tasks
    cfg8 = build_algorithm_config("python", "smco", True, "state_preserving",
        evolution_strategy="rand1bin", evolution_points=(0.5, 0.75),
        elimination_rate=0.25, de_factor=0.8, de_crossover=0.7, n_starts=8)
    cfg16 = build_algorithm_config("python", "smco", True, "state_preserving",
        evolution_strategy="rand1bin", evolution_points=(0.5, 0.75),
        elimination_rate=0.25, de_factor=0.8, de_crossover=0.7, n_starts=16)
    index = {("Rastrigin", 4, 0): {
        "artifact_dir": "art", "transform_sha256": "ih",
        "start_points_hash": "hash_n8",
        "extra_starts": {"16": {"hash": "hash_n16", "n_starts": 16}},
    }}
    tasks = expand_tasks("e6_ablations", "synthetic_highdim", ["Rastrigin"], [4], 1,
                         [cfg8, cfg16], fe_budget_per_d=100, checkpoints_per_d=(100,),
                         instance_index=index)
    by_n = {t["n_starts"]: t["start_points_hash"] for t in tasks}
    assert by_n[8] == "hash_n8"
    assert by_n[16] == "hash_n16"
```

- [ ] **Step 2: 跑确认失败**

Run: `.venv/bin/python -m pytest tests/test_experiment_manifests.py::test_expand_tasks_selects_start_points_hash_by_n_starts -v`
Expected: FAIL — `AssertionError`（当前 n16 也取 "hash_n8"）

- [ ] **Step 3: 实现** — 在 `src/smco/experiment_manifests.py`：

(a) 加模块级 helper（在 `expand_tasks` 之前）：

```python
def _select_start_points_hash(entry: dict, n_starts: int):
    """Pick the start_points_hash for the requested n_starts tier.

    n_starts=8 (or an entry without extra_starts) → the default starts hash;
    other tiers → the matching extra_starts hash.
    """
    if n_starts != 8:
        extra = (entry.get("extra_starts") or {}).get(str(int(n_starts)))
        if extra:
            return extra.get("hash")
    return entry.get("start_points_hash")
```

(b) 在 `expand_tasks` 里，把 provenance 构造改为按 config.n_starts 选 hash。当前循环是 `for config in configs:` 内构造 provenance；把它移到 config 循环内并使用 helper：

```python
                for config in configs:
                    provenance: dict = {}
                    if instance_index is not None:
                        entry = instance_index.get((function, dim, instance))
                        if entry is not None:
                            provenance = {
                                "instance_artifact_dir": entry.get("artifact_dir"),
                                "instance_hash": entry.get("transform_sha256"),
                                "start_points_hash": _select_start_points_hash(entry, int(config["n_starts"])),
                            }
                    seed = derive_seed(
                        stage, suite, function, dim, instance, replication,
                        config["algorithm_id"],
                    )
                    tasks.append(
                        build_task(
                            stage, suite, function, dim, instance, replication,
                            config=config, fe_budget=fe_budget, checkpoints=checkpoints,
                            seed=seed, **provenance,
                        )
                    )
```

（即将原 `if instance_index is not None:` provenance 块从 instance 循环移入 config 循环，并用 `_select_start_points_hash` 替换固定取值。）

- [ ] **Step 4: 跑通过**

Run: `.venv/bin/python -m pytest tests/test_experiment_manifests.py -v`
Expected: PASS（全部）

- [ ] **Step 5: Commit**

```bash
git add src/smco/experiment_manifests.py tests/test_experiment_manifests.py
git commit -m "feat(manifest): expand_tasks picks start_points_hash by config n_starts" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: start_count_configs(winner, dim)

**Files:**
- Modify: `src/smco/ablations.py`
- Test: `tests/test_ablations.py`

- [ ] **Step 1: 写失败测试** — 在 `tests/test_ablations.py` 末尾追加：

```python
def test_start_count_configs_three_tiers():
    from smco.ablations import start_count_configs
    configs = start_count_configs("PY-SP-SMCO-EVO", 1000)
    labels = [label for label, _cfg in configs]
    ns = sorted({cfg["n_starts"] for _label, cfg in configs})
    assert ns == [8, 16, 32]  # ceil(sqrt(1000)) = 32
    assert set(labels) == {"n8", "n16", "n32"}
    # different n_starts → different configuration_hash
    hashes = {cfg["configuration_hash"] for _label, cfg in configs}
    assert len(hashes) == 3


def test_start_count_configs_non_evo_empty():
    from smco.ablations import start_count_configs
    assert start_count_configs("PY-BASE-SMCO", 1000) == []
```

- [ ] **Step 2: 跑确认失败**

Run: `.venv/bin/python -m pytest tests/test_ablations.py -v -k start_count`
Expected: FAIL — `ImportError: cannot import name 'start_count_configs'`

- [ ] **Step 3: 实现** — 在 `src/smco/ablations.py`：

(a) 顶部加 `import math`。

(b) 在 `ablation_configs` 之后加：

```python
def start_count_configs(winner_algorithm_id: str, dim: int) -> list[tuple[str, dict]]:
    """E6.1 start-count ablation: ``(label, config)`` for n_starts in {8, 16, ceil(sqrt(dim))}.

    8 is the control (winner default); strategy/points/rate are fixed at the EVO
    defaults so only n_starts varies. Empty for a non-EVO winner.
    """
    parsed = parse_algorithm_id(winner_algorithm_id)
    if not parsed["evolutionary"]:
        return []
    language, family, semantics = (
        parsed["language"], parsed["family"], parsed["state_semantics"]
    )
    n_list = sorted({8, 16, int(math.ceil(math.sqrt(int(dim))))})
    configs: list[tuple[str, dict]] = []
    for n in n_list:
        cfg = build_algorithm_config(
            language, family, True, semantics,
            evolution_strategy="rand1bin", evolution_points=(0.5, 0.75),
            elimination_rate=0.25, **_EVO_DEFAULTS, n_starts=n,
        )
        configs.append((f"n{n}", cfg))
    return configs
```

并把 `__all__` 加 `"start_count_configs"`。

- [ ] **Step 4: 跑通过**

Run: `.venv/bin/python -m pytest tests/test_ablations.py -v`
Expected: PASS（全部）

- [ ] **Step 5: Commit**

```bash
git add src/smco/ablations.py tests/test_ablations.py
git commit -m "feat(ablations): start_count_configs (E6.1 {8,16,ceil sqrt(d)})" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: E6.1 per-dim manifest 路径

**Files:**
- Modify: `scripts/run_smco_evo_ablations.py`
- Test: `tests/test_ablations.py`（端到端 manifest 构建）

- [ ] **Step 1: 写失败测试** — 在 `tests/test_ablations.py` 末尾追加：

```python
def test_build_start_count_ablation_manifest_per_dim(tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "abl", Path("scripts/run_smco_evo_ablations.py"))
    abl = importlib.util.module_from_spec(spec); spec.loader.exec_module(abl)
    # instance index with extra_starts for n8/16/ceil(sqrt(1000))=32
    index = {("Rastrigin", 1000, 0): {
        "artifact_dir": "art", "transform_sha256": "ih",
        "start_points_hash": "h8",
        "extra_starts": {"16": {"hash": "h16", "n_starts": 16},
                         "32": {"hash": "h32", "n_starts": 32}}}}
    idx_path = tmp_path / "idx.json"
    import json
    idx_path.write_text(json.dumps({"instances": [
        {"function": "Rastrigin", "dimension": 1000, "instance_id": 0,
         "artifact_dir": "art", "transform_sha256": "ih", "start_points_hash": "h8",
         "extra_starts": {"16": {"hash": "h16"}, "32": {"hash": "h32"}}}]}))
    manifest = abl.build_start_count_ablation_manifest(
        winner="PY-SP-SMCO-EVO", functions=["Rastrigin"], dims=[1000], n_instances=1,
        fe_budget_per_d=100, checkpoints_per_d=(100,), instances_index=idx_path)
    assert manifest["frozen"] is True
    ns = sorted({t["n_starts"] for t in manifest["tasks"]})
    assert ns == [8, 16, 32]
    # each task's start_points_hash matches its tier
    by_n = {t["n_starts"]: t["start_points_hash"] for t in manifest["tasks"]}
    assert by_n == {8: "h8", 16: "h16", 32: "h32"}
```

- [ ] **Step 2: 跑确认失败**

Run: `.venv/bin/python -m pytest tests/test_ablations.py::test_build_start_count_ablation_manifest_per_dim -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'build_start_count_ablation_manifest'`

- [ ] **Step 3: 实现** — 在 `scripts/run_smco_evo_ablations.py`：

(a) import 改为也引入 `start_count_configs`：

```python
from smco.ablations import ablation_configs, start_count_configs
```

(b) 在 `build_ablation_manifest` 之后加：

```python
def build_start_count_ablation_manifest(
    *, winner, functions, dims, n_instances, fe_budget_per_d,
    checkpoints_per_d, instances_index=None, out_dir=None,
):
    """E6.1: per-dim expand (ceil(sqrt(d)) is dimension-dependent)."""
    index = load_instance_index(instances_index) if instances_index else None
    all_tasks = []
    for function in functions:
        for dim in dims:
            dim = int(dim)
            configs = [cfg for _label, cfg in start_count_configs(winner, dim)]
            if not configs:
                raise ValueError(f"winner {winner!r} is not EVO; ablations are EVO-only")
            all_tasks.extend(expand_tasks(
                "e6_ablations", "synthetic_highdim", [function], [dim], n_instances, configs,
                fe_budget_per_d=fe_budget_per_d, checkpoints_per_d=checkpoints_per_d,
                instance_index=index,
            ))
    manifest = freeze_manifest(build_manifest("e6_ablations", "synthetic_highdim", all_tasks))
    if out_dir is not None:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "e6_ablations__synthetic_highdim.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest
```

(c) `main` 加 `--dimension` 选择路径（在现有 `build_ablation_manifest` 调用处替换）：

```python
    parser.add_argument(
        "--dimension", default="schedule", choices=["strategy", "schedule", "start_count"],
        help="E6 ablation dimension.",
    )
```

并在 `main` 里替换 manifest 构建为按 dimension 分派：

```python
    out = None if args.dry_run else args.out_dir
    if args.dimension == "start_count":
        manifest = build_start_count_ablation_manifest(
            winner=args.winner, functions=args.functions, dims=args.dims,
            n_instances=args.n_instances, fe_budget_per_d=args.fe_budget_per_d,
            checkpoints_per_d=args.checkpoints_per_d, instances_index=args.instances_index,
            out_dir=out)
        n_configs = len(start_count_configs(args.winner, int(args.dims[0])))
    else:
        manifest = build_ablation_manifest(
            winner=args.winner, functions=args.functions, dims=args.dims,
            n_instances=args.n_instances, fe_budget_per_d=args.fe_budget_per_d,
            checkpoints_per_d=args.checkpoints_per_d, instances_index=args.instances_index,
            out_dir=out)
        n_configs = len(ablation_configs(args.winner))
    print(
        f"ablation manifest ({args.dimension}): {len(manifest['tasks'])} tasks "
        f"(~{n_configs} configs/dim x {len(args.functions)} funcs x {len(args.dims)} dims "
        f"x {args.n_instances} instances), frozen={manifest['frozen']}"
    )
    return 0
```

（删除原打印块，用上面统一打印。）

- [ ] **Step 4: 跑通过**

Run: `.venv/bin/python -m pytest tests/test_ablations.py -v`
Expected: PASS（全部）

- [ ] **Step 5: Commit**

```bash
git add scripts/run_smco_evo_ablations.py tests/test_ablations.py
git commit -m "feat(ablations): E6.1 per-dim start-count manifest path" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 6: _identity_key 增 n_starts（merge_results）

**Files:**
- Modify: `src/smco/merge_results.py`
- Test: `tests/test_merge_results.py`

- [ ] **Step 1: 写失败测试** — 在 `tests/test_merge_results.py`：

(a) `_row` helper 加 `"n_starts": 8` 字段（在 base dict 里 `"seed": 1,` 之后加一行 `"n_starts": 8,`）。

(b) 末尾追加：

```python
def test_identity_key_distinguishes_n_starts():
    a = _row("r1"); b = _row("r2", n_starts=16)
    assert _identity_key(a) != _identity_key(b)
```

- [ ] **Step 2: 跑确认失败**

Run: `.venv/bin/python -m pytest tests/test_merge_results.py -v -k identity_key_distinguishes`
Expected: FAIL — `KeyError: 'n_starts'`（_row 缺字段；或 _identity_key 不含 n_starts 导致两键相等）

- [ ] **Step 3: 实现** — 在 `src/smco/merge_results.py` 的 `_identity_key` 末尾加 `int(row["n_starts"])`：

```python
def _identity_key(row: dict) -> tuple:
    """Identity (excluding run_id) — same key => duplicate unless supersedes."""
    return (
        row["function"], int(row["dimension"]), int(row["instance"]),
        row["algorithm_id"], row["language"], row["state_semantics"],
        row["evolution_strategy"], int(row["seed"]), int(row["n_starts"]),
    )
```

- [ ] **Step 4: 跑通过**

Run: `.venv/bin/python -m pytest tests/test_merge_results.py -v`
Expected: PASS（全部）

- [ ] **Step 5: Commit**

```bash
git add src/smco/merge_results.py tests/test_merge_results.py
git commit -m "fix(merge): _identity_key includes n_starts (E6.1 tiers distinct)" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 7: factorial runner load_starts(n_starts) + provenance

**Files:**
- Modify: `scripts/run_smco_evo_highdim_factorial.py`
- Test: `tests/test_highdim_worker.py`（端到端 n≠8 加载）

- [ ] **Step 1: 写失败测试** — 在 `tests/test_highdim_worker.py` 末尾追加：

```python
def test_run_task_file_loads_extra_starts_tier(tmp_path):
    cli = _load_worker_cli()
    inst = generate_instance("Rastrigin", 4, 0, seed=1)
    rng = np.random.default_rng(5); span = inst.bounds_upper - inst.bounds_lower
    starts8 = inst.bounds_lower + rng.uniform(size=(8, 4)) * span
    starts16 = inst.bounds_lower + rng.uniform(size=(16, 4)) * span
    art_dir = tmp_path / "instances" / "dev_Rastrigin_d4_i0"
    meta = write_instance_artifacts(inst, starts8, art_dir, extra_starts={16: starts16})
    cfg = build_algorithm_config("python", "smco", True, "state_preserving",
        evolution_strategy="rand1bin", evolution_points=(0.5, 0.75),
        elimination_rate=0.25, de_factor=0.8, de_crossover=0.7, n_starts=16)
    task = build_task("e6_ablations", "synthetic_highdim", "Rastrigin", 4, 0, 0,
        config=cfg, fe_budget=200, checkpoints=(100, 200), seed=42,
        instance_artifact_dir="instances/dev_Rastrigin_d4_i0",
        instance_hash=meta["transform_sha256"],
        start_points_hash=meta["extra_starts"]["16"]["hash"])
    (tmp_path / "task.json").write_text(json.dumps(task))
    rc = cli.run_task_file(str(tmp_path / "task.json"), instance_root=str(tmp_path),
                           result_dir=str(tmp_path / "raw"), log_dir=str(tmp_path / "logs"))
    assert rc == 0
    payload = json.loads((tmp_path / "raw" / f"{task['run_id']}.json").read_text())
    assert payload["status"] == "success"
    assert payload["task"]["n_starts"] == 16
```

- [ ] **Step 2: 跑确认失败**

Run: `.venv/bin/python -m pytest tests/test_highdim_worker.py::test_run_task_file_loads_extra_starts_tier -v`
Expected: FAIL — provenance mismatch（runner 校验 starts.csv.gz 而非 starts_n16.csv.gz）或 starts 形状不符

- [ ] **Step 3: 实现** — 在 `scripts/run_smco_evo_highdim_factorial.py` 的 `_verify_provenance`，把 starts hash 校验改为按 n_starts 选文件：

```python
    expected_starts_hash = task.get("start_points_hash")
    if expected_starts_hash:
        n_starts = int(task.get("n_starts", 8))
        starts_file = "starts.csv.gz" if n_starts == 8 else f"starts_n{n_starts}.csv.gz"
        actual = _sha256_file(inst_dir / starts_file)
        if actual != expected_starts_hash:
            raise ValueError(
                f"start_points_hash mismatch (n_starts={n_starts}): "
                f"task={expected_starts_hash!r} artifact={actual!r}"
            )
```

并在 `run_task_file` 里把 `starts = load_starts(inst_dir)` 改为：

```python
        starts = load_starts(inst_dir, n_starts=int(task.get("n_starts", 8)))
```

- [ ] **Step 4: 跑通过**

Run: `.venv/bin/python -m pytest tests/test_highdim_worker.py -v`
Expected: PASS（全部）

- [ ] **Step 5: Commit**

```bash
git add scripts/run_smco_evo_highdim_factorial.py tests/test_highdim_worker.py
git commit -m "feat(runner): load starts by task n_starts + per-tier provenance" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 8: R loader n_starts + worker 传参 + 本机重验

**Files:**
- Modify: `vendor/SMCO_R/main/highdim_instances.R`
- Modify: `scripts/run_smco_evo_highdim_r.R`

- [ ] **Step 1: 实现** —

(a) `vendor/SMCO_R/main/highdim_instances.R`：`load_highdim_instance` 签名加 `n_starts = 8L`，并把读 starts 那行改为按 n_starts 选文件：

```r
load_highdim_instance <- function(artifact_dir, function_name, dimension,
                                  asymmetry_strength, objective_scale,
                                  known_optimum_value, n_starts = 8L) {
```

把 `starts <- .read_gz_matrix(file.path(artifact_dir, "starts.csv.gz"))` 改为：

```r
  starts_file <- if (as.integer(n_starts) == 8L) "starts.csv.gz"
                 else paste0("starts_n", as.integer(n_starts), ".csv.gz")
  starts_path <- file.path(artifact_dir, starts_file)
  if (!file.exists(starts_path))
    stop("no starts artifact for n_starts=", n_starts, " in ", artifact_dir)
  starts <- .read_gz_matrix(starts_path)
```

(b) `scripts/run_smco_evo_highdim_r.R`：把 `load_highdim_instance(...)` 调用加 `n_starts = as.integer(.task$n_starts)`：

```r
  .inst <- load_highdim_instance(
    .inst_dir, .task[["function"]], as.integer(.task$dimension),
    as.numeric(.meta$asymmetry_strength), as.numeric(.meta$objective_scale),
    as.numeric(.meta$known_optimum_value), n_starts = as.integer(.task$n_starts)
  )
```

- [ ] **Step 2: R 语法校验**

Run: `Rscript -e 'parse(file="scripts/run_smco_evo_highdim_r.R"); parse(file="vendor/SMCO_R/main/highdim_instances.R"); cat("parse OK\n")'`
Expected: `parse OK`

- [ ] **Step 3: 本机端到端重验 n16** — 新建 `/tmp/verify_r_e6_startcount.py`：

```python
import json, subprocess, sys, math
from pathlib import Path
import numpy as np
sys.path.insert(0, "src")
from smco.experiment_manifests import build_algorithm_config, build_task
from smco.highdim_instances import generate_instance, write_instance_artifacts

root = Path("/tmp/r_e6_startcount"); root.mkdir(exist_ok=True)
inst = generate_instance("Rastrigin", 4, 0, seed=1)
rng = np.random.default_rng(5); span = inst.bounds_upper - inst.bounds_lower
s8 = inst.bounds_lower + rng.uniform(size=(8, 4)) * span
s16 = inst.bounds_lower + rng.uniform(size=(16, 4)) * span
art = root / "instances" / "dev_Rastrigin_d4_i0"
meta = write_instance_artifacts(inst, s8, art, extra_starts={16: s16})
cfg = build_algorithm_config("r", "smco", True, "state_preserving",
    evolution_strategy="rand1bin", evolution_points=(0.5, 0.75),
    elimination_rate=0.25, de_factor=0.8, de_crossover=0.7, n_starts=16)
task = build_task("e6_ablations", "synthetic_highdim", "Rastrigin", 4, 0, 0, config=cfg,
    fe_budget=200, checkpoints=(100, 200), seed=42,
    instance_artifact_dir="instances/dev_Rastrigin_d4_i0",
    instance_hash=meta["transform_sha256"], start_points_hash=meta["extra_starts"]["16"]["hash"])
(root / "task.json").write_text(json.dumps(task))
raw = root / "raw"; log = root / "logs"; raw.mkdir(exist_ok=True); log.mkdir(exist_ok=True)
subprocess.run(["Rscript", "scripts/run_smco_evo_highdim_r.R", "--task", str(root / "task.json"),
    "--instance-root", str(root), "--result-dir", str(raw), "--log-dir", str(log)], check=True)
payload = json.loads((raw / f"{task['run_id']}.json").read_text())
assert payload["status"] == "success", payload
assert payload["task"]["n_starts"] == 16
print("R E6.1 n16 verify OK")
```

- [ ] **Step 4: 跑重验**

Run: `.venv/bin/python /tmp/verify_r_e6_startcount.py`
Expected: `R E6.1 n16 verify OK`

- [ ] **Step 5: Commit**

```bash
git add vendor/SMCO_R/main/highdim_instances.R scripts/run_smco_evo_highdim_r.R
git commit -m "feat(R): load_highdim_instance(n_starts) + worker passes task n_starts" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 9: 全量 pytest

**Files:** Verify only.

- [ ] **Step 1: 全量**

Run: `.venv/bin/python -m pytest -q`
Expected: 全绿（Task 11 后基线 348 + 本计划新增测试；无 fail/error）。红则修，不在红时 commit。

- [ ] **Step 2: 冒烟 E6.1 manifest 生成（CLI）**

Run:
```bash
.venv/bin/python scripts/generate_smco_evo_manifests.py --stage instances \
  --suite-stage development --functions Rastrigin --dims 1000 --n-instances 1 \
  --extra-n-starts 16 sqrt --out-dir /tmp/e6_smoke
.venv/bin/python scripts/run_smco_evo_ablations.py --dimension start_count \
  --winner PY-SP-SMCO-EVO --functions Rastrigin --dims 1000 --n-instances 1 \
  --instances-index /tmp/e6_smoke/instances_index.json --dry-run
```
Expected: dry-run 报告 ~3 configs/dim（n8/n16/n32）× 1 func × 1 dim × 1 instance = 3 tasks。

---

## Self-Review（已完成）

- **Spec coverage**：§4.2 artifact → Task 1；§4.2 seed + §4.3 generate → Task 2；§4.3 expand_tasks → Task 3；§4.3 start_count_configs → Task 4；§4.3 E6.1 manifest → Task 5；§4.5 identity_key → Task 6；§4.3 runner → Task 7；§4.4 R → Task 8。全覆盖。
- **Placeholder scan**：无 TBD/TODO；每步含完整代码。
- **Type consistency**：`start_count_configs(winner, dim)` 在 Task 4/5 一致；`build_start_count_ablation_manifest` 在 Task 5 定义；`_select_start_points_hash`/`_resolve_n_starts`/`load_starts(dir,n_starts)` 签名跨 task 一致。
