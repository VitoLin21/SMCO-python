# E4/E5 formal COCO external evidence contract

## Scope and isolation

E4 and E5 are external supporting checks. They do not amend or replace the
already frozen high-dimensional primary index at
`result/smco-evo-paper-highdim-2026/canonical_artifacts.json`. Each campaign
instead has its own frozen external index and code-owned contract:

- E4: `e4_formal_manifest` + `e4_formal_merged`, 2,520 rows,
  `bbob-largescale`, stage `e4_bbob_largescale`.
- E5: `e5_formal_manifest` + `e5_formal_merged`, 480 rows, `bbob`, stage
  `e5_lowdim_check`.

The external index is `external_supporting`, not a source of the E1--E3 main
claim. E4 only validates the frozen winner directly when its provenance says
`ran_language=python`, `is_frozen_winner_validation=true`, and
`external_check_kind=frozen_winner`.

## Freeze gate

After a complete task-level merge and its 12-check audit passes, freeze only a
new external index:

```bash
.venv/bin/python scripts/freeze_smco_evo_external_coco_artifacts.py \
  --campaign e4 \
  --manifest result/e4-2026-07-31/e4_bbob_largescale__bbob-largescale.json \
  --merged-dir result/e4-2026-08-02/formal/merged \
  --out result/e4-2026-08-02/formal/external_canonical_artifacts.json
```

Use the analogous E5 manifest and `result/e5-2026-08-02/formal/merged` for
`--campaign e5`. The command never writes the primary canonical index.

The external validator binds all of the following:

1. the byte hash and internal frozen hash of the manifest;
2. `valid_runs.csv` and `provenance_audit.json` hashes plus a passing 12-check
   audit;
3. the audit's `manifest_sha256`, exact manifest run-id set and valid-run set;
4. `coco_native_runs.csv` hash, exact run-id set and expected row count.

Thus a result/audit from another manifest, a missing or duplicate native row,
or a post-freeze byte change fails the external index.

## Analysis boundary

Current cocoex installations do not expose an auditable per-instance optimum.
Formal outcomes therefore use `metric_mode=coco_native`: no normalized gap and
no synthetic relative target times. The generic Task-12 primary-table command
cannot resolve this index or analysis kind.

Use only the dedicated native route:

```bash
.venv/bin/python scripts/analyze_smco_evo_coco_external.py \
  --external-index result/e4-2026-08-02/formal/external_canonical_artifacts.json \
  --artifact-key e4_formal_merged \
  --out-dir result/e4-2026-08-02/formal/analysis
```

It reports only completion, official COCO `final_target_hit`, and FE use by
algorithm. It intentionally does not pool raw objective values across BBOB
functions and does not produce normalized-gap ECDF, relative-target ERT, or
the high-dimensional primary table.
