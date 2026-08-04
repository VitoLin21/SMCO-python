#!/bin/bash
# E3-F per-node launcher. Usage: launch_e3f.sh <workers_per_shard> <shard-id1> [<shard-id2> ...]
# Must run with cwd = node REPO (or REPO env set). Starts one nohup dispatch per shard.
W=$1; shift
REPO=${REPO:-$(pwd)}
cd "$REPO" || { echo "cd $REPO failed"; exit 1; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=src
MID=$(hostname -I | awk '{print $1}')
EHS=$(.venv/bin/python -c 'from smco.provenance import default_environment_hash as f;print(f())' 2>/dev/null)
SHA=650c9cc330649ebeef4f16c80d9567850224ca07
EXT=result/smco-evo-ultrahighdim-2026/e3f
for SH in "$@"; do
  NUM=${SH#shard-}
  nohup .venv/bin/python scripts/run_smco_evo_ultrahighdim_extension.py dispatch \
    --manifest $EXT/manifest.json --instance-root "$REPO" \
    --evidence-root $EXT/evidence_${MID}_s${NUM} \
    --shards $EXT/shards.json --shard-id $SH \
    --workers $W --machine-id "$MID" \
    --git-commit $SHA --environment-hash "$EHS" \
    > $EXT/dispatch_${MID}_s${NUM}.log 2>&1 &
  echo "launched $SH -> evidence_${MID}_s${NUM} pid=$!"
done
