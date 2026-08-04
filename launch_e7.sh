#!/bin/bash
# E7 per-node launcher. Usage: launch_e7.sh <workers_per_shard> <shard-id1> [...]
# Runs in R nodes only (m253/本机 不跑 E7 R). E7 = 1736 physically-new tasks.
# Total-core cap: set MAX_WORKERS env per node (handoff hard limits: 本机=12,
# zf*=76/64, m251/m253=48, math@ focal=按 48 核取半). Refuses to over-subscribe.
W=$1; shift
REPO=${REPO:-$(pwd)}
cd "$REPO" || { echo "cd $REPO failed"; exit 1; }
N_SHARDS=$#
TOTAL_WORKERS=$(( W * N_SHARDS ))
if [ -n "$MAX_WORKERS" ] && [ "$TOTAL_WORKERS" -gt "$MAX_WORKERS" ]; then
  echo "REFUSE: total workers $TOTAL_WORKERS (=$W x $N_SHARDS shards) exceeds MAX_WORKERS=$MAX_WORKERS" >&2
  exit 1
fi
echo "launch: $N_SHARDS shards x $W workers = $TOTAL_WORKERS workers${MAX_WORKERS:+ (cap $MAX_WORKERS)}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=src
MID=$(hostname -I | awk '{print $1}')
EHS=$(.venv/bin/python -c 'from smco.provenance import default_environment_hash as f;print(f())' 2>/dev/null)
SHA=1da43e5f0bcd9393d9f1a9a9764244024adfc56f
EXT=result/smco-evo-ultrahighdim-2026/e7
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
