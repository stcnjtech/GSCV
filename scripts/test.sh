#!/usr/bin/env bash
# Test GSCV on the four test sets of the paper (96 frames, intra period 32, quality indices 0-7), then FID / KID.
# One process per GPU, each coding a balanced shard of the sequences:
#   bash scripts/test.sh                       # GPUs 0-3, results in results/gscv and results/fid_kid
#   GPUS="0 1" bash scripts/test.sh
set -euo pipefail
cd "$(dirname "$0")/.."
read -r -a G <<< "${GPUS:-0 1 2 3}"
SETS=${SETS:-"data/testsets/AOM_B data/testsets/HEVC_F data/testsets/HEVC_SCC data/scvcd5k/SCVCD5K_TEST64"}
OUT=${OUT:-results/gscv}
mkdir -p "$OUT"
for i in "${!G[@]}"; do
    # shellcheck disable=SC2086
    python scripts/test.py --sets $SETS --frames 96 --intra_period 32 --qps 0 1 2 3 4 5 6 7 --fid_kid \
        --gpu "${G[$i]}" --shard "$i/${#G[@]}" --out "$OUT" > "$OUT.shard$i.log" 2>&1 &
done
wait
python scripts/test.py --merge --out "$OUT"
# shellcheck disable=SC2086
python scripts/fid_kid.py --sets $SETS --results GSCV="$OUT" --gpu "${G[0]}" --out results/fid_kid
