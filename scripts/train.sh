#!/usr/bin/env bash
# Train GSCV stage by stage: S0 -> S1a -> S1b -> S2a -> S2b -> S3_disc -> S3, each stage starting from final.pt of
# the previous one (S0 from the released GVC1D weights; S3_disc and S3 both from S2b, S3 with the discriminator of
# S3_disc). Run from anywhere; 4 GPUs by default.
#   bash scripts/train.sh                                  # every stage, into runs/gscv
#   GPUS=0,1,2,3 STAGES="S0 S1a" bash scripts/train.sh     # some stages on the given GPUs
#   bash scripts/train.sh data.num_workers=4               # extra arguments: OmegaConf overrides for every stage
# Every stage is checked first (scripts/train.py --status):
#   new / resume  : trained, or continued from <out>/ckpt.pt after an interruption
#   extend        : passes were appended to a finished stage in the configuration: they are trained
#   done          : skipped
#   stale         : its init weights changed since it started (an upstream stage was trained again or extended)
#   missing_init  : its init weights (the previous stage's final.pt) do not exist yet
#   mismatch      : it finished other passes than the configuration lists
# A stale or mismatching stage stops the run; FORCE=1 moves the directories of the stages in STAGES aside
# (<stage>.old_<time>) and trains them again.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
NPROC=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
CONFIG=${CONFIG:-configs/gscv.yaml}
OUT_ROOT=${OUT_ROOT:-runs/gscv}
OPTS=("train.output_root=$OUT_ROOT")
mkdir -p "$OUT_ROOT"
for STAGE in ${STAGES:-S0 S1a S1b S2a S2b S3_disc S3}; do
    if [ -n "${FORCE:-}" ] && [ -d "$OUT_ROOT/$STAGE" ]; then
        OLD="$OUT_ROOT/$STAGE.old_$(date +%Y%m%d_%H%M%S)"
        mv "$OUT_ROOT/$STAGE" "$OLD"
        echo "FORCE=1: moved $OUT_ROOT/$STAGE to $OLD"
    fi
    STATUS=$(python scripts/train.py --config "$CONFIG" --stage "$STAGE" --status "${OPTS[@]}" "$@" | tail -1)
    case "$STATUS" in
        done) echo "==== $STAGE already finished ($OUT_ROOT/$STAGE)"; continue ;;
        new|resume|extend) ;;
        stale|mismatch)
            echo "==== $STAGE is $STATUS: move $OUT_ROOT/$STAGE away (or run it with FORCE=1)"; exit 1 ;;
        missing_init)
            echo "==== $STAGE cannot start: its init weights (the previous stage's final.pt) are missing;"
            echo "     train the previous stage first (STAGES lists the stages in order)"; exit 1 ;;
        *) echo "==== status check of $STAGE failed: '$STATUS'"; exit 1 ;;
    esac
    echo "==== $STAGE ($STATUS) on GPUs $CUDA_VISIBLE_DEVICES ($(date))"
    accelerate launch --num_processes "$NPROC" --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
        --main_process_port "${PORT:-29531}" \
        scripts/train.py --config "$CONFIG" --stage "$STAGE" "${OPTS[@]}" "$@" \
        2>&1 | tee -a "$OUT_ROOT/train_${STAGE}.out"
done
