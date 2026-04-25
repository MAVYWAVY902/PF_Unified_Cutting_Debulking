#!/usr/bin/env bash
# Post-training diagnostic + deploy-prep pipeline.
#
# Three steps in series, all single-GPU:
#   1. monitor_nan.py           NaN-guard frequency + breach correlation
#                               (gates the rest — refuses to continue if BAD verdict)
#   2. eval_checkpoint_sweep.py 11-checkpoint sweep eval → trajectory plot + summary CSV
#   3. render_rollout.py × 2    Pareto-preservation + aggressive deploy videos
#                               (auto-dedup if both pick the same checkpoint)
#
# Usage:
#   bash tools/post_train.sh <run_name> [nan_eps=50] [sweep_eps=20]
#
# Override gating:
#   FORCE=1 bash tools/post_train.sh <run_name>     # skip NaN-verdict gate
#   SKIP_NAN=1 bash tools/post_train.sh <run_name>  # don't run step 1 at all
#
# Wall-clock estimate (RTX 2070, torus ep=1024 @ ~50 fps):
#   step 1  ~17 min   (50 eps  × 1024 / 50)
#   step 2  ~70 min   (11 ckpt ×   20 × 1024 / 50)
#   step 3  ~10 min   (2 videos × 1024 frames + ffmpeg)
#   total   ~1h 40min  (no parallelism — same GPU)

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash tools/post_train.sh <run_name> [nan_eps=50] [sweep_eps=20]" >&2
    exit 1
fi

RUN_NAME="$1"
NAN_EPS="${2:-50}"
SWEEP_EPS="${3:-20}"

REPO="/home/yunxin/pf-shape-matching"
PY="/home/yunxin/miniconda3/envs/debulk_warp/bin/python"
CKPT_DIR="$REPO/checkpoints/$RUN_NAME"
SWEEP_SUM="$REPO/results/sweep/$RUN_NAME/${RUN_NAME}_summary.csv"
NAN_SUM="$REPO/results/nan_monitor/$RUN_NAME/summary.json"

cd "$REPO"

# ---------------- pre-flight ----------------
if [[ ! -f "$CKPT_DIR/${RUN_NAME}_final.zip" ]]; then
    echo "[error] $CKPT_DIR/${RUN_NAME}_final.zip not found." >&2
    echo "[error] Training has not finished. Aborting." >&2
    exit 2
fi
if [[ ! -x "$PY" ]]; then
    echo "[error] python not at $PY" >&2
    exit 2
fi

START_TS=$(date +%s)
echo
echo "============================================================"
echo "  post_train.sh — $RUN_NAME"
echo "  started   $(date -d "@$START_TS" '+%Y-%m-%d %H:%M:%S')"
echo "  NaN eps   $NAN_EPS"
echo "  sweep eps $SWEEP_EPS"
echo "============================================================"

# ---------------- step 1: NaN guard ----------------
if [[ "${SKIP_NAN:-0}" == "1" ]]; then
    echo
    echo "[1/3] SKIPPED (SKIP_NAN=1)"
else
    echo
    echo "[1/3] NaN-guard monitor   (~17 min)"
    echo "------------------------------------------------------------"
    "$PY" tools/monitor_nan.py --run-name "$RUN_NAME" --episodes "$NAN_EPS"

    if [[ ! -f "$NAN_SUM" ]]; then
        echo "[warn] $NAN_SUM not produced; cannot gate. Continuing." >&2
    else
        VERDICT=$("$PY" -c "
import json
s = json.load(open('$NAN_SUM'))
e = s.get('eval')
print(e.get('verdict', 'NO_DATA') if e else 'NO_DATA')
")
        echo
        echo "[1/3] verdict: $VERDICT"

        if [[ "$VERDICT" =~ ^BAD ]] && [[ "${FORCE:-0}" != "1" ]]; then
            echo
            echo "[abort] NaN guard verdict is BAD — guard is hiding upstream bug." >&2
            echo "[abort] Skipping sweep + rollout to avoid contaminating the data trail." >&2
            echo "[abort] Override: FORCE=1 bash tools/post_train.sh $RUN_NAME" >&2
            exit 3
        fi
    fi
fi

# ---------------- step 2: checkpoint sweep ----------------
echo
echo "[2/3] checkpoint sweep eval  (~70 min)"
echo "------------------------------------------------------------"
"$PY" tools/eval_checkpoint_sweep.py --run-name "$RUN_NAME" --episodes "$SWEEP_EPS"

if [[ ! -f "$SWEEP_SUM" ]]; then
    echo "[warn] sweep summary missing at $SWEEP_SUM — skipping rollout videos." >&2
    exit 0
fi

# ---------------- step 3: rollout videos ----------------
echo
echo "[3/3] rollout videos"
echo "------------------------------------------------------------"

# Identify the two deploy-point candidates from the sweep:
#   • preservation = highest interior with cap ≤ budget   (render_rollout --best default)
#   • aggressive   = highest interior with cap ≤ 1.0     (= highest interior overall)
# If both pick the same step, the second render is skipped.
PRESERVATION_STEP=$("$PY" -c "
import csv
rows = list(csv.DictReader(open('$SWEEP_SUM')))
safe = [r for r in rows if float(r['cap_mean']) <= 0.05]
pool = safe if safe else rows
best = max(pool, key=lambda r: float(r['interior_mean']))
print('final' if best.get('is_final', '').lower() in ('true', '1') else best['step'])
" 2>/dev/null || echo "")

AGGRESSIVE_STEP=$("$PY" -c "
import csv
rows = list(csv.DictReader(open('$SWEEP_SUM')))
best = max(rows, key=lambda r: float(r['interior_mean']))
print('final' if best.get('is_final', '').lower() in ('true', '1') else best['step'])
" 2>/dev/null || echo "")

echo "  preservation candidate : step=$PRESERVATION_STEP"
echo "  aggressive   candidate : step=$AGGRESSIVE_STEP"

render_step () {
    local step="$1"
    local label="$2"
    if [[ -z "$step" ]]; then
        echo "  [skip] $label — empty step value"
        return
    fi
    if [[ "$step" == "final" ]]; then
        local tag="final"
    else
        local tag=$(printf "step%06d" "$step")
    fi
    local out_dir="$REPO/rollout_frames_${RUN_NAME}_${tag}"
    if [[ -f "$out_dir/rollout.mp4" ]]; then
        echo "  [skip] $label — $out_dir/rollout.mp4 already exists"
        return
    fi
    echo
    echo "  [$label] rendering step=$step → $out_dir"
    "$PY" tools/render_rollout.py --run-name "$RUN_NAME" --step "$step" || {
        echo "  [warn] render failed for step=$step" >&2
    }
}

render_step "$PRESERVATION_STEP" "preservation"
if [[ "$AGGRESSIVE_STEP" != "$PRESERVATION_STEP" ]]; then
    render_step "$AGGRESSIVE_STEP" "aggressive"
else
    echo
    echo "  [aggressive] same step as preservation — single deploy point, no second video"
fi

# ---------------- summary ----------------
END_TS=$(date +%s)
DUR=$((END_TS - START_TS))
HRS=$((DUR / 3600))
MIN=$(( (DUR % 3600) / 60 ))

echo
echo "============================================================"
echo "  done in ${HRS}h ${MIN}min."
echo
echo "  artifacts:"
echo "    $REPO/results/nan_monitor/$RUN_NAME/"
echo "      ├── summary.json"
echo "      ├── nan_events.csv"
echo "      └── nan_diagnostic.png"
echo "    $REPO/results/sweep/$RUN_NAME/"
echo "      ├── ${RUN_NAME}_summary.csv"
echo "      ├── ${RUN_NAME}_trajectory.png"
echo "      └── ${RUN_NAME}_step*.csv"
echo "    $REPO/rollout_frames_${RUN_NAME}_step*/rollout.mp4"
echo "============================================================"
