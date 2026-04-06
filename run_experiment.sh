#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# run_experiment.sh - Complete Experiment Pipeline
#
# Usage:
#   ./run_experiment.sh <method> [options]
#
# Methods:
#   fullft    Full Fine-Tuning
#   bitfit    BitFit (Bias-only)
#   lora      LoRA
#
# Options:
#   --dataset <sst2|qnli|mrpc>      Dataset (default: sst2)
#   --use_governor <CPU_IDX> <GPU_IDX>  Enable DVFS governor
#   --no_gsheet                     Skip Google Sheets upload
#   --log_steps <N>                 Log events every N steps (default: 100)
#   --epochs <N>                    Number of epochs (default: 3)
#   --batch_size <N>                Batch size (default: 32)
#
# Examples:
#   ./run_experiment.sh fullft
#   ./run_experiment.sh bitfit --dataset qnli
#   ./run_experiment.sh lora --dataset mrpc --use_governor 15 7
#   ./run_experiment.sh fullft --dataset sst2 --use_governor 20 10 --epochs 3
# =============================================================================

# ─────────────────────────────────────────────
# CONFIG (edit these if needed)
# ─────────────────────────────────────────────
BASE="/home/nvidia/Desktop/llm-dvfs"
PYTHON_BIN="/home/nvidia/llm-dvfs-env/bin/python3"
MEASURE_INTERVAL=100          # tegrastats interval in ms
LOG_STEPS=100                 # event log every N training steps

# Script mapping
declare -A METHOD_MAP
METHOD_MAP["fullft"]="BERT_FullFT_events_log.py"
METHOD_MAP["bitfit"]="BERT_BitFit_events_log.py"
METHOD_MAP["lora"]="BERT_lora_events_log.py"

# ─────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────
if [ $# -lt 1 ]; then
    echo ""
    echo "Usage: $0 <method> [options]"
    echo ""
    echo "Methods:  fullft | bitfit | lora"
    echo "Datasets: sst2 (default) | qnli | mrpc"
    echo ""
    echo "Options:"
    echo "  --dataset <name>               Dataset to use"
    echo "  --use_governor <CPU> <GPU>     Enable DVFS (e.g. --use_governor 15 7)"
    echo "  --no_gsheet                    Skip Google Sheets upload"
    echo "  --log_steps <N>                Log event every N steps (default: $LOG_STEPS)"
    echo "  --epochs <N>                   Training epochs"
    echo "  --batch_size <N>               Batch size"
    echo ""
    echo "Examples:"
    echo "  $0 fullft"
    echo "  $0 bitfit --dataset qnli"
    echo "  $0 lora --dataset mrpc --use_governor 15 7"
    echo ""
    exit 1
fi

# Method
METHOD_KEY="${1,,}"
shift

if [[ ! -v METHOD_MAP[$METHOD_KEY] ]]; then
    echo "❌ Unknown method: '$METHOD_KEY'. Choose from: fullft, bitfit, lora"
    exit 1
fi

SCRIPT="${METHOD_MAP[$METHOD_KEY]}"

# Defaults
DATASET="sst2"
USE_GOV=false
CPU_IDX=0
GPU_IDX=0
UPLOAD_GSHEET=true
PY_ARGS=()

# Parse remaining args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            DATASET="${2,,}"
            shift 2
            ;;
        --use_governor)
            USE_GOV=true
            CPU_IDX="$2"
            GPU_IDX="$3"
            shift 3
            ;;
        --no_gsheet)
            UPLOAD_GSHEET=false
            shift
            ;;
        --log_steps)
            LOG_STEPS="$2"
            shift 2
            ;;
        *)
            PY_ARGS+=("$1")
            shift
            ;;
    esac
done

# Validate dataset
if [[ ! "$DATASET" =~ ^(sst2|qnli|mrpc)$ ]]; then
    echo "❌ Unknown dataset: '$DATASET'. Choose from: sst2, qnli, mrpc"
    exit 1
fi

# ─────────────────────────────────────────────
# OUTPUT DIRECTORY & EXPERIMENT NAME
# ─────────────────────────────────────────────
DATE=$(date +"%Y%m%d_%H%M%S")

EXP_NAME="${METHOD_KEY}_${DATASET}"
if [ "$USE_GOV" = true ]; then
    EXP_NAME="${EXP_NAME}_FPG_${CPU_IDX}_${GPU_IDX}"
fi

OUTDIR="$BASE/runs/${EXP_NAME}_${DATE}"
mkdir -p "$OUTDIR"

# ─────────────────────────────────────────────
# BANNER
# ─────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║           EXPERIMENT PIPELINE START             ║"
echo "╠══════════════════════════════════════════════════╣"
printf  "║  Experiment : %-35s║\n" "$EXP_NAME"
printf  "║  Method     : %-35s║\n" "$METHOD_KEY ($SCRIPT)"
printf  "║  Dataset    : %-35s║\n" "$DATASET"
printf  "║  DVFS       : %-35s║\n" "$([ "$USE_GOV" = true ] && echo "CPU=$CPU_IDX GPU=$GPU_IDX" || echo "disabled")"
printf  "║  GSheets    : %-35s║\n" "$([ "$UPLOAD_GSHEET" = true ] && echo "yes" || echo "no")"
printf  "║  Output     : %-35s║\n" "$OUTDIR"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ─────────────────────────────────────────────
# STEP 1: DVFS GOVERNOR (OPTIONAL)
# ─────────────────────────────────────────────
PID_GOV=""
if [ "$USE_GOV" = true ]; then
    echo "━━━ [1/5] DVFS GOVERNOR ━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    GOV_BIN="$BASE/governorv1true"
    if [ ! -f "$GOV_BIN" ]; then
        echo "⚠️  Governor binary not found: $GOV_BIN"
        echo "⚠️  Continuing WITHOUT governor..."
        USE_GOV=false
    else
        if [ ! -x "$GOV_BIN" ]; then
            echo "   Making governor executable..."
            sudo chmod +x "$GOV_BIN"
        fi
        echo "   Starting governor (CPU=$CPU_IDX, GPU=$GPU_IDX)..."
        sudo "$GOV_BIN" "$CPU_IDX" "$GPU_IDX" &
        PID_GOV=$!
        sleep 2
        echo "   ✓ Governor running (PID=$PID_GOV)"
    fi
else
    echo "━━━ [1/5] DVFS GOVERNOR ━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "   Skipped (no --use_governor)"
fi

# ─────────────────────────────────────────────
# STEP 2: TEGRASTATS
# ─────────────────────────────────────────────
echo ""
echo "━━━ [2/5] TEGRASTATS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
RAW_TEGRA="$OUTDIR/tegrastats_measure.txt"

sudo -n true 2>/dev/null || {
    echo "❌ sudo requires password. Add to sudoers:"
    echo "   nvidia ALL=(ALL) NOPASSWD: /usr/bin/tegrastats"
    # Cleanup governor if started
    [ -n "$PID_GOV" ] && sudo kill "$PID_GOV" 2>/dev/null || true
    exit 1
}

sudo -n tegrastats --interval "$MEASURE_INTERVAL" --logfile "$RAW_TEGRA" &
PID_TEGRA=$!
sleep 1
echo "   ✓ tegrastats running (PID=$PID_TEGRA, interval=${MEASURE_INTERVAL}ms)"
echo "   ✓ Logging to: $RAW_TEGRA"

# ─────────────────────────────────────────────
# STEP 3: TRAINING
# ─────────────────────────────────────────────
echo ""
echo "━━━ [3/5] TRAINING ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
PY_OUT="$OUTDIR/python_output.txt"
EVENTS_LOG="$OUTDIR/events_log.csv"

echo "   Script : $SCRIPT"
echo "   Args   : --dataset $DATASET --log_every_n_steps $LOG_STEPS ${PY_ARGS[*]:-}"
echo "   Output : $PY_OUT"
echo "   Events : $EVENTS_LOG"
echo ""

set +e
"$PYTHON_BIN" "$BASE/$SCRIPT" \
    --dataset "$DATASET" \
    --log_every_n_steps "$LOG_STEPS" \
    --output_dir "$OUTDIR" \
    "${PY_ARGS[@]}" \
    2>&1 | tee "$PY_OUT"
PY_RC=${PIPESTATUS[0]}
set -e

if [ "$PY_RC" -ne 0 ]; then
    echo ""
    echo "⚠️  Training exited with code $PY_RC"
else
    echo ""
    echo "   ✓ Training complete"
fi

# ─────────────────────────────────────────────
# STEP 4: STOP MONITORING
# ─────────────────────────────────────────────
echo ""
echo "━━━ [4/5] STOPPING MONITORS ━━━━━━━━━━━━━━━━━━━━━━━━"

echo "   Stopping tegrastats..."
sudo kill "$PID_TEGRA" 2>/dev/null || true
wait "$PID_TEGRA" 2>/dev/null || true
echo "   ✓ tegrastats stopped"

if [ -n "$PID_GOV" ]; then
    echo "   Stopping governor..."
    sudo kill "$PID_GOV" 2>/dev/null || true
    wait "$PID_GOV" 2>/dev/null || true
    echo "   ✓ Governor stopped"
fi

# ─────────────────────────────────────────────
# STEP 5: PARSE + OVERLAY + UPLOAD
# ─────────────────────────────────────────────
echo ""
echo "━━━ [5/5] POST-PROCESSING ━━━━━━━━━━━━━━━━━━━━━━━━━━"

PARSED_CSV="$OUTDIR/tegrastats_parsed.csv"

# 5a. Parse tegrastats with event labeling + upload to Google Sheets
echo ""
echo "   [5a] Parsing tegrastats with event labels..."

if [ -f "$EVENTS_LOG" ]; then
    # Use new labeled parser
    GSHEET_FLAG=""
    if [ "$UPLOAD_GSHEET" = true ]; then
        GSHEET_FLAG="--gsheet --experiment $EXP_NAME"
    fi
    
    "$PYTHON_BIN" "$BASE/parse_tegrastats_labeled.py" \
        "$RAW_TEGRA" \
        "$PARSED_CSV" \
        "$EVENTS_LOG" \
        "$PY_OUT" \
        $GSHEET_FLAG
    echo "   ✓ Tegrastats parsed with event labels -> $PARSED_CSV"
else
    # Fallback to old parser if no events_log
    echo "   ⚠️  events_log.csv not found - using basic parser"
    GSHEET_FLAG=""
    if [ "$UPLOAD_GSHEET" = true ]; then
        GSHEET_FLAG="--gsheet --experiment $EXP_NAME"
    fi
    
    "$PYTHON_BIN" "$BASE/parse_tegrastats.py" \
        "$RAW_TEGRA" \
        "$PARSED_CSV" \
        "$PY_OUT" \
        $GSHEET_FLAG
    echo "   ✓ Tegrastats parsed -> $PARSED_CSV"
fi

# ─────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║           EXPERIMENT PIPELINE DONE              ║"
echo "╠══════════════════════════════════════════════════╣"
printf  "║  Experiment : %-35s║\n" "$EXP_NAME"
printf  "║  Status     : %-35s║\n" "$([ "$PY_RC" -eq 0 ] && echo "✓ Success" || echo "⚠ Training failed (rc=$PY_RC)")"
echo "╠══════════════════════════════════════════════════╣"
echo "║  Output files:                                   ║"
printf  "║    tegrastats (raw) : %-29s║\n" "$(basename $RAW_TEGRA)"
printf  "║    tegrastats (csv) : %-29s║\n" "$(basename $PARSED_CSV)"
printf  "║    events log       : %-29s║\n" "$(basename $EVENTS_LOG)"
printf  "║    python output    : %-29s║\n" "$(basename $PY_OUT)"
echo "╠══════════════════════════════════════════════════╣"
printf  "║  Dir: %-44s║\n" "$OUTDIR"
echo "╚══════════════════════════════════════════════════╝"
echo ""

exit "$PY_RC"