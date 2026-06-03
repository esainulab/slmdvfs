#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# run_experiment_l2.sh - Level 2 Experiment Pipeline (Tegrastats + Phase Labeling)
#
# Same style/logic as your run_experiment.sh :contentReference[oaicite:2]{index=2}:
#   [1/5] DVFS GOVERNOR (optional)
#   [2/5] TEGRASTATS
#   [3/5] TRAINING (events_log.csv)
#   [4/5] STOP MONITORS
#   [5/5] POST-PROCESSING (labeled parser if events_log exists; else basic)
#
# Differences vs run_experiment.sh:
# - Prefers *_detailed_timing.py scripts if present; otherwise uses *_events_log.py
# - Stops early (no parsing) if training script is missing or training fails.
#
# Usage:
#   ./run_experiment_l2.sh <method> [options] [-- <extra python args>]
# =============================================================================

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE="/home/nvidia/Desktop/llm-dvfs"
PYTHON_BIN="/home/nvidia/llm-dvfs-env/bin/python3"

MEASURE_INTERVAL=10          # tegrastats interval in ms
LOG_STEPS=50                  # event log every N training steps
EPOCHS=1

# Parsers
PARSER_LABELED="$BASE/parse_tegrastats_labeled.py"
PARSER_BASIC="$BASE/parse_tegrastats.py"

# Prefer detailed timing scripts if they exist; otherwise fallback to events_log scripts you already have :contentReference[oaicite:3]{index=3}
declare -A METHOD_SCRIPT_PREFERRED
METHOD_SCRIPT_PREFERRED["fullft"]="BERT_FullFT_detailed_timing.py"
METHOD_SCRIPT_PREFERRED["bitfit"]="BERT_BitFit_detailed_timing.py"
METHOD_SCRIPT_PREFERRED["lora"]="BERT_LoRA_detailed_timing.py"

declare -A METHOD_SCRIPT_FALLBACK
METHOD_SCRIPT_FALLBACK["fullft"]="BERT_FullFT_events_log.py"
METHOD_SCRIPT_FALLBACK["bitfit"]="BERT_BitFit_events_log.py"
METHOD_SCRIPT_FALLBACK["lora"]="BERT_lora_events_log.py"

# ─────────────────────────────────────────────
# ARGUMENT PARSING (same pattern as run_experiment.sh :contentReference[oaicite:4]{index=4})
# ─────────────────────────────────────────────
if [ $# -lt 1 ]; then
    echo ""
    echo "Usage: $0 <method> [options] [-- <extra python args>]"
    echo ""
    echo "Methods:  fullft | bitfit | lora"
    echo "Datasets: sst2 (default) | qnli | mrpc"
    echo ""
    echo "Options:"
    echo "  --dataset <name>               Dataset to use"
    echo "  --use_governor <CPU> <GPU>     Enable DVFS (e.g. --use_governor 15 7)"
    echo "  --no_gsheet                    Skip Google Sheets upload"
    echo "  --log_steps <N>                Log event every N steps (default: $LOG_STEPS)"
    echo "  --epochs <N>                   Training epochs (default: $EPOCHS)"
    echo "  --batch_size <N>               Batch size (forwarded to python)"
    echo "  --interval_ms <N>              tegrastats interval (default: $MEASURE_INTERVAL)"
    echo ""
    exit 1
fi

METHOD_KEY="${1,,}"
shift

if [[ "$METHOD_KEY" != "fullft" && "$METHOD_KEY" != "bitfit" && "$METHOD_KEY" != "lora" ]]; then
    echo "❌ Unknown method: '$METHOD_KEY'. Choose from: fullft, bitfit, lora"
    exit 1
fi

DATASET="sst2"
USE_GOV=false
CPU_IDX=0
GPU_IDX=0
UPLOAD_GSHEET=true
BATCH_SIZE=""
PY_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)        DATASET="${2,,}"; shift 2 ;;
        --use_governor)   USE_GOV=true; CPU_IDX="${2:-0}"; GPU_IDX="${3:-0}"; shift 3 ;;
        --no_gsheet)      UPLOAD_GSHEET=false; shift ;;
        --log_steps)      LOG_STEPS="$2"; shift 2 ;;
        --epochs)         EPOCHS="$2"; shift 2 ;;
        --batch_size)     BATCH_SIZE="$2"; shift 2 ;;
        --interval_ms)    MEASURE_INTERVAL="$2"; shift 2 ;;
        --)               shift; PY_ARGS+=("$@"); break ;;
        *)                PY_ARGS+=("$1"); shift ;;
    esac
done

if [[ ! "$DATASET" =~ ^(sst2|qnli|mrpc)$ ]]; then
    echo "❌ Unknown dataset: '$DATASET'. Choose from: sst2, qnli, mrpc"
    exit 1
fi

# Choose script: preferred detailed timing if exists, else fallback events_log
SCRIPT_PREF="${METHOD_SCRIPT_PREFERRED[$METHOD_KEY]}"
SCRIPT_FALL="${METHOD_SCRIPT_FALLBACK[$METHOD_KEY]}"

if [ -f "$BASE/$SCRIPT_PREF" ]; then
    SCRIPT="$SCRIPT_PREF"
    SCRIPT_MODE="detailed_timing"
else
    SCRIPT="$SCRIPT_FALL"
    SCRIPT_MODE="events_log"
fi

# ─────────────────────────────────────────────
# OUTPUT DIR
# ─────────────────────────────────────────────
DATE=$(date +"%Y%m%d_%H%M%S")

EXP_NAME="${METHOD_KEY}_${DATASET}_L2"
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
echo "║        LEVEL-2 EXPERIMENT PIPELINE START        ║"
echo "╠══════════════════════════════════════════════════╣"
printf  "║  Experiment : %-35s║\n" "$EXP_NAME"
printf  "║  Method     : %-35s║\n" "$METHOD_KEY ($SCRIPT)"
printf  "║  ScriptMode : %-35s║\n" "$SCRIPT_MODE"
printf  "║  Dataset    : %-35s║\n" "$DATASET"
printf  "║  Epochs     : %-35s║\n" "$EPOCHS"
printf  "║  Log Steps  : %-35s║\n" "$LOG_STEPS"
printf  "║  Interval   : %-35s║\n" "${MEASURE_INTERVAL}ms"
printf  "║  DVFS       : %-35s║\n" "$([ "$USE_GOV" = true ] && echo "CPU=$CPU_IDX GPU=$GPU_IDX" || echo "disabled")"
printf  "║  GSheets    : %-35s║\n" "$([ "$UPLOAD_GSHEET" = true ] && echo "yes" || echo "no")"
printf  "║  Output     : %-35s║\n" "$OUTDIR"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ─────────────────────────────────────────────
# STEP 1: DVFS GOVERNOR (OPTIONAL) :contentReference[oaicite:5]{index=5}
# ─────────────────────────────────────────────
PID_GOV=""
echo "━━━ [1/5] DVFS GOVERNOR ━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ "$USE_GOV" = true ]; then
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
    echo "   Skipped (no --use_governor)"
fi

# ─────────────────────────────────────────────
# STEP 2: TEGRASTATS
# ─────────────────────────────────────────────
echo ""
echo "━━━ [2/5] TEGRASTATS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
RAW_TEGRA="$OUTDIR/tegrastats_detailed.txt"

sudo -n true 2>/dev/null || {
    echo "❌ sudo requires password. Add to sudoers:"
    echo "   nvidia ALL=(ALL) NOPASSWD: /usr/bin/tegrastats"
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

# HARD CHECK: training script must exist
if [ ! -f "$BASE/$SCRIPT" ]; then
    echo "❌ Training script not found: $BASE/$SCRIPT"
    echo "   (Looked for preferred: $SCRIPT_PREF; fallback: $SCRIPT_FALL)"
    echo "   Stopping monitors and exiting..."
    sudo kill "$PID_TEGRA" 2>/dev/null || true
    wait "$PID_TEGRA" 2>/dev/null || true
    [ -n "$PID_GOV" ] && sudo kill "$PID_GOV" 2>/dev/null || true
    exit 2
fi

echo "   Script : $SCRIPT"
echo "   Output : $PY_OUT"
echo "   Events : $EVENTS_LOG"
echo ""

CORE_ARGS=( "--dataset" "$DATASET" "--log_every_n_steps" "$LOG_STEPS" "--output_dir" "$OUTDIR" )

# Only add --epochs if your script supports it (your old scripts may not).
# But we’ll pass it anyway; if your script rejects it, you can remove this line.
CORE_ARGS+=( "--epochs" "$EPOCHS" )

if [[ -n "${BATCH_SIZE:-}" ]]; then
    CORE_ARGS+=( "--batch_size" "$BATCH_SIZE" )
fi

echo "   Args   : ${CORE_ARGS[*]} ${PY_ARGS[*]:-}"
echo ""

set +e
"$PYTHON_BIN" "$BASE/$SCRIPT" \
    "${CORE_ARGS[@]}" \
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

# If training failed, stop here (keep raw logs for debugging)
if [ "$PY_RC" -ne 0 ]; then
    echo ""
    echo "━━━ [5/5] POST-PROCESSING ━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "   ⚠️ Skipping parsing because training failed (rc=$PY_RC)."
    echo "   Raw tegrastats saved at: $RAW_TEGRA"
    echo "   Python output saved at : $PY_OUT"
    exit "$PY_RC"
fi

# ─────────────────────────────────────────────
# STEP 5: PARSE + LABEL + UPLOAD
# ─────────────────────────────────────────────
echo ""
echo "━━━ [5/5] POST-PROCESSING ━━━━━━━━━━━━━━━━━━━━━━━━━━"
PARSED_CSV="$OUTDIR/tegrastats_parsed.csv"

echo ""
echo "   [5a] Parsing tegrastats with event labels..."

GSHEET_FLAG=""
if [ "$UPLOAD_GSHEET" = true ]; then
    GSHEET_FLAG="--gsheet --experiment $EXP_NAME"
fi

# Same logic as your original script :contentReference[oaicite:6]{index=6}
if [ -f "$EVENTS_LOG" ] && [ -f "$PARSER_LABELED" ]; then
    "$PYTHON_BIN" "$PARSER_LABELED" \
        "$RAW_TEGRA" \
        "$PARSED_CSV" \
        "$EVENTS_LOG" \
        "$PY_OUT" \
        $GSHEET_FLAG
    echo "   ✓ Tegrastats parsed with event labels -> $PARSED_CSV"
else
    echo "   ⚠️  events_log.csv not found (or labeled parser missing) - using basic parser"
    "$PYTHON_BIN" "$PARSER_BASIC" \
        "$RAW_TEGRA" \
        "$PARSED_CSV" \
        "$PY_OUT" \
        $GSHEET_FLAG
    echo "   ✓ Tegrastats parsed -> $PARSED_CSV"
fi

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║        LEVEL-2 EXPERIMENT PIPELINE DONE         ║"
echo "╠══════════════════════════════════════════════════╣"
printf  "║  Experiment : %-35s║\n" "$EXP_NAME"
printf  "║  Status     : %-35s║\n" "✓ Success"
echo "╠══════════════════════════════════════════════════╣"
echo "║  Output files:                                   ║"
printf  "║    tegrastats (raw) : %-29s║\n" "$(basename "$RAW_TEGRA")"
printf  "║    tegrastats (csv) : %-29s║\n" "$(basename "$PARSED_CSV")"
printf  "║    events log       : %-29s║\n" "$(basename "$EVENTS_LOG")"
printf  "║    python output    : %-29s║\n" "$(basename "$PY_OUT")"
echo "╠══════════════════════════════════════════════════╣"
printf  "║  Dir: %-44s║\n" "$OUTDIR"
echo "╚══════════════════════════════════════════════════╝"
echo ""

exit 0