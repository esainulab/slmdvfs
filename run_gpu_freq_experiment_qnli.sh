#!/bin/bash
# GPU Frequency Capping Experiment - QNLI, N=256
# Tests BERT-base and DeBERTa-xlarge at 7 frequencies
#
# Usage: ./run_gpu_freq_experiment_qnli.sh
#
# Estimated time: ~24h total (BERT: ~9h, DeBERTa: ~14h)

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

# BERT: baseline (uncapped) only — 714/612 done, 1300/1122/1020/918 done
BERT_FREQS=(BASELINE)
BERT_MODEL="bert-base-uncased:128:BERT-base:"

# DeBERTa: baseline (uncapped) + remaining capped frequencies
DEBERTA_FREQS=(BASELINE 816 714 612)
DEBERTA_MODEL="microsoft/deberta-v2-xlarge:16:DeBERTa-xlarge:--no-fp16 --lr 1e-5"

DATASET="qnli"
MAX_LENGTH=256
EPOCHS=1
LOG_STEPS=500
COOLDOWN=30         # seconds between runs (longer for DeBERTa heat)
TEMP_THRESHOLD=85   # °C
TEGRASTATS_INTERVAL=10  # milliseconds

# Paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OUTPUT_BASE="${SCRIPT_DIR}/runs/qnli_gpu_freq"
GPU_DEVFREQ_PATH="/sys/class/devfreq/17000000.gpu"

DEFAULT_PYTHON_BIN="/home/nvidia/llm-dvfs-env/bin/python"
PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
    if [ -x "$DEFAULT_PYTHON_BIN" ]; then
        PYTHON_BIN="$DEFAULT_PYTHON_BIN"
    else
        PYTHON_BIN="$(command -v python3)"
    fi
fi

TRAINING_SCRIPT="${SCRIPT_DIR}/BERT_sst2_FullFT.py"
PARSER_SCRIPT="${SCRIPT_DIR}/parse_tegrastats_labeled.py"

# ============================================================================
# SUDO CREDENTIAL CACHING
# ============================================================================

setup_sudo() {
    echo "🔐 Requesting sudo credentials (will be cached for the entire run)..."
    sudo -v || { echo "❌ sudo authentication failed"; exit 1; }

    (while true; do sudo -n true; sleep 60; kill -0 "$$" 2>/dev/null || exit; done) &
    SUDO_KEEPALIVE_PID=$!

    trap 'kill $SUDO_KEEPALIVE_PID 2>/dev/null; restore_gpu_freq' EXIT
    echo "✓ sudo credentials cached"
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

print_header() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    printf "║  %-58s  ║\n" "$1"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
}

print_section() {
    echo ""
    echo "━━━ $1 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

setup_gpu_governor() {
    local current_gov=$(cat ${GPU_DEVFREQ_PATH}/governor)
    if [ "$current_gov" != "userspace" ]; then
        echo "Setting governor to 'userspace'..."
        sudo bash -c "echo userspace > ${GPU_DEVFREQ_PATH}/governor"
        echo "✓ Governor set to userspace"
    else
        echo "✓ Governor already userspace"
    fi
}

set_gpu_freq() {
    local freq_mhz=$1
    local freq_hz=$((freq_mhz * 1000000))
    echo "Setting GPU frequency to ${freq_mhz} MHz..."
    sudo bash -c "echo $freq_hz > ${GPU_DEVFREQ_PATH}/max_freq"
    sudo bash -c "echo $freq_hz > ${GPU_DEVFREQ_PATH}/min_freq"
    sleep 2
}

restore_gpu_freq() {
    # Reset to max on exit so system isn't left capped
    local max_hz=$(cat ${GPU_DEVFREQ_PATH}/available_frequencies | tr ' ' '\n' | sort -n | tail -1)
    sudo bash -c "echo $max_hz > ${GPU_DEVFREQ_PATH}/max_freq" 2>/dev/null || true
    sudo bash -c "echo $max_hz > ${GPU_DEVFREQ_PATH}/min_freq" 2>/dev/null || true
    sudo bash -c "echo nvhost_podgov > ${GPU_DEVFREQ_PATH}/governor" 2>/dev/null || true
    echo "✓ GPU governor restored"
}

verify_gpu_freq() {
    local actual_hz=$(cat ${GPU_DEVFREQ_PATH}/cur_freq)
    echo $((actual_hz / 1000000))
}

check_temperature() {
    local max_temp=0
    for zone in /sys/devices/virtual/thermal/thermal_zone*/temp; do
        if [ -f "$zone" ]; then
            local temp
            temp="$(cat "$zone" 2>/dev/null || true)"
            [ -z "$temp" ] && continue
            temp=$((temp / 1000))
            [ $temp -gt $max_temp ] && max_temp=$temp
        fi
    done
    echo $max_temp
}

wait_for_cooldown() {
    local current_temp=$(check_temperature)
    if [ $current_temp -gt 75 ]; then
        echo "⚠️  Temperature: ${current_temp}°C — waiting for cooldown..."
        while [ $current_temp -gt 70 ]; do
            sleep 10
            current_temp=$(check_temperature)
            echo "   Temperature: ${current_temp}°C"
        done
        echo "✓ Cooled to ${current_temp}°C"
    fi
}

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

preflight_checks() {
    print_header "PRE-FLIGHT CHECKS"

    print_section "Python Environment"
    if [ -z "$PYTHON_BIN" ] || [ ! -x "$PYTHON_BIN" ]; then
        echo "❌ Python interpreter not found: $PYTHON_BIN"
        exit 1
    fi
    echo "Python bin: $PYTHON_BIN"
    "$PYTHON_BIN" --version 2>&1

    if ! "$PYTHON_BIN" -c "import torch, transformers, datasets, evaluate, numpy, tqdm" 2>/dev/null; then
        echo "❌ Missing required Python packages"
        exit 1
    fi
    echo "✓ Python dependencies OK"

    print_section "Training Script"
    if [ ! -f "$TRAINING_SCRIPT" ]; then
        echo "❌ Training script not found: $TRAINING_SCRIPT"
        exit 1
    fi
    echo "✓ $TRAINING_SCRIPT"

    if [ ! -f "$PARSER_SCRIPT" ]; then
        echo "⚠️  Parser not found: $PARSER_SCRIPT (will skip parsing)"
    else
        echo "✓ $PARSER_SCRIPT"
    fi

    print_section "Power Mode"
    sudo nvpmodel -q 2>/dev/null | grep "NV Power Mode" || echo "Mode info not available"

    print_section "Available GPU Frequencies"
    if [ ! -f "${GPU_DEVFREQ_PATH}/available_frequencies" ]; then
        echo "❌ GPU devfreq path not found: $GPU_DEVFREQ_PATH"
        exit 1
    fi
    echo "Available (Hz): $(cat ${GPU_DEVFREQ_PATH}/available_frequencies)"
    echo "To test (MHz): ${FREQS[*]}"

    print_section "Disk Space"
    df -h "$SCRIPT_DIR" | tail -1

    print_section "System Temperature"
    local temp=$(check_temperature)
    echo "Current: ${temp}°C"
    if [ $temp -gt $TEMP_THRESHOLD ]; then
        echo "❌ Temperature too high. Wait for cooldown."
        exit 1
    fi

    print_section "GPU Frequency Control Test"
    setup_gpu_governor
    set_gpu_freq 816
    local verified=$(verify_gpu_freq)
    echo "✓ Set 816 MHz → verified: ${verified} MHz"

    print_section "Experiment Summary"
    echo "Dataset:      $DATASET"
    echo "Max length:   $MAX_LENGTH tokens"
    echo "Epochs:       $EPOCHS"
    echo ""
    echo "Phase 1 — BERT-base:      freqs=${BERT_FREQS[*]}"
    echo "Phase 2 — DeBERTa-xlarge: freqs=${DEBERTA_FREQS[*]}"
    echo ""
    echo "Total runs:   $(( ${#BERT_FREQS[@]} + ${#DEBERTA_FREQS[@]} ))"
    echo "Est. time:    ~2h (BERT) + ~49h (DeBERTa fp32)"

    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    read -p "All checks passed. Start experiment? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
}

# ============================================================================
# SINGLE RUN
# ============================================================================

run_single() {
    local freq=$1
    local model_name=$2
    local batch_size=$3
    local model_label=$4
    local run_num=$5
    local total_runs=$6
    local start_time_global=$7
    local extra_args=${8:-}

    print_header "RUN $run_num/$total_runs: ${model_label} @ ${freq} MHz"

    local run_start=$(date +%s)
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local experiment_name="${model_label//[^a-zA-Z0-9]/_}_QNLI_${freq}"
    local output_dir="${OUTPUT_BASE}/${experiment_name}_${timestamp}"

    mkdir -p "$output_dir"

    # Temperature check
    local temp=$(check_temperature)
    echo "Pre-run temperature: ${temp}°C"
    if [ $temp -gt $TEMP_THRESHOLD ]; then
        echo "❌ Temperature too high! Aborting."
        exit 1
    fi
    wait_for_cooldown

    # Set frequency (or restore governor for baseline)
    if [ "$freq" = "BASELINE" ]; then
        echo "Restoring GPU governor to nvhost_podgov (uncapped baseline)..."
        sudo bash -c "echo nvhost_podgov > ${GPU_DEVFREQ_PATH}/governor"
        sleep 2
        local verified_freq=$(verify_gpu_freq)
        echo "✓ GPU uncapped, current freq: ${verified_freq} MHz"
    else
        set_gpu_freq $freq
        local verified_freq=$(verify_gpu_freq)
        echo "✓ GPU frequency: ${verified_freq} MHz"
    fi

    # Start tegrastats
    print_section "Starting Monitoring"
    sudo tegrastats --interval $TEGRASTATS_INTERVAL --logfile "${output_dir}/tegrastats_measure.txt" &
    local TEGRA_PID=$!
    echo "✓ tegrastats PID: $TEGRA_PID"
    sleep 2

    # Run training
    print_section "Training (${model_label}, QNLI, N=${MAX_LENGTH}, batch=${batch_size}${extra_args:+, $extra_args})"
    set +e
    "$PYTHON_BIN" "$TRAINING_SCRIPT" \
        --model_name "$model_name" \
        --dataset "$DATASET" \
        --max_length $MAX_LENGTH \
        --epochs $EPOCHS \
        --batch_size $batch_size \
        --log_every_n_steps $LOG_STEPS \
        --output_dir "$output_dir" \
        --enable_tqdm \
        $extra_args \
        2>&1 | tee "${output_dir}/python_output.txt"
    local train_exit=${PIPESTATUS[0]}
    set -e

    # Stop tegrastats
    sudo kill $TEGRA_PID 2>/dev/null || true
    echo "✓ tegrastats stopped"

    if [ $train_exit -ne 0 ]; then
        echo "❌ Training failed (exit code $train_exit)"
        exit 1
    fi

    # Parse results
    if [ -f "$PARSER_SCRIPT" ]; then
        print_section "Parsing Results"
        "$PYTHON_BIN" "$PARSER_SCRIPT" \
            "${output_dir}/tegrastats_measure.txt" \
            "${output_dir}/tegrastats_parsed.csv" \
            "${output_dir}/events_log.csv" \
            "${output_dir}/python_output.txt" \
            --experiment "$experiment_name" \
            --gsheet \
            2>&1 || echo "⚠️  Parsing failed (non-fatal)"
    fi

    # Summary
    local run_end=$(date +%s)
    local run_duration=$((run_end - run_start))
    temp=$(check_temperature)

    echo ""
    echo "✓ Run done in $((run_duration / 60))m $((run_duration % 60))s"
    echo "  Post-run temperature: ${temp}°C"
    echo "  Results: $output_dir"

    local elapsed=$((run_end - start_time_global))
    local avg=$((elapsed / run_num))
    local remaining=$((total_runs - run_num))
    local eta=$((avg * remaining))
    echo "  Progress: $run_num/$total_runs | ETA: $((eta / 3600))h $((eta % 3600 / 60))m"
}

# ============================================================================
# MAIN EXPERIMENT LOOP
# ============================================================================

run_experiments() {
    print_header "QNLI GPU FREQUENCY EXPERIMENT (N=256)"

    local total_runs=$(( ${#BERT_FREQS[@]} + ${#DEBERTA_FREQS[@]} ))
    local current_run=0
    local start_time=$(date +%s)

    setup_gpu_governor
    mkdir -p "$OUTPUT_BASE"

    # Save metadata
    cat > "${OUTPUT_BASE}/experiment_metadata.txt" << EOF
QNLI GPU Frequency Experiment
==============================
Date: $(date)
Dataset: $DATASET
Max sequence length: $MAX_LENGTH
Epochs: $EPOCHS

BERT-base frequencies (MHz): ${BERT_FREQS[*]}
DeBERTa-xlarge frequencies (MHz): ${DEBERTA_FREQS[*]}

System Info:
$(uname -a)
Python: $("$PYTHON_BIN" --version 2>&1)
Python bin: $PYTHON_BIN

Available GPU Frequencies:
$(cat ${GPU_DEVFREQ_PATH}/available_frequencies)
EOF

    # Phase 1: BERT-base (remaining frequencies only)
    print_header "PHASE 1: BERT-base (${#BERT_FREQS[@]} runs)"
    IFS=':' read -r model_name batch_size model_label extra_args <<< "$BERT_MODEL"
    for freq in "${BERT_FREQS[@]}"; do
        current_run=$((current_run + 1))
        run_single "$freq" "$model_name" "$batch_size" "$model_label" \
            "$current_run" "$total_runs" "$start_time" "$extra_args"
        if [ $current_run -lt $total_runs ]; then
            print_section "Cooldown ($COOLDOWN seconds)"
            sleep $COOLDOWN
        fi
    done

    # Phase 2: DeBERTa-xlarge (all 7 frequencies, fp32 + lr=1e-5)
    print_header "PHASE 2: DeBERTa-xlarge (${#DEBERTA_FREQS[@]} runs)"
    IFS=':' read -r model_name batch_size model_label extra_args <<< "$DEBERTA_MODEL"
    for freq in "${DEBERTA_FREQS[@]}"; do
        current_run=$((current_run + 1))
        run_single "$freq" "$model_name" "$batch_size" "$model_label" \
            "$current_run" "$total_runs" "$start_time" "$extra_args"
        if [ $current_run -lt $total_runs ]; then
            print_section "Cooldown ($COOLDOWN seconds)"
            sleep $COOLDOWN
        fi
    done

    local end_time=$(date +%s)
    local total_duration=$((end_time - start_time))

    print_header "EXPERIMENT COMPLETE!"
    echo "Total time: $((total_duration / 3600))h $((total_duration % 3600 / 60))m"
    echo "Results: $OUTPUT_BASE"
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                QNLI EXPERIMENT COMPLETE                      ║"
    echo "╠══════════════════════════════════════════════════════════════╣"
    printf "║  Frequencies tested: %-39s ║\n" "${#FREQS[@]}"
    printf "║  Models per freq:    %-39s ║\n" "${#MODELS[@]}"
    printf "║  Total runs:         %-39s ║\n" "$total_runs"
    printf "║  Results directory:  %-39s ║\n" "runs/qnli_gpu_freq/"
    printf "║  Check Google Sheets for compiled results                   ║\n"
    echo "╚══════════════════════════════════════════════════════════════╝"
}

# ============================================================================
# MAIN
# ============================================================================

main() {
    cd "$SCRIPT_DIR"
    setup_sudo
    preflight_checks
    run_experiments
}

main "$@"
