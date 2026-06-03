#!/bin/bash
# NVP Mode 2 (MODE_30W) Baseline Experiments — BASELINE (uncapped GPU)
# Models: BERT-tiny, BERT-base, DeBERTa-xlarge
# Datasets: SST-2, QNLI
# Total runs: 6
#
# Usage: ./run_mode2_baselines.sh

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

# [model_name, batch_size, max_length, model_label, dataset, extra_args]
EXPERIMENTS=(
    "prajjwal1/bert-tiny:512:128:BERT-tiny:sst2:"
    "prajjwal1/bert-tiny:256:256:BERT-tiny:qnli:"
    "bert-base-uncased:128:128:BERT-base:sst2:"
    "bert-base-uncased:128:256:BERT-base:qnli:"
    "microsoft/deberta-v2-xlarge:16:128:DeBERTa-xlarge:sst2:--no-fp16 --lr 1e-5"
    "microsoft/deberta-v2-xlarge:16:256:DeBERTa-xlarge:qnli:--no-fp16 --lr 1e-5"
)

EPOCHS=1
LOG_STEPS=500
COOLDOWN=60         # seconds between runs
TEMP_THRESHOLD=85   # °C
TEGRASTATS_INTERVAL=10  # milliseconds

GPU_DEVFREQ_PATH="/sys/class/devfreq/17000000.gpu"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OUTPUT_BASE="${SCRIPT_DIR}/runs/mode2_baselines"

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
    # Use -n (non-interactive); requires NOPASSWD or pre-cached credentials
    if sudo -n true 2>/dev/null; then
        echo "sudo credentials OK"
    else
        echo "Requesting sudo credentials..."
        sudo true || { echo "sudo authentication failed"; exit 1; }
    fi
    (while true; do sudo -n true; sleep 60; kill -0 "$$" 2>/dev/null || exit; done) &
    SUDO_KEEPALIVE_PID=$!
    trap 'kill $SUDO_KEEPALIVE_PID 2>/dev/null; restore_gpu_freq' EXIT
    echo "sudo credentials cached"
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

restore_gpu_freq() {
    local max_hz=$(cat ${GPU_DEVFREQ_PATH}/available_frequencies | tr ' ' '\n' | sort -n | tail -1)
    sudo bash -c "echo $max_hz > ${GPU_DEVFREQ_PATH}/max_freq" 2>/dev/null || true
    sudo bash -c "echo $max_hz > ${GPU_DEVFREQ_PATH}/min_freq" 2>/dev/null || true
    sudo bash -c "echo nvhost_podgov > ${GPU_DEVFREQ_PATH}/governor" 2>/dev/null || true
    echo "GPU governor restored"
}

set_baseline_gpu() {
    sudo bash -c "echo nvhost_podgov > ${GPU_DEVFREQ_PATH}/governor"
    sleep 2
    local cur_mhz=$(( $(cat ${GPU_DEVFREQ_PATH}/cur_freq) / 1000000 ))
    echo "GPU uncapped (nvhost_podgov), current: ${cur_mhz} MHz"
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
        echo "Temperature: ${current_temp}C — waiting for cooldown..."
        while [ $current_temp -gt 70 ]; do
            sleep 10
            current_temp=$(check_temperature)
            echo "   Temperature: ${current_temp}C"
        done
        echo "Cooled to ${current_temp}C"
    fi
}

# ============================================================================
# SINGLE RUN
# ============================================================================

run_single() {
    local model_name=$1
    local batch_size=$2
    local max_length=$3
    local model_label=$4
    local dataset=$5
    local extra_args=${6:-}
    local run_num=$7
    local total_runs=$8
    local start_time_global=$9

    local dataset_upper=$(echo "$dataset" | tr '[:lower:]' '[:upper:]')
    print_header "RUN $run_num/$total_runs: ${model_label} ${dataset_upper} BASELINE (Mode 2)"

    local run_start=$(date +%s)
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local experiment_name="${model_label}_${dataset_upper}_BASELINE_mode2"
    local output_dir="${OUTPUT_BASE}/${experiment_name}_${timestamp}"

    mkdir -p "$output_dir"

    # Temperature check
    local temp=$(check_temperature)
    echo "Pre-run temperature: ${temp}C"
    if [ $temp -gt $TEMP_THRESHOLD ]; then
        echo "Temperature too high! Aborting."
        exit 1
    fi
    wait_for_cooldown

    # Set GPU to uncapped baseline
    print_section "GPU Setup"
    set_baseline_gpu

    # Start tegrastats
    print_section "Starting Monitoring"
    sudo tegrastats --interval $TEGRASTATS_INTERVAL --logfile "${output_dir}/tegrastats_measure.txt" &
    local TEGRA_PID=$!
    echo "tegrastats PID: $TEGRA_PID"
    sleep 2

    # Run training
    print_section "Training (${model_label}, ${dataset_upper}, N=${max_length}, batch=${batch_size})"
    set +e
    "$PYTHON_BIN" "$TRAINING_SCRIPT" \
        --model_name "$model_name" \
        --dataset "$dataset" \
        --max_length $max_length \
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
    echo "tegrastats stopped"

    if [ $train_exit -ne 0 ]; then
        echo "Training failed (exit code $train_exit)"
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
            2>&1 || echo "Parsing failed (non-fatal)"
    fi

    # Summary
    local run_end=$(date +%s)
    local run_duration=$((run_end - run_start))
    temp=$(check_temperature)

    echo ""
    echo "Run done in $((run_duration / 60))m $((run_duration % 60))s"
    echo "  Post-run temperature: ${temp}C"
    echo "  Results: $output_dir"

    local elapsed=$((run_end - start_time_global))
    local avg=$((elapsed / run_num))
    local remaining=$((total_runs - run_num))
    local eta=$((avg * remaining))
    echo "  Progress: $run_num/$total_runs | ETA: $((eta / 3600))h $((eta % 3600 / 60))m"
}

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

preflight_checks() {
    print_header "PRE-FLIGHT CHECKS"

    print_section "Python Environment"
    echo "Python bin: $PYTHON_BIN"
    "$PYTHON_BIN" --version 2>&1
    if ! "$PYTHON_BIN" -c "import torch, transformers, datasets, evaluate, numpy, tqdm" 2>/dev/null; then
        echo "Missing required Python packages"; exit 1
    fi
    echo "Python dependencies OK"

    print_section "NVP Mode"
    nvpmodel -q
    local mode=$(nvpmodel -q 2>/dev/null | head -1)
    if [[ "$mode" != *"MODE_30W"* ]]; then
        echo "WARNING: Expected MODE_30W (mode 2), got: $mode"
        read -p "Continue anyway? [y/N] " -n 1 -r; echo
        [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
    fi

    print_section "Disk Space"
    df -h "$SCRIPT_DIR" | tail -1

    print_section "System Temperature"
    local temp=$(check_temperature)
    echo "Current: ${temp}C"
    [ $temp -gt $TEMP_THRESHOLD ] && { echo "Temperature too high."; exit 1; }

    print_section "Experiment Plan"
    echo "NVP Mode: MODE_30W (mode 2)"
    echo "GPU: BASELINE (uncapped, nvhost_podgov)"
    echo ""
    printf "  %-3s  %-25s  %-10s  %-8s  %-6s\n" "#" "Model" "Dataset" "Batch" "N"
    printf "  %-3s  %-25s  %-10s  %-8s  %-6s\n" "---" "-------------------------" "----------" "--------" "------"
    local i=1
    for exp in "${EXPERIMENTS[@]}"; do
        IFS=':' read -r mn bs ml lb ds ea <<< "$exp"
        printf "  %-3s  %-25s  %-10s  %-8s  %-6s\n" "$i" "$lb" "$ds" "$bs" "$ml"
        i=$((i+1))
    done
    echo ""

    echo "All checks passed. Starting experiment..."
}

# ============================================================================
# MAIN
# ============================================================================

main() {
    cd "$SCRIPT_DIR"
    setup_sudo
    preflight_checks

    print_header "MODE 2 BASELINE EXPERIMENTS"

    local total_runs=${#EXPERIMENTS[@]}
    local current_run=0
    local start_time=$(date +%s)

    mkdir -p "$OUTPUT_BASE"
    cat > "${OUTPUT_BASE}/experiment_metadata.txt" << EOF
NVP Mode 2 (MODE_30W) Baseline Experiments
===========================================
Date: $(date)
NVP Mode: MODE_30W (2)
GPU: BASELINE (uncapped)
Epochs: $EPOCHS

Runs:
$(for exp in "${EXPERIMENTS[@]}"; do IFS=':' read -r mn bs ml lb ds ea <<< "$exp"; echo "  $lb $ds batch=$bs N=$ml"; done)

System Info:
$(uname -a)
Python: $("$PYTHON_BIN" --version 2>&1)
EOF

    for exp in "${EXPERIMENTS[@]}"; do
        IFS=':' read -r model_name batch_size max_length model_label dataset extra_args <<< "$exp"
        current_run=$((current_run + 1))
        run_single "$model_name" "$batch_size" "$max_length" "$model_label" \
            "$dataset" "$extra_args" "$current_run" "$total_runs" "$start_time"
        if [ $current_run -lt $total_runs ]; then
            print_section "Cooldown ($COOLDOWN seconds)"
            sleep $COOLDOWN
        fi
    done

    local end_time=$(date +%s)
    local total_duration=$((end_time - start_time))

    print_header "ALL MODE 2 BASELINES COMPLETE!"
    echo "Total time: $((total_duration / 3600))h $((total_duration % 3600 / 60))m"
    echo "Results: $OUTPUT_BASE"
}

main "$@"
