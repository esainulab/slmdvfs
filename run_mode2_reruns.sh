#!/bin/bash
# Mode 2 Re-runs for Fair Comparison
# 1. BERT-base SST-2 at 5 epochs (to match MAXN baseline)
# 2. DeBERTa SST-2 at batch=128 (to match MAXN batch size)
#
# Usage: ./run_mode2_reruns.sh

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

COOLDOWN=60
TEMP_THRESHOLD=85
TEGRASTATS_INTERVAL=10

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
# HELPERS
# ============================================================================

setup_sudo() {
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
    local epochs=$6
    local extra_args=${7:-}
    local experiment_tag=$8
    local run_num=$9
    local total_runs=${10}
    local start_time_global=${11}

    local dataset_upper=$(echo "$dataset" | tr '[:lower:]' '[:upper:]')
    print_header "RUN $run_num/$total_runs: $experiment_tag"

    local run_start=$(date +%s)
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local experiment_name="${experiment_tag}"
    local output_dir="${OUTPUT_BASE}/${experiment_name}_${timestamp}"

    mkdir -p "$output_dir"

    local temp=$(check_temperature)
    echo "Pre-run temperature: ${temp}C"
    [ $temp -gt $TEMP_THRESHOLD ] && { echo "Temperature too high! Aborting."; exit 1; }
    wait_for_cooldown

    print_section "GPU Setup"
    set_baseline_gpu

    print_section "Starting Monitoring"
    sudo tegrastats --interval $TEGRASTATS_INTERVAL --logfile "${output_dir}/tegrastats_measure.txt" &
    local TEGRA_PID=$!
    echo "tegrastats PID: $TEGRA_PID"
    sleep 2

    print_section "Training (${model_label}, ${dataset_upper}, N=${max_length}, batch=${batch_size}, epochs=${epochs})"
    set +e
    "$PYTHON_BIN" "$TRAINING_SCRIPT" \
        --model_name "$model_name" \
        --dataset "$dataset" \
        --max_length $max_length \
        --epochs $epochs \
        --batch_size $batch_size \
        --log_every_n_steps 500 \
        --output_dir "$output_dir" \
        --enable_tqdm \
        $extra_args \
        2>&1 | tee "${output_dir}/python_output.txt"
    local train_exit=${PIPESTATUS[0]}
    set -e

    sudo kill $TEGRA_PID 2>/dev/null || true
    echo "tegrastats stopped"

    [ $train_exit -ne 0 ] && { echo "Training failed (exit code $train_exit)"; exit 1; }

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

    local run_end=$(date +%s)
    local run_duration=$((run_end - run_start))
    temp=$(check_temperature)
    echo ""
    echo "Run done in $((run_duration / 60))m $((run_duration % 60))s"
    echo "  Post-run temperature: ${temp}C"
    echo "  Results: $output_dir"

    local elapsed=$((run_end - start_time_global))
    local avg=$((elapsed / run_num))
    local eta=$(( avg * (total_runs - run_num) ))
    echo "  Progress: $run_num/$total_runs | ETA: $((eta / 3600))h $((eta % 3600 / 60))m"
}

# ============================================================================
# MAIN
# ============================================================================

main() {
    cd "$SCRIPT_DIR"
    setup_sudo

    print_header "MODE 2 RE-RUNS FOR FAIR COMPARISON"
    echo "Run 1: BERT-base  SST-2  5 epochs  batch=128  N=128"
    echo "Run 2: DeBERTa    SST-2  1 epoch   batch=128  N=128"
    echo ""

    nvpmodel -q
    local temp=$(check_temperature)
    echo "Temperature: ${temp}C"
    [ $temp -gt $TEMP_THRESHOLD ] && { echo "Temperature too high."; exit 1; }

    mkdir -p "$OUTPUT_BASE"
    local start_time=$(date +%s)

    # Run 1: BERT-base SST-2, 5 epochs
    run_single \
        "bert-base-uncased" "128" "128" "BERT-base" "sst2" "5" "" \
        "BERT-base_SST2_BASELINE_mode2_5ep" "1" "2" "$start_time"

    print_section "Cooldown ($COOLDOWN seconds)"
    sleep $COOLDOWN

    # Run 2: DeBERTa SST-2, 1 epoch, batch=128
    run_single \
        "microsoft/deberta-v2-xlarge" "128" "128" "DeBERTa-xlarge" "sst2" "1" "--no-fp16 --lr 1e-5" \
        "DeBERTa-xlarge_SST2_BASELINE_mode2_b128" "2" "2" "$start_time"

    local end_time=$(date +%s)
    local total=$(( end_time - start_time ))
    print_header "RE-RUNS COMPLETE!"
    echo "Total time: $((total / 3600))h $((total % 3600 / 60))m"
    echo "Results: $OUTPUT_BASE"
}

main "$@"
