#!/bin/bash
# GPU Baseline Experiment - NO FREQUENCY CAPPING
# Let the GPU governor run naturally to establish true baseline performance

set -e  # Exit on error

# ============================================================================
# CONFIGURATION
# ============================================================================

# Experiment parameters (SAME as frequency capping experiments)
DATASET="sst2"
EPOCHS=1
BATCH_SIZE=128
MODEL_NAME="microsoft/deberta-v2-xlarge"
# For the "pure" training script, keep logging infrequent to avoid overhead/noise.
LOG_STEPS=500
TEGRASTATS_INTERVAL=10  # milliseconds
YES=0  # Set to 1 to skip interactive confirmation

# Paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OUTPUT_BASE="${SCRIPT_DIR}/runs/gpu_freq_capping"
GPU_DEVFREQ_PATH="/sys/class/devfreq/17000000.gpu"

# Python interpreter
# Prefer the known venv used by previous runs (see `runs/*/python_output.txt` paths).
DEFAULT_PYTHON_BIN="/home/nvidia/llm-dvfs-env/bin/python"
PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
    if [ -x "$DEFAULT_PYTHON_BIN" ]; then
        PYTHON_BIN="$DEFAULT_PYTHON_BIN"
    else
        PYTHON_BIN="$(command -v python3)"
    fi
fi

# Training script to use (no phase/event logging overhead)
TRAINING_SCRIPT="${SCRIPT_DIR}/BERT_sst2_FullFT.py"
PARSER_SCRIPT="${SCRIPT_DIR}/parse_tegrastats_labeled.py"

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

# Get current GPU governor
get_gpu_governor() {
    cat ${GPU_DEVFREQ_PATH}/governor
}

# Get GPU frequency limits
get_gpu_freq_limits() {
    local min_freq=$(cat ${GPU_DEVFREQ_PATH}/min_freq)
    local max_freq=$(cat ${GPU_DEVFREQ_PATH}/max_freq)
    echo "Min: $((min_freq / 1000000)) MHz, Max: $((max_freq / 1000000)) MHz"
}

# Get current GPU frequency
get_current_gpu_freq() {
    local freq_hz=$(cat ${GPU_DEVFREQ_PATH}/cur_freq)
    echo $((freq_hz / 1000000))
}

# Check system temperature
check_temperature() {
    local max_temp=0
    for zone in /sys/devices/virtual/thermal/thermal_zone*/temp; do
        if [ -f "$zone" ]; then
            local temp=$(cat "$zone")
            temp=$((temp / 1000))
            [ $temp -gt $max_temp ] && max_temp=$temp
        fi
    done
    echo $max_temp
}

# ============================================================================
# BASELINE SETUP
# ============================================================================

setup_baseline_governor() {
    print_section "Setting up BASELINE (No Capping)"
    
    local current_gov=$(get_gpu_governor)
    echo "Current governor: $current_gov"
    
    # Set to nvhost_podgov (default dynamic governor) or simple_ondemand
    # This allows GPU to scale naturally based on load
    
    local target_governor="nvhost_podgov"
    
    # Check if nvhost_podgov is available
    local available_govs=$(cat ${GPU_DEVFREQ_PATH}/available_governors)
    echo "Available governors: $available_govs"
    
    if [[ ! "$available_govs" =~ "nvhost_podgov" ]]; then
        # Fallback to simple_ondemand if nvhost_podgov not available
        if [[ "$available_govs" =~ "simple_ondemand" ]]; then
            target_governor="simple_ondemand"
        else
            echo "⚠️  Warning: Neither nvhost_podgov nor simple_ondemand available"
            echo "   Using first available governor"
            target_governor=$(echo $available_govs | awk '{print $1}')
        fi
    fi
    
    echo "Setting governor to: $target_governor"
    sudo bash -c "echo $target_governor > ${GPU_DEVFREQ_PATH}/governor"
    
    # Set to maximum available frequency limits (uncapped)
    local max_available=$(cat ${GPU_DEVFREQ_PATH}/available_frequencies | awk '{print $NF}')
    local min_available=$(cat ${GPU_DEVFREQ_PATH}/available_frequencies | awk '{print $1}')
    
    echo "Setting frequency range to MAXIMUM (uncapped):"
    echo "  Min: $((min_available / 1000000)) MHz"
    echo "  Max: $((max_available / 1000000)) MHz"
    
    sudo bash -c "echo $min_available > ${GPU_DEVFREQ_PATH}/min_freq"
    sudo bash -c "echo $max_available > ${GPU_DEVFREQ_PATH}/max_freq"
    
    sleep 2
    
    # Verify
    local new_gov=$(get_gpu_governor)
    local limits=$(get_gpu_freq_limits)
    
    echo ""
    echo "✓ Baseline configuration:"
    echo "  Governor: $new_gov"
    echo "  Frequency limits: $limits"
    echo "  GPU will scale dynamically based on workload"
}

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

preflight_checks() {
    print_header "BASELINE EXPERIMENT - PRE-FLIGHT CHECKS"

    # Check Python env + deps before touching GPU clocks / starting tegrastats
    print_section "Python Environment"
    if [ -z "$PYTHON_BIN" ] || [ ! -x "$PYTHON_BIN" ]; then
        echo "❌ Python interpreter not found/executable: $PYTHON_BIN"
        echo "   Tip: set PYTHON_BIN=/path/to/python"
        exit 1
    fi
    echo "Python bin: $PYTHON_BIN"
    "$PYTHON_BIN" --version 2>&1 || true

    # Verify required imports early (torch is the common failure mode)
    if ! "$PYTHON_BIN" -c "import torch, transformers, datasets, evaluate, numpy, tqdm" 2>/dev/null; then
        echo "❌ Missing required Python packages in: $PYTHON_BIN"
        echo "   Required: torch, transformers, datasets, evaluate, numpy, tqdm"
        echo ""
        echo "   If you have the project venv:"
        echo "     export PYTHON_BIN=/home/nvidia/llm-dvfs-env/bin/python"
        echo ""
        echo "   Quick check:"
        echo "     $PYTHON_BIN -c \"import torch; print(torch.__version__)\""
        exit 1
    fi
    echo "✓ Python dependencies available"
    
    # Check required scripts
    print_section "Required Scripts"
    if [ ! -f "$TRAINING_SCRIPT" ]; then
        echo "❌ Training script not found: $TRAINING_SCRIPT"
        exit 1
    else
        echo "✓ Training script: $TRAINING_SCRIPT"
    fi
    
    if [ ! -f "$PARSER_SCRIPT" ]; then
        echo "⚠️  Parser script not found (will skip parsing)"
    else
        echo "✓ Parser script: $PARSER_SCRIPT"
    fi
    
    # Check power mode
    print_section "Power Mode"
    sudo nvpmodel -q 2>/dev/null | grep "NV Power Mode" || echo "Mode info not available"
    
    # Check GPU configuration
    print_section "GPU Configuration"
    echo "Available frequencies:"
    cat ${GPU_DEVFREQ_PATH}/available_frequencies | awk '
    {
        for(i=1; i<=NF; i++) {
            printf "  %d MHz\n", $i/1000000
        }
    }'
    
    # Check disk space
    print_section "Disk Space"
    df -h "$SCRIPT_DIR" | tail -1
    
    # Check temperature
    print_section "System Temperature"
    local temp=$(check_temperature)
    echo "Current temperature: ${temp}°C"
    
    if [ $temp -gt 80 ]; then
        echo "❌ Temperature too high! Wait for cooldown."
        exit 1
    fi
    
    # Show experiment parameters
    print_section "Experiment Parameters (SAME as capping experiments)"
    echo "Model:      $MODEL_NAME"
    echo "Dataset:    $DATASET"
    echo "Epochs:     $EPOCHS"
    echo "Batch size: $BATCH_SIZE"
    echo "Log steps:  $LOG_STEPS"
    echo ""
    echo "This baseline will use DYNAMIC GPU SCALING (no capping)"
    echo "GPU will scale from min to max based on workload"
    
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    if [ "$YES" -eq 1 ]; then
        echo "Auto-confirming (--yes flag set)"
    else
        read -p "Start baseline experiment? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted by user"
            exit 1
        fi
    fi
}

# ============================================================================
# RUN BASELINE EXPERIMENT
# ============================================================================

run_baseline() {
    print_header "BASELINE EXPERIMENT - UNCAPPED GPU"
    
    local start_time=$(date +%s)
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local experiment_name="GPU_freq_BASELINE_uncapped"
    local output_dir="${OUTPUT_BASE}/${experiment_name}_${timestamp}"
    
    mkdir -p "$output_dir"
    
    # Setup baseline governor
    setup_baseline_governor
    
    # Check temperature
    local temp=$(check_temperature)
    echo ""
    echo "Pre-run temperature: ${temp}°C"
    
    # Monitor GPU frequency during idle
    print_section "GPU Frequency (Idle)"
    local idle_freq=$(get_current_gpu_freq)
    echo "Current GPU frequency: ${idle_freq} MHz"
    echo "This will increase during training as GPU scales up"
    
    # Save metadata
    cat > "${output_dir}/experiment_metadata.txt" << EOF
GPU BASELINE Experiment (No Frequency Capping)
===============================================
Date: $(date)
Method: Full Fine-Tuning
Model: $MODEL_NAME
Dataset: $DATASET
Epochs: $EPOCHS
Batch Size: $BATCH_SIZE
Log Steps: $LOG_STEPS

GPU Configuration:
  Governor: $(get_gpu_governor)
  Frequency limits: $(get_gpu_freq_limits)
  Current frequency: ${idle_freq} MHz (idle)
  Note: GPU will scale dynamically during training

System Info:
$(uname -a)
Python: $("$PYTHON_BIN" --version 2>&1)
Python bin: $PYTHON_BIN

Available GPU Frequencies:
$(cat ${GPU_DEVFREQ_PATH}/available_frequencies)
EOF
    
    # Start tegrastats
    print_section "Starting Monitoring"
    sudo tegrastats --interval $TEGRASTATS_INTERVAL --logfile "${output_dir}/tegrastats_measure.txt" &
    local TEGRA_PID=$!
    echo "✓ tegrastats started (PID: $TEGRA_PID)"
    sleep 2
    
    # Run training
    print_section "Training - BASELINE (Uncapped)"
    echo "GPU will scale dynamically based on workload"
    echo ""

    # Don't let `set -e` abort before we stop tegrastats; capture Python's exit code, not `tee`'s.
    set +e
    "$PYTHON_BIN" "$TRAINING_SCRIPT" \
        --model_name "$MODEL_NAME" \
        --dataset "$DATASET" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --log_every_n_steps $LOG_STEPS \
        --output_dir "$output_dir" \
        --enable_tqdm \
        2>&1 | tee "${output_dir}/python_output.txt"
    local train_exit_code=${PIPESTATUS[0]}
    set -e
    
    # Stop tegrastats
    sudo kill $TEGRA_PID 2>/dev/null
    echo "✓ tegrastats stopped"
    
    if [ $train_exit_code -ne 0 ]; then
        echo "❌ Training failed with exit code $train_exit_code"
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
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    temp=$(check_temperature)
    
    print_header "BASELINE EXPERIMENT COMPLETE"
    echo "Duration: $((duration / 60))m $((duration % 60))s"
    echo "Post-run temperature: ${temp}°C"
    echo "Results saved to: $output_dir"
    echo ""
    echo "This baseline represents UNCAPPED GPU performance"
    echo "Use this as reference for frequency capping comparisons"
}

# ============================================================================
# MAIN
# ============================================================================

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model_name) MODEL_NAME="$2"; shift 2 ;;
            --epochs)     EPOCHS="$2";     shift 2 ;;
            --batch_size) BATCH_SIZE="$2"; shift 2 ;;
            --dataset)    DATASET="$2";    shift 2 ;;
            --yes|-y)     YES=1;           shift   ;;
            *) echo "Unknown argument: $1"; exit 1 ;;
        esac
    done
}

main() {
    parse_args "$@"
    cd "$SCRIPT_DIR"

    preflight_checks
    run_baseline
    
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║              BASELINE EXPERIMENT COMPLETE                    ║"
    echo "╠══════════════════════════════════════════════════════════════╣"
    echo "║  Configuration: UNCAPPED (Dynamic GPU Scaling)               ║"
    echo "║  GPU scaled automatically based on workload                  ║"
    echo "║  Results: runs/gpu_freq_capping/GPU_freq_BASELINE_*/         ║"
    echo "║  Check Google Sheets for results                             ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Next step: Run frequency capping experiments for comparison"
}

main "$@"
