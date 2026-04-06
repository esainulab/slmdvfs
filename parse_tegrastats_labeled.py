#!/usr/bin/env python3
"""
Enhanced Tegrastats Parser with Event Labeling

Parses tegrastats and labels each row with the corresponding training event.
Output: tegrastats_parsed.csv with event_name column

Usage:
    python3 parse_tegrastats_labeled.py <tegrastats.txt> <output.csv> [events_log.csv] [python_output.txt] [--gsheet]
"""

import re
import sys
import pandas as pd
from datetime import datetime
import argparse
import numpy as np

# Google Sheets integration (optional)
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSHEETS_AVAILABLE = True
except ImportError:
    GSHEETS_AVAILABLE = False

# ==================== CONFIGURATION ====================

GOOGLE_SHEET_NAME = "LLM-DVFS-Experiments"
WORKSHEET_NAME = "Experimental Results"
CREDENTIALS_FILE = "google_credentials.json"

# Column mapping for Google Sheets
GSHEET_COLUMNS = {
    "experiment_name": "Name",
    "cpu_freq_max_mhz": "CPU Freq (MHz)",
    "cpu_load_max": "CPU Load (%)",
    "gpu_freq_mhz": "GPU Freq (MHz)",
    "gpu_util_percent": "GPU Load (%)",
    "emc_freq_mhz": "Memory Freq (MHz)",
    "emc_util_percent": "Memory Load (%)",
    "ram_used_mb": "RAM (MB)",
    "ram_total_mb": "Total RAM (MB)",
    "power_VDD_CPU_CV_mw": "CPU Power (mW)",
    "power_VDD_GPU_SOC_mw": "GPU Power (mW)",
    "power_VIN_SYS_5V0_mw": "System Power (mW)",
    "total_power_mw": "Total Power (mW)",
    "train_time_s": "Train Time (s)",
    "accuracy": "Accuracy (%)",
    "energy_j": "Energy (J)",
    "sam_metric": "SAM Metric"
}

# ==================== REGEX PATTERNS ====================

timestamp_re = re.compile(r"^(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})")
ram_re = re.compile(r"RAM (\d+)/(\d+)MB")
lfb_re = re.compile(r"\(lfb (\d+)x(\d+)MB\)")
swap_re = re.compile(r"SWAP (\d+)/(\d+)MB")
cpu_re = re.compile(r"CPU \[(.*?)\]")
emc_re = re.compile(r"EMC_FREQ (\d+)%@(\d+)")
gr3d_re = re.compile(r"GR3D_FREQ (\d+)%")
gr3d_freq_re = re.compile(r"GR3D_FREQ \d+%@\[(\d+),(\d+)\]")
engine_re = re.compile(r"\b(NVENC|NVDEC|NVJPG|NVJPG1|VIC|OFA|NVDLA0|NVDLA1|PVA0_FREQ)\s+(off|on)\b")
ape_re = re.compile(r"\bAPE\s+(\d+)\b")
temp_re = re.compile(r"(\w+)@([\d\.]+)C")
power_re = re.compile(r"\b(VDD_[A-Z0-9_]+|VIN_SYS_5V0)\s+(\d+)mW")

# Python output patterns
train_time_re = re.compile(r"Train wall time:\s+([\d\.]+)\s*s", re.IGNORECASE)
accuracy_re = re.compile(r"Validation accuracy:\s+([\d\.]+)", re.IGNORECASE)

def parse_cpu_block(cpu_block: str):
    loads = []
    freqs = []
    on_cores = 0
    off_cores = 0

    for core in cpu_block.split(","):
        core = core.strip()
        if core == "off":
            off_cores += 1
            continue
        m = re.match(r"(\d+)%@(\d+)", core)
        if m:
            on_cores += 1
            loads.append(int(m.group(1)))
            freqs.append(int(m.group(2)))

    return loads, freqs, on_cores, off_cores

def simplify_event_name(event_name):
    """
    Simplify event names to meaningful phase labels
    
    Examples:
        epoch_1_start → epoch_1
        epoch_1_end → epoch_1
        epoch_1_step_0_forward → forward_pass
        epoch_1_step_100_backward → backward_pass
        epoch_1_step_50_optimizer → optimizer_step
        epoch_1_step_0_data_loading → data_loading
        data_loading_start → data_loading
        evaluation_start → evaluation
    """
    # Remove _start and _end suffixes
    name = event_name.replace('_start', '').replace('_end', '')
    
    # Handle detailed timing phases: epoch_X_step_Y_PHASE → PHASE
    if '_step_' in name:
        # Extract the phase (forward, backward, optimizer, data_loading)
        if 'forward' in name:
            return 'forward_pass'
        elif 'backward' in name:
            return 'backward_pass'
        elif 'optimizer' in name:
            return 'optimizer_step'
        elif 'data_loading' in name:
            return 'data_loading'
        else:
            # Generic step event → training
            parts = name.split('_')
            if len(parts) >= 2 and parts[0] == 'epoch':
                epoch_num = parts[1]
                return f'epoch_{epoch_num}_training'
    
    # Handle epoch-level training events
    if 'training' in name and 'epoch_' in name:
        # epoch_1_training → epoch_1_training
        parts = name.split('_')
        if len(parts) >= 2:
            return f'epoch_{parts[1]}_training'
    
    return name

def load_events(events_log_path):
    """Load and process events log"""
    events = pd.read_csv(events_log_path)
    
    # Parse timestamps
    events['timestamp_dt'] = pd.to_datetime(events['timestamp'], format='%m-%d-%Y %H:%M:%S.%f')
    
    # Simplify event names
    events['simple_name'] = events['phase'].apply(simplify_event_name)
    
    # Sort by timestamp
    events = events.sort_values('timestamp_dt').reset_index(drop=True)
    
    return events

def extract_train_time_from_events(events_df: pd.DataFrame):
    """
    Fallback when python_output.txt doesn't include train time.
    Computes (training_end - training_start) in seconds using unix_timestamp or timestamps.
    """
    if events_df is None or events_df.empty:
        return None

    if "phase" not in events_df.columns:
        return None

    starts = events_df[events_df["phase"] == "training_start"]
    ends = events_df[events_df["phase"] == "training_end"]

    if starts.empty or ends.empty:
        return None

    start_row = starts.iloc[0]
    end_row = ends.iloc[-1]

    if "unix_timestamp" in events_df.columns:
        try:
            return float(end_row["unix_timestamp"]) - float(start_row["unix_timestamp"])
        except Exception:
            pass

    if "timestamp_dt" in events_df.columns:
        try:
            return (end_row["timestamp_dt"] - start_row["timestamp_dt"]).total_seconds()
        except Exception:
            pass

    if "elapsed_seconds" in events_df.columns:
        try:
            return float(end_row["elapsed_seconds"]) - float(start_row["elapsed_seconds"])
        except Exception:
            pass

    return None

def extract_accuracy_from_events(events_df: pd.DataFrame):
    """
    Fallback when python_output.txt doesn't include accuracy.
    Uses the most recent non-null {final_accuracy, accuracy} value.
    """
    if events_df is None or events_df.empty:
        return None

    for col in ["final_accuracy", "accuracy"]:
        if col not in events_df.columns:
            continue
        series = pd.to_numeric(events_df[col], errors="coerce").dropna()
        if series.empty:
            continue
        val = float(series.iloc[-1])
        # Heuristic: GLUE accuracy is usually 0..1; Google Sheets expects percent.
        return val * 100.0 if val <= 1.0 else val

    return None

def label_tegrastats_with_events(tegra_df, events_df):
    """
    Label each tegrastats row with the event it belongs to
    Uses time-based matching: assign each tegrastats sample to the active event at that time
    """
    print("\n🏷️  Labeling tegrastats samples with events...")

    tegra = tegra_df.copy()
    events = events_df.copy()

    tegra = tegra.sort_values("timestamp_dt")
    events = events.sort_values("timestamp_dt")

    # Efficient "latest event at or before sample timestamp"
    merged = pd.merge_asof(
        tegra[["timestamp_dt"]],
        events[["timestamp_dt", "simple_name"]],
        on="timestamp_dt",
        direction="backward",
        allow_exact_matches=True,
    )

    tegra["event_name"] = merged["simple_name"].fillna("unknown")

    # Count samples per event
    event_counts = tegra["event_name"].value_counts()
    print("\n📊 Samples per event:")
    for event, count in event_counts.items():
        print(f"   {event:.<40} {count:>5} samples")

    # Preserve original row order
    tegra = tegra.sort_index()
    return tegra

def parse_tegrastats(input_file):
    """Parse tegrastats file into DataFrame"""
    rows = []
    
    with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "RAM" not in line:
                continue

            row = {}

            # Timestamp
            ts = timestamp_re.search(line)
            if ts:
                row["timestamp"] = datetime.strptime(ts.group(1), "%m-%d-%Y %H:%M:%S")
            else:
                row["timestamp"] = None

            # RAM + LFB
            ram = ram_re.search(line)
            if ram:
                row["ram_used_mb"] = int(ram.group(1))
                row["ram_total_mb"] = int(ram.group(2))

            lfb = lfb_re.search(line)
            if lfb:
                row["lfb_blocks"] = int(lfb.group(1))
                row["lfb_block_mb"] = int(lfb.group(2))

            # SWAP
            swap = swap_re.search(line)
            if swap:
                row["swap_used_mb"] = int(swap.group(1))
                row["swap_total_mb"] = int(swap.group(2))

            # CPU
            cpu = cpu_re.search(line)
            if cpu:
                loads, freqs, on_cores, off_cores = parse_cpu_block(cpu.group(1))
                row["cpu_cores_on"] = on_cores
                row["cpu_cores_off"] = off_cores
                row["cpu_load_avg"] = (sum(loads) / len(loads)) if loads else 0
                row["cpu_load_max"] = max(loads) if loads else 0
                row["cpu_freq_avg_mhz"] = (sum(freqs) / len(freqs)) if freqs else 0
                row["cpu_freq_max_mhz"] = max(freqs) if freqs else 0

            # EMC (memory controller)
            emc = emc_re.search(line)
            if emc:
                row["emc_util_percent"] = int(emc.group(1))
                row["emc_freq_mhz"] = int(emc.group(2))

            # GPU util
            gutil = gr3d_re.search(line)
            if gutil:
                row["gpu_util_percent"] = int(gutil.group(1))

            # GPU freq (if present)
            gfreq = gr3d_freq_re.search(line)
            if gfreq:
                row["gpu_freq_mhz"] = int(gfreq.group(1))
                row["gpu_freq_aux"] = int(gfreq.group(2))

            # Engines on/off -> 1/0
            for name, state in engine_re.findall(line):
                row[f"engine_{name.lower()}"] = 1 if state == "on" else 0

            # APE
            ape = ape_re.search(line)
            if ape:
                row["ape"] = int(ape.group(1))

            # Temperatures
            for name, temp in temp_re.findall(line):
                row[f"temp_{name}_c"] = float(temp)

            # Power rails
            for rail, mw in power_re.findall(line):
                row[f"power_{rail}_mw"] = int(mw)

            rows.append(row)

    df = pd.DataFrame(rows)

    # Make sure some useful columns exist even if missing
    for col in ["gpu_freq_mhz", "gpu_util_percent", "emc_freq_mhz", "emc_util_percent"]:
        if col not in df.columns:
            df[col] = pd.NA

    return df

def parse_python_output(python_output_file):
    """Extract training time and accuracy from python output"""
    train_time = None
    accuracy = None
    
    try:
        with open(python_output_file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            
            match = train_time_re.search(content)
            if match:
                train_time = float(match.group(1))
            
            match = accuracy_re.search(content)
            if match:
                accuracy = float(match.group(1)) * 100
    except FileNotFoundError:
        print(f"⚠️  Warning: Python output file not found: {python_output_file}")
    except Exception as e:
        print(f"⚠️  Warning: Error parsing python output: {e}")
    
    return train_time, accuracy

def compute_event_averages(df):
    """Compute average metrics per event"""
    # Group by event_name (excluding 'unknown')
    events = df[df['event_name'] != 'unknown'].copy()
    
    if len(events) == 0:
        return pd.DataFrame()
    
    # Numeric columns to average
    numeric_cols = [col for col in df.columns if col not in ['timestamp', 'timestamp_dt', 'event_name']]
    numeric_cols = [col for col in numeric_cols if pd.api.types.is_numeric_dtype(df[col])]
    
    # Compute averages per event
    event_avgs = events.groupby('event_name')[numeric_cols].mean().reset_index()
    
    return event_avgs

def compute_overall_averages(df):
    """Compute overall averages for all samples (for AVG row)"""
    avg_row = {'timestamp': 'AVG', 'event_name': 'overall'}
    
    for col in df.columns:
        if col in ['timestamp', 'timestamp_dt', 'event_name']:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            try:
                avg_row[col] = int(pd.to_numeric(df[col], errors='coerce').mean())
            except Exception:
                avg_row[col] = ""
        else:
            avg_row[col] = ""
    
    return avg_row

def compute_derived_metrics(avg_row, train_time, accuracy):
    """Compute total power, energy, and SAM metric"""
    cpu_power = avg_row.get("power_VDD_CPU_CV_mw", 0) or 0
    gpu_power = avg_row.get("power_VDD_GPU_SOC_mw", 0) or 0
    total_power_mw = cpu_power + gpu_power
    avg_row["total_power_mw"] = total_power_mw
    
    avg_row["train_time_s"] = train_time if train_time is not None else ""
    avg_row["accuracy"] = round(accuracy, 2) if accuracy is not None else ""
    
    if train_time is not None and total_power_mw:
        energy_j = (total_power_mw * train_time) / 1000.0
        avg_row["energy_j"] = round(energy_j, 2)
    else:
        avg_row["energy_j"] = ""
    
    if accuracy is not None and avg_row.get("energy_j"):
        import math
        alpha = 1
        beta = 5
        try:
            sam = beta * (accuracy ** alpha) / math.log10(avg_row["energy_j"])
            avg_row["sam_metric"] = round(sam, 2)
        except (ValueError, ZeroDivisionError):
            avg_row["sam_metric"] = ""
    else:
        avg_row["sam_metric"] = ""
    
    return avg_row

def upload_to_google_sheets(avg_row, experiment_name, credentials_file, sheet_name, worksheet_name):
    """Upload average results to Google Sheets"""
    if not GSHEETS_AVAILABLE:
        print("⚠️  Google Sheets libraries not installed. Install with:")
        print("   pip install gspread google-auth")
        return False
    
    try:
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
        creds = Credentials.from_service_account_file(credentials_file, scopes=scopes)
        client = gspread.authorize(creds)
        
        try:
            spreadsheet = client.open(sheet_name)
        except gspread.SpreadsheetNotFound:
            print(f"⚠️  Spreadsheet '{sheet_name}' not found. Creating it...")
            spreadsheet = client.create(sheet_name)
            spreadsheet.share('', perm_type='anyone', role='writer')
        
        try:
            worksheet = spreadsheet.worksheet(worksheet_name)
        except gspread.WorksheetNotFound:
            print(f"   Creating worksheet '{worksheet_name}'...")
            worksheet = spreadsheet.add_worksheet(title=worksheet_name, rows=1000, cols=26)
        
        data_row = {}
        data_row["experiment_name"] = experiment_name
        
        for internal_col, display_name in GSHEET_COLUMNS.items():
            if internal_col == "experiment_name":
                continue
            value = avg_row.get(internal_col, "")
            data_row[internal_col] = value
        
        all_values = worksheet.get_all_values()
        
        if not all_values or not all_values[0]:
            headers = [GSHEET_COLUMNS.get(k, k) for k in data_row.keys()]
            worksheet.append_row(headers)
            print(f"   Added headers with display names")
        
        if all_values and all_values[0]:
            header_row = all_values[0]
            row_values = []
            for header in header_row:
                internal_name = None
                for int_name, disp_name in GSHEET_COLUMNS.items():
                    if disp_name == header:
                        internal_name = int_name
                        break
                
                if internal_name and internal_name in data_row:
                    row_values.append(str(data_row[internal_name]))
                else:
                    row_values.append("")
            worksheet.append_row(row_values)
        else:
            row_values = [str(v) for v in data_row.values()]
            worksheet.append_row(row_values)
        
        print(f"✅ Uploaded to Google Sheets: {sheet_name} / {worksheet_name}")
        print(f"   Experiment: {experiment_name}")
        print(f"   SAM Metric: {avg_row.get('sam_metric', 'N/A')}")
        return True
        
    except FileNotFoundError:
        print(f"⚠️  Credentials file not found: {credentials_file}")
        return False
    except Exception as e:
        print(f"⚠️  Error uploading to Google Sheets: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Parse tegrastats with event labeling"
    )
    parser.add_argument("tegrastats_file", help="Input tegrastats text file")
    parser.add_argument("output_file", help="Output CSV file")
    parser.add_argument("events_log", nargs="?", default=None, help="Events log CSV file (optional)")
    parser.add_argument("python_output", nargs="?", default=None,
                       help="Python script output file (for train time and accuracy)")
    parser.add_argument("--gsheet", action="store_true", 
                       help="Upload results to Google Sheets")
    parser.add_argument("--experiment", type=str, default="",
                       help="Experiment name for Google Sheets")
    parser.add_argument("--credentials", type=str, default=CREDENTIALS_FILE,
                       help="Path to Google service account JSON")
    parser.add_argument("--sheet-name", type=str, default=GOOGLE_SHEET_NAME,
                       help="Google Sheet name")
    parser.add_argument("--worksheet", type=str, default=WORKSHEET_NAME,
                       help="Worksheet/tab name")
    
    args = parser.parse_args()
    
    # Parse tegrastats
    print(f"📊 Parsing tegrastats: {args.tegrastats_file}...")
    df = parse_tegrastats(args.tegrastats_file)
    
    if df.empty:
        print("⚠️  No data found in tegrastats file!")
        sys.exit(1)
    
    print(f"   Loaded {len(df)} tegrastats samples")
    
    # Add timestamp_dt for matching
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
    
    # Load events log (optional)
    events_df = None
    if args.events_log:
        try:
            print(f"\n📋 Loading events log: {args.events_log}...")
            events_df = load_events(args.events_log)
            print(f"   Loaded {len(events_df)} events")

            # Label tegrastats with events
            df = label_tegrastats_with_events(df, events_df)
        except FileNotFoundError:
            print(f"⚠️  Events log not found: {args.events_log} (skipping labeling)")
        except Exception as e:
            print(f"⚠️  Failed to load/label events log: {e} (skipping labeling)")

    if events_df is None:
        df["event_name"] = "overall"
    
    # Compute event averages
    print("\n📈 Computing per-event averages...")
    event_avgs = compute_event_averages(df)
    if not event_avgs.empty:
        print("\nEvent Averages:")
        print(event_avgs[['event_name', 'cpu_load_avg', 'gpu_util_percent', 'power_VDD_GPU_SOC_mw']].to_string(index=False))
    
    # Compute overall averages
    avg_row = compute_overall_averages(df)
    
    # Parse python output if provided
    train_time = None
    accuracy = None
    if args.python_output:
        print(f"\n📊 Parsing python output: {args.python_output}...")
        train_time, accuracy = parse_python_output(args.python_output)
        
        if train_time is not None:
            print(f"   Train time: {train_time:.2f} s")
        if accuracy is not None:
            print(f"   Accuracy: {accuracy:.2f}%")

    # Fallback: derive train_time / accuracy from events if missing
    if train_time is None and events_df is not None:
        train_time = extract_train_time_from_events(events_df)
        if train_time is not None:
            print(f"   (fallback) Train time from events: {train_time:.2f} s")

    if accuracy is None and events_df is not None:
        accuracy = extract_accuracy_from_events(events_df)
        if accuracy is not None:
            print(f"   (fallback) Accuracy from events: {accuracy:.2f}%")
    
    # Compute derived metrics
    avg_row = compute_derived_metrics(avg_row, train_time, accuracy)
    
    # Add AVG row
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    
    # Reorder columns: timestamp, event_name, then the rest
    cols = ['timestamp', 'event_name'] + [c for c in df.columns if c not in ['timestamp', 'event_name', 'timestamp_dt']]
    df = df[cols]
    
    # Save to CSV
    df.to_csv(args.output_file, index=False)
    print(f"\n✅ Saved: {args.output_file}")
    print(f"   Total rows: {len(df)} ({len(df)-1} samples + 1 AVG row)")
    
    # Display computed metrics
    if avg_row.get("total_power_mw"):
        print(f"\n📈 Computed Metrics:")
        print(f"   Total Power: {avg_row['total_power_mw']} mW")
    if avg_row.get("energy_j"):
        print(f"   Energy: {avg_row['energy_j']:.2f} J")
    if avg_row.get("sam_metric"):
        print(f"   SAM Metric: {avg_row['sam_metric']:.2f}")
    
    # Upload to Google Sheets if requested
    if args.gsheet:
        experiment_name = args.experiment
        if not experiment_name:
            import os
            path_parts = args.tegrastats_file.split('/')
            if 'runs' in path_parts:
                idx = path_parts.index('runs')
                if idx + 1 < len(path_parts):
                    experiment_name = path_parts[idx + 1]
            if not experiment_name:
                experiment_name = os.path.basename(args.tegrastats_file).replace('.txt', '')
        
        print(f"\n📤 Uploading to Google Sheets...")
        upload_to_google_sheets(
            avg_row=avg_row,
            experiment_name=experiment_name,
            credentials_file=args.credentials,
            sheet_name=args.sheet_name,
            worksheet_name=args.worksheet
        )
    else:
        print("\n💡 Tip: Use --gsheet flag to upload results to Google Sheets")

if __name__ == "__main__":
    main()
