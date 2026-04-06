#!/usr/bin/env python3
"""
Overlay events_log.csv with tegrastats data
Creates analysis showing CPU/GPU metrics during each training phase
"""

import pandas as pd
import sys
import argparse
from datetime import datetime
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Overlay events with tegrastats")
    parser.add_argument("events_log", help="Path to events_log.csv")
    parser.add_argument("tegrastats_csv", help="Path to parsed tegrastats CSV")
    parser.add_argument("output", help="Output CSV path for overlaid data")
    parser.add_argument("--summary", help="Optional summary output path", default=None)
    return parser.parse_args()

def load_events(events_path):
    """Load events log"""
    events = pd.read_csv(events_path)
    # Convert timestamp to datetime
    events['timestamp_dt'] = pd.to_datetime(events['timestamp'], format='mixed')
    return events

def load_tegrastats(tegrastats_path):
    """Load tegrastats CSV"""
    tegra = pd.read_csv(tegrastats_path)
    # Convert timestamp (excluding AVG row)
    tegra = tegra[tegra['timestamp'] != 'AVG'].copy()
    tegra['timestamp_dt'] = pd.to_datetime(tegra['timestamp'], format='mixed')
    return tegra

def find_phase_metrics(tegra_df, phase_start, phase_end):
    """Find tegrastats metrics during a phase"""
    # Get all tegrastats samples within the phase time window
    phase_data = tegra_df[
        (tegra_df['timestamp_dt'] >= phase_start) &
        (tegra_df['timestamp_dt'] <= phase_end)
    ]
    
    if len(phase_data) == 0:
        return None
    
    # Compute averages for key metrics
    metrics = {}
    
    # Numeric columns to average
    numeric_cols = [
        'cpu_freq_avg_mhz', 'cpu_freq_max_mhz', 'cpu_load_avg', 'cpu_load_max',
        'gpu_freq_mhz', 'gpu_util_percent',
        'emc_freq_mhz', 'emc_util_percent',
        'ram_used_mb', 'power_VDD_CPU_CV_mw', 'power_VDD_GPU_SOC_mw', 'power_VIN_SYS_5V0_mw'
    ]
    
    for col in numeric_cols:
        if col in phase_data.columns:
            values = pd.to_numeric(phase_data[col], errors='coerce')
            metrics[f'avg_{col}'] = values.mean()
            metrics[f'max_{col}'] = values.max()
            metrics[f'min_{col}'] = values.min()
    
    metrics['num_samples'] = len(phase_data)
    metrics['duration_s'] = (phase_end - phase_start).total_seconds()
    
    return metrics

def create_overlay(events_df, tegra_df):
    """Create overlay of events with tegrastats metrics"""
    overlay_data = []
    
    print(f"\n{'='*80}")
    print("Overlaying events with tegrastats data...")
    print(f"{'='*80}\n")
    
    # Group events into phases (consecutive events define phase boundaries)
    for i in range(len(events_df) - 1):
        event = events_df.iloc[i]
        next_event = events_df.iloc[i + 1]
        
        phase_start = event['timestamp_dt']
        phase_end = next_event['timestamp_dt']
        
        # Find metrics during this phase
        phase_metrics = find_phase_metrics(tegra_df, phase_start, phase_end)
        
        if phase_metrics is None:
            print(f"⚠️  No tegrastats data for phase: {event['phase']}")
            continue
        
        # Create overlay record
        overlay = {
            'phase': event['phase'],
            'description': event['description'],
            'start_time': event['timestamp'],
            'end_time': next_event['timestamp'],
            'duration_s': phase_metrics['duration_s'],
            'elapsed_start_s': event['elapsed_seconds'],
        }
        
        # Add all metrics
        overlay.update(phase_metrics)
        
        # Add event metadata if present
        for col in events_df.columns:
            if col not in ['timestamp', 'unix_timestamp', 'elapsed_seconds', 'phase', 'description', 'timestamp_dt']:
                if pd.notna(event[col]):
                    overlay[f'event_{col}'] = event[col]
        
        overlay_data.append(overlay)
        
        # Print progress for major phases
        if 'epoch' in event['phase'] or 'evaluation' in event['phase'] or 'training' in event['phase']:
            print(f"✓ {event['phase']:<40} | {phase_metrics['duration_s']:>6.2f}s | "
                  f"GPU: {phase_metrics.get('avg_gpu_util_percent', 0):>5.1f}% @ {phase_metrics.get('avg_gpu_freq_mhz', 0):>4.0f}MHz | "
                  f"CPU: {phase_metrics.get('avg_cpu_load_avg', 0):>5.1f}%")
    
    return pd.DataFrame(overlay_data)

def create_phase_summary(overlay_df):
    """Create summary statistics by phase type"""
    # Group by phase type (e.g., all step_start, all step_end, etc.)
    overlay_df['phase_type'] = overlay_df['phase'].str.extract(r'(epoch_\d+|training|evaluation|data_loading|model_loading|tokenization|step_\d+)')[0]
    
    # Compute summary by phase type
    summary_cols = [
        'duration_s', 
        'avg_cpu_load_avg', 'avg_cpu_freq_avg_mhz',
        'avg_gpu_util_percent', 'avg_gpu_freq_mhz',
        'avg_power_VDD_GPU_SOC_mw', 'avg_power_VDD_CPU_CV_mw', 'avg_power_VIN_SYS_5V0_mw'
    ]
    
    summary = overlay_df.groupby('phase_type')[summary_cols].agg(['mean'])
    
    return summary

def main():
    args = parse_args()
    
    print("=" * 80)
    print("Events & Tegrastats Overlay Tool")
    print("=" * 80)
    
    # Load data
    print(f"\n📂 Loading events from: {args.events_log}")
    events_df = load_events(args.events_log)
    print(f"   Loaded {len(events_df)} events")
    
    print(f"\n📂 Loading tegrastats from: {args.tegrastats_csv}")
    tegra_df = load_tegrastats(args.tegrastats_csv)
    print(f"   Loaded {len(tegra_df)} tegrastats samples")
    
    # Create overlay
    overlay_df = create_overlay(events_df, tegra_df)
    
    # Save overlay
    overlay_df.to_csv(args.output, index=False)
    print(f"\n✅ Overlay saved to: {args.output}")
    print(f"   Total phases analyzed: {len(overlay_df)}")
    
    # Create and save summary if requested
    if args.summary:
        summary_df = create_phase_summary(overlay_df)
        summary_df.to_csv(args.summary)
        print(f"✅ Summary saved to: {args.summary}")
    
    # Print key statistics
    print(f"\n{'='*80}")
    print("KEY STATISTICS")
    print(f"{'='*80}")
    
    # Find training epochs
    epoch_phases = overlay_df[overlay_df['phase'].str.contains('epoch_.*_end', na=False)]
    if len(epoch_phases) > 0:
        print("\nPer-Epoch Metrics:")
        for _, row in epoch_phases.iterrows():
            print(f"  {row['phase']:<20}: {row['duration_s']:>6.1f}s | "
                  f"GPU: {row.get('avg_gpu_util_percent', 0):>5.1f}% @ {row.get('avg_gpu_freq_mhz', 0):>4.0f}MHz | "
                  f"Power: {row.get('avg_power_VIN_SYS_5V0_mw', 0):>5.0f}mW")
    
    # Find evaluation phases
    eval_phases = overlay_df[overlay_df['phase'].str.contains('evaluation', na=False)]
    if len(eval_phases) > 0:
        print("\nEvaluation Metrics:")
        for _, row in eval_phases.iterrows():
            print(f"  {row['phase']:<20}: {row['duration_s']:>6.1f}s | "
                  f"GPU: {row.get('avg_gpu_util_percent', 0):>5.1f}% @ {row.get('avg_gpu_freq_mhz', 0):>4.0f}MHz")
    
    # Find data loading phases
    data_phases = overlay_df[overlay_df['phase'].str.contains('data_loading|tokenization', na=False)]
    if len(data_phases) > 0:
        print("\nData Preparation Metrics:")
        for _, row in data_phases.iterrows():
            print(f"  {row['phase']:<20}: {row['duration_s']:>6.1f}s | "
                  f"CPU: {row.get('avg_cpu_load_avg', 0):>5.1f}% | "
                  f"GPU: {row.get('avg_gpu_util_percent', 0):>5.1f}%")
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    main()