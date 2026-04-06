import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = "/home/nvidia/Desktop/llm-dvfs/runs/phase_fullft_mrpc/tegrastats_parsed.csv"
MAX_POINTS = 6000  # for large logs (keeps plot responsive)

# Load data
df = pd.read_csv(CSV_PATH)

# ---- Time axis (seconds since start) ----
df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.sort_values("timestamp_dt").reset_index(drop=True)
t0 = df["timestamp_dt"].iloc[0]
df["elapsed_s"] = (df["timestamp_dt"] - t0).dt.total_seconds()

# ---- Power columns ----
CPU_COL = "power_VDD_CPU_CV_mw"
GPU_COL = "power_VDD_GPU_SOC_mw"

if CPU_COL not in df.columns or GPU_COL not in df.columns:
    raise ValueError("Required power columns not found in CSV")

# Total power = CPU + GPU (as requested)
df["total_power_cpu_gpu_mw"] = df[CPU_COL].astype(float) + df[GPU_COL].astype(float)

# ---- Build event background segments (FULL resolution) ----
event = df["event_name"].fillna("unknown").astype(str).to_numpy()
t = df["elapsed_s"].to_numpy()

change_idx = np.flatnonzero(event[1:] != event[:-1]) + 1
starts = np.r_[0, change_idx]
ends = np.r_[change_idx, len(df)]

segments = [(t[s], t[e-1], event[s]) for s, e in zip(starts, ends) if e - s > 0]

# Color map for events
unique_events = list(dict.fromkeys([seg[2] for seg in segments]))
cmap = plt.get_cmap("tab20")
event_to_color = {ev: cmap(i % cmap.N) for i, ev in enumerate(unique_events)}

# ---- Downsample ONLY the line (not the background) ----
step = max(1, int(np.ceil(len(df) / MAX_POINTS)))
df_plot = df.iloc[::step].copy()

def add_event_background(ax):
    for x0, x1, ev in segments:
        if x1 > x0:
            ax.axvspan(x0, x1, alpha=0.10, color=event_to_color[ev], linewidth=0)

def plot_graph(y_col, title, ylabel):
    plt.figure(figsize=(12, 4))
    ax = plt.gca()
    
    # Background event bands
    add_event_background(ax)
    
    # Line plot (downsampled)
    ax.plot(df_plot["elapsed_s"], df_plot[y_col], linewidth=1)
    
    ax.set_title(title)
    ax.set_xlabel("Time (s) since start")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    
    # Compact legend for event colors
    show = unique_events[:10]
    handles = [
        plt.Line2D([0], [0], color=event_to_color[e], linewidth=6, alpha=0.6)
        for e in show
    ]
    labels = show.copy()
    if len(unique_events) > 10:
        handles.append(plt.Line2D([0], [0], color="none"))
        labels.append(f"+{len(unique_events)-10} more events")
    
    ax.legend(handles, labels, loc="upper right", fontsize=8, title="event_name (bg)")
    plt.tight_layout()
    plt.show()

# ---- Display all three graphs on screen ----
plot_graph(CPU_COL, "CPU Power vs Time", "Power (mW)")
plot_graph(GPU_COL, "GPU Power vs Time", "Power (mW)")
plot_graph("total_power_cpu_gpu_mw", "Total Power vs Time (CPU + GPU)", "Power (mW)")