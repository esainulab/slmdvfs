import pandas as pd
import matplotlib.pyplot as plt
import math

# =========================
# Data from the table
# =========================
data = {
    "model": ["Gemma-2B"] * 6,
    "dataset": ["qnli"] * 6,
    "config": [
        "Mode0", "306MHz", "408MHz",
        "510MHz", "1224MHz", "1300MHz"
    ],
    "gpu_freq_mhz": [1293, 305, 407, 507, 1219, 1295],
    "energy_j": [
        694858.59, 721533.29, 615101.16,
        595930.38, 712022.74, 748437.96
    ],
}

df = pd.DataFrame(data)

# =========================
# Separate Manual Sweep and MAXN Mode0
# =========================
manual_df = df[df["config"] != "Mode0"].copy()
mode0_df = df[df["config"] == "Mode0"].copy()

# Use config frequency on X-axis, as in the reference graphs
manual_df["config_freq_mhz"] = (
    manual_df["config"]
    .str.replace("MHz", "", regex=False)
    .astype(int)
)

manual_df = manual_df.sort_values("config_freq_mhz").reset_index(drop=True)

mode0_energy = mode0_df["energy_j"].iloc[0]

# =========================
# Find optimal point
# =========================
optimal_idx = manual_df["energy_j"].idxmin()
optimal_x = manual_df.loc[optimal_idx, "config_freq_mhz"]
optimal_y = manual_df.loc[optimal_idx, "energy_j"]

# =========================
# Plot
# =========================
plt.rcParams["font.family"] = "DejaVu Sans"

fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=120)

# Manual Sweep line
ax.plot(
    manual_df["config_freq_mhz"],
    manual_df["energy_j"],
    color="#3498db",
    marker="o",
    markersize=5,
    linewidth=2.2,
    label="Manual Sweep"
)

# Optimal point
ax.scatter(
    optimal_x,
    optimal_y,
    color="#2ecc71",
    marker="*",
    s=320,
    zorder=5,
    label="Optimal"
)

# Optimal frequency label
ax.text(
    optimal_x,
    optimal_y - 13000,
    f"{optimal_x} MHz",
    color="#27ae60",
    fontsize=9,
    fontweight="bold",
    ha="center",
    va="top"
)

# MAXN Mode0 horizontal dashed line
ax.axhline(
    y=mode0_energy,
    color="#b12ac8",
    linestyle="--",
    linewidth=1.5,
    label="MAXN (Mode0)"
)

# =========================
# Titles and labels
# =========================
ax.set_title(
    "Gemma-2B\n(2B params)",
    fontsize=14,
    fontweight="bold",
    pad=8
)

ax.set_xlabel(
    "GPU Frequency (MHz)\nQNLI",
    fontsize=10,
    fontstyle="italic"
)

ax.set_ylabel(
    "Energy (J)",
    fontsize=11
)

# =========================
# Grid and axes styling
# =========================
ax.grid(
    True,
    linestyle="-",
    linewidth=0.8,
    alpha=0.25
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["left"].set_color("#777777")
ax.spines["bottom"].set_color("#777777")

ax.spines["left"].set_linewidth(1.2)
ax.spines["bottom"].set_linewidth(1.2)

ax.tick_params(axis="both", labelsize=9, colors="#333333")

# Show only existing manual frequencies on X-axis
ax.set_xticks(manual_df["config_freq_mhz"])

# Avoid scientific notation on Y-axis
ax.ticklabel_format(style="plain", axis="y")

# =========================
# Axis limits similar to reference graphs
# =========================
x_min = manual_df["config_freq_mhz"].min() - 60
x_max = manual_df["config_freq_mhz"].max() + 50
ax.set_xlim(x_min, x_max)

y_min_raw = min(manual_df["energy_j"].min(), mode0_energy)
y_max_raw = max(manual_df["energy_j"].max(), mode0_energy)

y_min = math.floor((y_min_raw - 30000) / 10000) * 10000
y_max = math.ceil((y_max_raw + 30000) / 10000) * 10000

ax.set_ylim(y_min, y_max)

# =========================
# Legend under the graph
# =========================
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.23),
    ncol=3,
    frameon=True,
    fontsize=9
)

plt.tight_layout()
plt.show()
