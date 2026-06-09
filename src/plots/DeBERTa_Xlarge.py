import pandas as pd
import matplotlib.pyplot as plt
import math

# =========================
# Data from screenshots
# =========================

# QNLI
qnli_data = {
    "config_freq_mhz": [306, 408, 510, 612, 714, 816, 918, 1020, 1122, 1224, 1300],
    "energy_j": [
        693591.48, 596669.90, 545930.79, 536503.90, 539312.18,
        549624.25, 572217.63, 610475.85, 641276.34, 698390.84, 737041.01
    ]
}
qnli_mode0_energy = 552288.86

# SST-2
sst2_data = {
    "config_freq_mhz": [306, 408, 510, 612, 714, 816, 918, 1020, 1122, 1224, 1300],
    "energy_j": [
        55292.41, 47512.24, 43025.92, 41044.05, 40927.33,
        40981.54, 42053.11, 44378.81, 46298.01, 49986.88, 52475.58
    ]
}
sst2_mode0_energy = 52317.17

qnli_df = pd.DataFrame(qnli_data)
sst2_df = pd.DataFrame(sst2_data)

# =========================
# Find optimal points
# =========================
qnli_opt_idx = qnli_df["energy_j"].idxmin()
qnli_opt_x = qnli_df.loc[qnli_opt_idx, "config_freq_mhz"]
qnli_opt_y = qnli_df.loc[qnli_opt_idx, "energy_j"]

sst2_opt_idx = sst2_df["energy_j"].idxmin()
sst2_opt_x = sst2_df.loc[sst2_opt_idx, "config_freq_mhz"]
sst2_opt_y = sst2_df.loc[sst2_opt_idx, "energy_j"]

# =========================
# Plot with broken Y-axis
# =========================
plt.rcParams["font.family"] = "DejaVu Sans"

fig, (ax_top, ax_bottom) = plt.subplots(
    2,
    1,
    sharex=True,
    figsize=(6.4, 5.2),
    dpi=120,
    gridspec_kw={"height_ratios": [1, 1], "hspace": 0.08}
)

# =========================
# Top plot: QNLI range
# =========================
ax_top.plot(
    qnli_df["config_freq_mhz"],
    qnli_df["energy_j"],
    color="#3498db",
    marker="o",
    markersize=5,
    linewidth=2.2,
    label="QNLI"
)

ax_top.scatter(
    qnli_opt_x,
    qnli_opt_y,
    color="#2ecc71",
    marker="*",
    s=280,
    zorder=5,
    label="QNLI Optimal"
)

ax_top.text(
    qnli_opt_x,
    qnli_opt_y - 12000,
    f"{qnli_opt_x} MHz",
    color="#27ae60",
    fontsize=9,
    fontweight="bold",
    ha="center",
    va="top"
)

ax_top.axhline(
    y=qnli_mode0_energy,
    color="#8e44ad",
    linestyle="--",
    linewidth=1.5,
    label="QNLI MAXN (Mode0)"
)

# =========================
# Bottom plot: SST-2 range
# =========================
ax_bottom.plot(
    sst2_df["config_freq_mhz"],
    sst2_df["energy_j"],
    color="#e67e22",
    marker="o",
    markersize=5,
    linewidth=2.2,
    label="SST-2"
)

ax_bottom.scatter(
    sst2_opt_x,
    sst2_opt_y,
    color="#e74c3c",
    marker="*",
    s=280,
    zorder=5,
    label="SST-2 Optimal"
)

ax_bottom.text(
    sst2_opt_x,
    sst2_opt_y - 900,
    f"{sst2_opt_x} MHz",
    color="#c0392b",
    fontsize=9,
    fontweight="bold",
    ha="center",
    va="top"
)

ax_bottom.axhline(
    y=sst2_mode0_energy,
    color="#7f8c8d",
    linestyle="--",
    linewidth=1.5,
    label="SST-2 MAXN (Mode0)"
)

# =========================
# Axis limits: compressed scale
# =========================
ax_top.set_ylim(520000, 760000)
ax_bottom.set_ylim(39000, 57000)

all_x = sorted(set(qnli_df["config_freq_mhz"]).union(set(sst2_df["config_freq_mhz"])))
ax_bottom.set_xticks(all_x)

ax_top.set_xlim(min(all_x) - 60, max(all_x) + 50)

# =========================
# Titles and labels
# =========================
ax_top.set_title(
    "DeBERTa-xlarge\nEnergy vs GPU Frequency",
    fontsize=14,
    fontweight="bold",
    pad=10
)

fig.text(
    0.03,
    0.5,
    "Energy (J)",
    va="center",
    rotation="vertical",
    fontsize=11
)

ax_bottom.set_xlabel(
    "GPU Frequency (MHz)",
    fontsize=11,
    fontstyle="italic"
)

# =========================
# Style
# =========================
for ax in [ax_top, ax_bottom]:
    ax.grid(True, linestyle="-", linewidth=0.8, alpha=0.25)

    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    ax.tick_params(axis="both", labelsize=9, colors="#333333")
    ax.ticklabel_format(style="plain", axis="y")

ax_top.spines["bottom"].set_visible(False)
ax_bottom.spines["top"].set_visible(False)

ax_top.tick_params(labelbottom=False)

# =========================
# Broken axis diagonal marks
# =========================
d = 0.012

kwargs = dict(transform=ax_top.transAxes, color="black", clip_on=False, linewidth=1.1)
ax_top.plot((-d, +d), (-d, +d), **kwargs)
ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)

kwargs = dict(transform=ax_bottom.transAxes, color="black", clip_on=False, linewidth=1.1)
ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

# =========================
# Legend under graph
# =========================
handles_top, labels_top = ax_top.get_legend_handles_labels()
handles_bottom, labels_bottom = ax_bottom.get_legend_handles_labels()

ax_bottom.legend(
    handles_top + handles_bottom,
    labels_top + labels_bottom,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.35),
    ncol=3,
    frameon=True,
    fontsize=8.5
)

plt.tight_layout(rect=[0.06, 0.08, 1, 1])
plt.show()
