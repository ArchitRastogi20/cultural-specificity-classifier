import matplotlib.pyplot as plt
import numpy as np

# =========================
# GLOBAL PARAMETERS
# =========================

# Figure size (in inches)
FIG_WIDTH = 10
FIG_HEIGHT = 7

# Base text size (ALL text scales from this)
BASE_FONT_SIZE = 20

# Output file name
OUTPUT_FILE = "f1_comparison.pdf"

# =========================
# DATA (HARDCODED)
# =========================

classes = [
    "Cultural Agnostic",
    "Cultural Representative",
    "Cultural Exclusive"
]

# F1-scores
lm_f1 = [0.90, 0.69, 0.68]
non_lm_f1 = [0.74, 0.47, 0.60]

# =========================
# STYLE SETTINGS
# =========================

lm_color = "#2C7FB8"      # Blue (LM – mDeBERTa)
non_lm_color = "#F39C12"  # Orange (Non-LM – XGBoost)

LABEL_FS  = BASE_FONT_SIZE - 1    # generic labels
TICK_FS   = BASE_FONT_SIZE + 1    # y-ticks
LEGEND_FS = BASE_FONT_SIZE + 1
ANNOT_FS  = BASE_FONT_SIZE - 2

YLABEL_FS = BASE_FONT_SIZE + 2    # y-axis label
XTICK_FS  = BASE_FONT_SIZE - 1    # x-axis tick labels

# =========================
# PLOTTING
# =========================

x = np.arange(len(classes))
width = 0.35

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

# Bars
bars_lm = ax.bar(
    x - width / 2,
    lm_f1,
    width,
    label="LM (mDeBERTa)",
    color=lm_color
)

bars_non_lm = ax.bar(
    x + width / 2,
    non_lm_f1,
    width,
    label="Non-LM (XGBoost)",
    color=non_lm_color
)

# =========================
# AXES & TICKS
# =========================

ax.set_ylabel("F1 Score", fontsize=YLABEL_FS)

ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=XTICK_FS)

ax.set_ylim(0, 1.0)
ax.tick_params(axis="y", labelsize=TICK_FS)

# =========================
# VALUE ANNOTATIONS
# =========================

for bar in bars_lm:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.02,
        f"{height:.2f}",
        ha="center",
        va="bottom",
        fontsize=ANNOT_FS
    )

for bar in bars_non_lm:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.02,
        f"{height:.2f}",
        ha="center",
        va="bottom",
        fontsize=ANNOT_FS
    )

# =========================
# LEGEND
# =========================

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.12),
    ncol=2,
    frameon=True,
    fontsize=LEGEND_FS
)

plt.tight_layout()

# =========================
# SAVE PDF
# =========================

plt.savefig(OUTPUT_FILE, format="pdf")
plt.close()

print(f"Saved figure to {OUTPUT_FILE}")
