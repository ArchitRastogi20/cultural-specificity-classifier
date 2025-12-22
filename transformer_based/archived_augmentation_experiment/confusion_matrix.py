import numpy as np
import matplotlib.pyplot as plt

# =========================
# GLOBAL PARAMETERS
# =========================

# Figure size (in inches) – SAME STYLE AS YOUR FILE
FIG_WIDTH = 10
FIG_HEIGHT = 7

# Base text size (ALL text scales from this)
BASE_FONT_SIZE = 20

# Output file name
OUTPUT_FILE = "confusion_matrix.pdf"

# =========================
# DATA (HARDCODED)
# =========================

labels = ["CA", "CE", "CR"]

# Confusion matrix counts
cm = np.array([
    [113,  2,  2],
    [  3, 50, 23],
    [ 12, 22, 73]
])

# Row-wise percentages (recall)
cm_percent = cm / cm.sum(axis=1, keepdims=True)

# =========================
# FONT SIZES
# =========================

AXIS_FS   = BASE_FONT_SIZE + 2
TICK_FS   = BASE_FONT_SIZE
ANNOT_FS  = BASE_FONT_SIZE - 1
CBAR_FS   = BASE_FONT_SIZE

# =========================
# PLOTTING
# =========================

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

# Heatmap
im = ax.imshow(cm_percent, cmap="Blues", vmin=0, vmax=1)

# Colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Recall", fontsize=CBAR_FS)
cbar.ax.tick_params(labelsize=TICK_FS)

# Axis labels
ax.set_xlabel("Predicted", fontsize=AXIS_FS)
ax.set_ylabel("True", fontsize=AXIS_FS)

# Ticks
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=TICK_FS)
ax.set_yticklabels(labels, fontsize=TICK_FS)

# =========================
# CELL ANNOTATIONS
# =========================

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        count = cm[i, j]
        percent = cm_percent[i, j] * 100
        text = f"{count}\n({percent:.1f}%)"

        ax.text(
            j, i, text,
            ha="center",
            va="center",
            fontsize=ANNOT_FS,
            color="white" if cm_percent[i, j] > 0.5 else "black"
        )

plt.tight_layout()

# =========================
# SAVE PDF
# =========================

plt.savefig(OUTPUT_FILE, format="pdf")
plt.close()

print(f"Saved figure to {OUTPUT_FILE}")
