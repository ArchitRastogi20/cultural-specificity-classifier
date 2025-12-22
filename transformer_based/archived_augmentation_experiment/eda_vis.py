import matplotlib.pyplot as plt

# =========================
# GLOBAL PARAMETERS
# =========================

# Figure size (in inches)
FIG_WIDTH = 10      # increase/decrease to scale image width
FIG_HEIGHT = 7     # increase/decrease to scale image height

# Base text size (ALL text scales from this)
BASE_FONT_SIZE = 20   # increase/decrease to scale text globally

# Output file name
OUTPUT_FILE = "class_distribution.pdf"

# =========================
# DATA (HARDCODED)
# =========================

categories = [
    "Cultural Agnostic",
    "Cultural Representative",
    "Cultural Exclusive"
]

# Percentages
train = [29.9, 27.0, 43.0]
test  = [39.0, 35.7, 25.3]

# Sample sizes
train_label = "Train (n=6,251)"
test_label  = "Test (n=300)"

# =========================
# STYLE SETTINGS
# =========================

train_color = "#2C7FB8"   # blue
test_color  = "#A6366A"   # maroon-pink

LABEL_FS  = BASE_FONT_SIZE -1
TICK_FS   = BASE_FONT_SIZE + 1
LEGEND_FS = BASE_FONT_SIZE + 1
ANNOT_FS  = BASE_FONT_SIZE
YLABEL_FS = BASE_FONT_SIZE +2
XTICK_FS = BASE_FONT_SIZE - 1

# =========================
# PLOTTING
# =========================

x = range(len(categories))
width = 0.35

plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

plt.bar(
    [i - width/2 for i in x],
    train,
    width=width,
    color=train_color,
    label=train_label
)

plt.bar(
    [i + width/2 for i in x],
    test,
    width=width,
    color=test_color,
    label=test_label
)

# Axis labels and ticks
plt.ylabel("Percentage (%)", fontsize=YLABEL_FS)
plt.xticks(x, categories, fontsize=XTICK_FS)
plt.yticks(fontsize=TICK_FS)
plt.ylim(0, 50)

# Bar annotations
for i, v in enumerate(train):
    plt.text(
        i - width/2,
        v + 1,
        f"{v}%",
        ha="center",
        fontsize=ANNOT_FS
    )

for i, v in enumerate(test):
    plt.text(
        i + width/2,
        v + 1,
        f"{v}%",
        ha="center",
        fontsize=ANNOT_FS
    )

# Legend (outside, fixed)
plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.18),
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
