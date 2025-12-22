import glob
import pandas as pd
from collections import Counter

# =========================
# USER INPUTS (edit these)
# =========================
TRAIN_CSV_GLOB = r"data\train.csv"
VALID_CSV_GLOB = r"data\valid.csv"
OUTPUT_TXT_PATH = r"eda\class_distribution.txt"

LABEL_COL = "label"

# =========================
# GLOBAL VARIABLE (RESULT)
# =========================
CLASS_DISTRIBUTION = {}

def compute_class_stats(csv_glob):
    files = glob.glob(csv_glob)
    if not files:
        raise ValueError(f"No CSV files found for pattern: {csv_glob}")

    all_labels = []

    for file in files:
        df = pd.read_csv(file)
        if LABEL_COL not in df.columns:
            raise ValueError(f"Column '{LABEL_COL}' not found in {file}")
        all_labels.extend(df[LABEL_COL].dropna().tolist())

    total = len(all_labels)
    counts = Counter(all_labels)

    percentages = {
        cls: round((count / total) * 100, 2)
        for cls, count in counts.items()
    }

    return counts, percentages, total


def main():
    global CLASS_DISTRIBUTION

    train_counts, train_pct, train_total = compute_class_stats(TRAIN_CSV_GLOB)
    valid_counts, valid_pct, valid_total = compute_class_stats(VALID_CSV_GLOB)

    CLASS_DISTRIBUTION = {
        "train": {
            "total_samples": train_total,
            "counts": dict(train_counts),
            "percentages": train_pct
        },
        "valid": {
            "total_samples": valid_total,
            "counts": dict(valid_counts),
            "percentages": valid_pct
        }
    }

    # Save to text file
    with open(OUTPUT_TXT_PATH, "w") as f:
        f.write("CLASS DISTRIBUTION SUMMARY\n")
        f.write("=" * 40 + "\n\n")

        f.write("TRAIN SET\n")
        f.write(f"Total samples: {train_total}\n")
        for cls in sorted(train_counts):
            f.write(
                f"{cls}: {train_counts[cls]} samples "
                f"({train_pct[cls]}%)\n"
            )

        f.write("\nVALID SET\n")
        f.write(f"Total samples: {valid_total}\n")
        for cls in sorted(valid_counts):
            f.write(
                f"{cls}: {valid_counts[cls]} samples "
                f"({valid_pct[cls]}%)\n"
            )

    print("Class distribution computed successfully.")
    print("Results saved to:", OUTPUT_TXT_PATH)


if __name__ == "__main__":
    main()
