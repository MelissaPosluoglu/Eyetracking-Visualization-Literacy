import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# SETTINGS
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

participants = {
    "high": ["Participant22", "Participant23", "Participant24"],
    "low": ["Participant4", "Participant5", "Participant20"]
}

QUESTION = 7  # oder loop über mehrere

# =========================
# FUNCTION
# =========================

def extract_metrics(csv_path):
    df = pd.read_csv(csv_path)

    fixations = df["fixation_count"].sum()
    aois = (df["fixation_count"] > 0).sum()
    mean_dwell = df[df["fixation_count"] > 0]["mean_duration"].mean()

    return fixations, aois, mean_dwell


def extract_transitions(path):
    df = pd.read_csv(path, index_col=0)
    return df.values.sum()

# =========================
# COLLECT DATA
# =========================

results = {"high": [], "low": []}

for group, plist in participants.items():
    for p in plist:
        path = PROJECT_ROOT / "results" / "testA" / p / "grid" / f"{p.capitalize()}_Question{QUESTION}_GRID.csv"

        fix, aois, dwell = extract_metrics(path)
        trans_path = PROJECT_ROOT / "results" / "testA" / p / "grid" / f"{p.capitalize()}_Question{QUESTION}_TRANSITIONS.csv"

        trans = extract_transitions(trans_path)

        results[group].append([fix, aois, dwell, trans])

# =========================
# CALCULATE MEAN
# =========================

def calc_mean(data):
    arr = np.array(data)
    return arr.mean(axis=0)

high_mean = calc_mean(results["high"])
low_mean = calc_mean(results["low"])

# =========================
# PRINT TABLE
# =========================

print("\n=== GROUP COMPARISON ===")
print(f"#Fixations       | High: {high_mean[0]:.1f} | Low: {low_mean[0]:.1f}")
print(f"#AOIs visited    | High: {high_mean[1]:.1f} | Low: {low_mean[1]:.1f}")
print(f"Mean Dwell Time  | High: {high_mean[2]:.1f} | Low: {low_mean[2]:.1f}")
print(f"Transitions      | High: {high_mean[3]:.1f} | Low: {low_mean[3]:.1f}")