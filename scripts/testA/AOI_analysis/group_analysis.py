import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# SETTINGS
# =========================

# Define the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Define participant groups
participants = {
    "high": ["Participant22", "Participant23", "Participant24"],
    "low": ["Participant4", "Participant5", "Participant20"]
}

# Question to analyze
QUESTION = 7  # Change this value or loop over multiple questions if needed

# =========================
# FUNCTIONS
# =========================

def extract_metrics(csv_path):
    """
    Extract basic grid-based metrics from a participant's grid CSV file.
    """

    df = pd.read_csv(csv_path)

    # Total number of fixations across all grid cells
    fixations = df["fixation_count"].sum()

    # Number of grid cells that received at least one fixation
    aois = (df["fixation_count"] > 0).sum()

    # Mean dwell time across visited grid cells only
    mean_dwell = df[df["fixation_count"] > 0]["mean_duration"].mean()

    return fixations, aois, mean_dwell


def extract_transitions(path):
    """
    Extract the total number of transitions from a transition matrix CSV.
    """

    df = pd.read_csv(path, index_col=0)

    # Sum all transition counts in the matrix
    return df.values.sum()

# =========================
# COLLECT DATA
# =========================

# Store participant-level metrics for each group
results = {
    "high": [],
    "low": []
}

for group, plist in participants.items():
    for p in plist:

        # Path to the participant's grid summary CSV
        path = (
            PROJECT_ROOT
            / "results"
            / "testA"
            / p
            / "grid"
            / f"{p.capitalize()}_Question{QUESTION}_GRID.csv"
        )

        # Extract fixation, AOI coverage, and dwell metrics
        fix, aois, dwell = extract_metrics(path)

        # Path to the participant's transition matrix CSV
        trans_path = (
            PROJECT_ROOT
            / "results"
            / "testA"
            / p
            / "grid"
            / f"{p.capitalize()}_Question{QUESTION}_TRANSITIONS.csv"
        )

        # Extract total transition count
        trans = extract_transitions(trans_path)

        results[group].append([fix, aois, dwell, trans])

# =========================
# CALCULATE GROUP MEANS
# =========================

def calc_mean(data):
    """
    Calculate the mean value for each metric within a group.
    """

    arr = np.array(data)
    return arr.mean(axis=0)


high_mean = calc_mean(results["high"])
low_mean = calc_mean(results["low"])

# =========================
# PRINT GROUP COMPARISON
# =========================

print("\n=== GROUP COMPARISON ===")
print(f"#Fixations       | High: {high_mean[0]:.1f} | Low: {low_mean[0]:.1f}")
print(f"#AOIs visited    | High: {high_mean[1]:.1f} | Low: {low_mean[1]:.1f}")
print(f"Mean Dwell Time  | High: {high_mean[2]:.1f} | Low: {low_mean[2]:.1f}")
print(f"Transitions      | High: {high_mean[3]:.1f} | Low: {low_mean[3]:.1f}")