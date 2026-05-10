import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import os

# ============================================================
# PATH
# ============================================================

# Define the base directory and the path to the test data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "testA")

# ============================================================
# LOAD FUNCTION
# ============================================================

def load_file(name, sep=","):
    """
    Load a CSV or TSV file and clean column names.
    """
    df = pd.read_csv(
        os.path.join(DATA_PATH, name),
        sep=sep,
        engine="python"
    )

    # Remove leading and trailing spaces from column names
    df.columns = df.columns.str.strip()

    return df

# ============================================================
# NORMALIZE PARTICIPANT IDS
# ============================================================

def normalize(p):
    """
    Convert participant IDs into a consistent format.

    Examples:
    P1 -> Participant1
    Participant1 -> Participant1
    """
    p = str(p).strip()
    p = p.replace("\ufeff", "")
    p = p.replace(" ", "")

    if p.startswith("Participant"):
        return p

    if p.startswith("P"):
        return "Participant" + p[1:]

    return p

# ============================================================
# LOAD DATASETS
# ============================================================

# Load AOI metric files for the five selected visualization tasks
treemap = load_file("treemap_metrics.csv")
pie = load_file("pie_metrics.csv")
line = load_file("line_metrics.csv", sep="\t")
stackedbar = load_file("stackedbar_metrics.csv")
stackedarea = load_file("stackedarea_metrics.csv", sep="\t")

datasets = [
    treemap,
    pie,
    line,
    stackedbar,
    stackedarea
]

# ============================================================
# NORMALIZE PARTICIPANTS
# ============================================================

# Standardize participant IDs in all metric datasets
for df in datasets:
    df["Participant"] = df.iloc[:, 0].apply(normalize)

# ============================================================
# EXTRACT TRANSITIONS
# ============================================================

def extract_transitions(df):
    """
    Extract the transition count column from a metric dataframe.

    The function avoids using transition-rate columns such as
    'Transitions_per_sec'.
    """
    cols = [c for c in df.columns if "Transitions" in c]

    if len(cols) == 0:
        return None

    # Use the transition count column, not the per-second transition rate
    col = [c for c in cols if "per" not in c.lower()]

    if len(col) == 0:
        return None

    col = col[0]

    # Keep only participant ID and transition count
    out = df[["Participant", col]].copy()
    out.columns = ["Participant", "Transitions"]

    return out


# Collect transition values from all metric datasets
trans_list = []

for df in datasets:
    extracted = extract_transitions(df)

    if extracted is not None:
        trans_list.append(extracted)

# ============================================================
# COMBINE TRANSITION DATA
# ============================================================

# Combine transition values across all selected tasks
all_trans = pd.concat(trans_list, ignore_index=True)

# ============================================================
# CLEAN DATA
# ============================================================

# Convert transition values to numeric format
all_trans["Transitions"] = pd.to_numeric(
    all_trans["Transitions"],
    errors="coerce"
)

# Remove missing values
all_trans = all_trans.dropna(subset=["Transitions"])

# Keep only plausible transition counts
all_trans = all_trans[all_trans["Transitions"] >= 0]
all_trans = all_trans[all_trans["Transitions"] <= 200]

# ============================================================
# LOAD PERFORMANCE DATA
# ============================================================

# Load answer data and standardize participant IDs
answers = load_file("answers.csv")
answers["Participant"] = answers["Participant"].apply(normalize)

# Extract one score per participant
scores = answers.groupby("Participant")["Score"].first().reset_index()

# ============================================================
# DEBUG OUTPUT
# ============================================================

print("\n=== TRANSITIONS DEBUG ===")
print("Rows:", len(all_trans))
print("Participants:", all_trans["Participant"].nunique())

print("\n=== ANSWERS DEBUG ===")
print("Participants:", scores["Participant"].nunique())

# ============================================================
# CHECK PARTICIPANT INTERSECTION
# ============================================================

# Find participants that exist in both metric and answer data
common = set(all_trans["Participant"]) & set(scores["Participant"])

print("\n=== INTERSECTION ===")
print("Common participants:", len(common))

if len(common) == 0:
    raise ValueError("No matching participant IDs found.")

# ============================================================
# FILTER AND MERGE DATA
# ============================================================

# Keep only participants available in both datasets
all_trans = all_trans[all_trans["Participant"].isin(common)]
scores = scores[scores["Participant"].isin(common)]

# Merge transition metrics with performance scores
df = all_trans.merge(scores, on="Participant")

# Remove rows with missing values
df = df.dropna(subset=["Transitions", "Score"])

print("\n=== FINAL DATA ===")
print("Rows:", len(df))
print(df.head())

# ============================================================
# CREATE PERFORMANCE GROUPS
# ============================================================

# Use a median split to classify participants into high and low performers
median = df["Score"].median()

df["Group"] = df["Score"].apply(
    lambda x: "High" if x >= median else "Low"
)

high = df[df["Group"] == "High"]["Transitions"]
low = df[df["Group"] == "Low"]["Transitions"]

# ============================================================
# BOXPLOT
# ============================================================

plt.figure(figsize=(6, 5))

# Plot transition count by performance group
plt.boxplot(
    [high, low],
    tick_labels=["High", "Low"]
)

# Add jittered individual data points
np.random.seed(42)

plt.scatter(
    np.random.normal(1, 0.04, len(high)),
    high,
    alpha=0.7
)

plt.scatter(
    np.random.normal(2, 0.04, len(low)),
    low,
    alpha=0.7
)

plt.title("Transitions (All 5 Tasks)")
plt.ylabel("Number of Transitions")
plt.xlabel("Performance Group")

plt.tight_layout()
plt.show()

# ============================================================
# MANN-WHITNEY U TEST
# ============================================================

# Compare transition counts between high and low performers
if len(high) >= 2 and len(low) >= 2:
    u, p = mannwhitneyu(
        high,
        low,
        alternative="two-sided"
    )

    print("\n==============================")
    print(" MANN-WHITNEY U TEST")
    print("==============================")
    print(f"U statistic = {u:.3f}")
    print(f"p-value     = {p:.5f}")

    if p < 0.05:
        print("→ Significant difference")
    else:
        print("→ No significant difference")
else:
    print("Not enough data")