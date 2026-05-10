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
# EXTRACT IRRELEVANT RATIO
# ============================================================

def extract_irrelevant(df):
    """
    Extract the irrelevant-ratio column from a metric dataframe.

    The function searches for the first column containing the word
    'Irrelevant' to handle slightly different column names.
    """
    cols = [c for c in df.columns if "Irrelevant" in c]

    if len(cols) == 0:
        return None

    col = cols[0]

    # Keep only participant ID and irrelevant-ratio values
    out = df[["Participant", col]].copy()
    out.columns = ["Participant", "Irrelevant"]

    return out


# Collect irrelevant-ratio values from all metric datasets
irr_list = []

for df in datasets:
    extracted = extract_irrelevant(df)

    if extracted is not None:
        irr_list.append(extracted)

# ============================================================
# COMBINE IRRELEVANT-RATIO DATA
# ============================================================

# Combine all extracted irrelevant-ratio values into one dataframe
all_irr = pd.concat(irr_list, ignore_index=True)

# ============================================================
# CLEAN DATA
# ============================================================

# Convert irrelevant-ratio values to numeric format
all_irr["Irrelevant"] = pd.to_numeric(
    all_irr["Irrelevant"],
    errors="coerce"
)

# Remove missing values
all_irr = all_irr.dropna(subset=["Irrelevant"])

# Keep only valid ratio values between 0 and 1
all_irr = all_irr[
    (all_irr["Irrelevant"] >= 0) &
    (all_irr["Irrelevant"] <= 1)
]

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

print("\n=== IRRELEVANT DEBUG ===")
print("Rows:", len(all_irr))
print("Participants:", all_irr["Participant"].nunique())

print("\n=== ANSWERS DEBUG ===")
print("Participants:", scores["Participant"].nunique())

# ============================================================
# CHECK PARTICIPANT INTERSECTION
# ============================================================

# Find participants that exist in both metric and answer data
common = set(all_irr["Participant"]) & set(scores["Participant"])

print("\n=== INTERSECTION ===")
print("Common participants:", len(common))

if len(common) == 0:
    raise ValueError("No matching participant IDs found.")

# ============================================================
# FILTER AND MERGE DATA
# ============================================================

# Keep only participants available in both datasets
all_irr = all_irr[all_irr["Participant"].isin(common)]
scores = scores[scores["Participant"].isin(common)]

# Merge irrelevant-ratio data with performance scores
df = all_irr.merge(scores, on="Participant")

# Remove rows with missing values
df = df.dropna(subset=["Irrelevant", "Score"])

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

high = df[df["Group"] == "High"]["Irrelevant"]
low = df[df["Group"] == "Low"]["Irrelevant"]

# ============================================================
# BOXPLOT
# ============================================================

plt.figure(figsize=(6, 5))

# Plot irrelevant attention ratio by performance group
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

plt.title("Irrelevant Attention Ratio (All 5 Tasks)")
plt.ylabel("Irrelevant Ratio")
plt.xlabel("Performance Group")

plt.tight_layout()
plt.show()

# ============================================================
# MANN-WHITNEY U TEST
# ============================================================

# Compare irrelevant attention ratio between high and low performers
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