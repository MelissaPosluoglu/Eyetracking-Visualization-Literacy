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
# PARTICIPANT ID NORMALIZATION
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

    # Already in the correct format
    if p.startswith("Participant"):
        return p

    # Convert short ID format to full participant ID
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
# NORMALIZE PARTICIPANT IDS
# ============================================================

# Standardize participant IDs in all metric datasets
for df in datasets:
    df["Participant"] = df.iloc[:, 0].apply(normalize)

# ============================================================
# EXTRACT TTFF VALUES
# ============================================================

def extract_ttff(df):
    """
    Extract the first TTFF column from a metric dataframe.

    TTFF means Time To First Fixation.
    """
    ttff_cols = [c for c in df.columns if "TTFF" in c]

    if len(ttff_cols) == 0:
        return None

    col = ttff_cols[0]

    # Keep only participant ID and TTFF values
    out = df[["Participant", col]].copy()
    out.columns = ["Participant", "TTFF"]

    return out


# Collect TTFF values from all metric datasets
ttff_list = []

for df in datasets:
    extracted = extract_ttff(df)

    if extracted is not None:
        ttff_list.append(extracted)

# ============================================================
# COMBINE TTFF DATA
# ============================================================

# Combine all extracted TTFF values into one dataframe
all_ttff = pd.concat(ttff_list, ignore_index=True)

# ============================================================
# CLEAN TTFF DATA
# ============================================================

# Convert TTFF values to numeric format
all_ttff["TTFF"] = pd.to_numeric(
    all_ttff["TTFF"],
    errors="coerce"
)

# Remove missing and invalid values
all_ttff = all_ttff.dropna(subset=["TTFF"])
all_ttff = all_ttff[all_ttff["TTFF"] >= 0]

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

print("\n=== TTFF DEBUG ===")
print("Rows:", len(all_ttff))
print("Participants:", all_ttff["Participant"].nunique())

print("\n=== ANSWERS DEBUG ===")
print("Participants:", scores["Participant"].nunique())

print("\nExamples:")
print("TTFF:", sorted(all_ttff["Participant"].unique())[:5])
print("ANSWERS:", sorted(scores["Participant"].unique())[:5])

# ============================================================
# CHECK PARTICIPANT INTERSECTION
# ============================================================

# Find participants that exist in both metric and answer data
common = set(all_ttff["Participant"]) & set(scores["Participant"])

print("\n=== INTERSECTION ===")
print("Common participants:", len(common))

if len(common) == 0:
    raise ValueError("No matching participant IDs found. Please check the input data.")

# ============================================================
# FILTER AND MERGE DATA
# ============================================================

# Keep only participants available in both datasets
all_ttff = all_ttff[all_ttff["Participant"].isin(common)]
scores = scores[scores["Participant"].isin(common)]

# Merge TTFF values with performance scores
df = all_ttff.merge(scores, on="Participant")

# ============================================================
# FINAL CLEANING
# ============================================================

# Remove rows with missing TTFF or score values
df = df.dropna(subset=["TTFF", "Score"])

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

high = df[df["Group"] == "High"]["TTFF"]
low = df[df["Group"] == "Low"]["TTFF"]

# ============================================================
# BOXPLOT
# ============================================================

plt.figure(figsize=(6, 5))

# Plot TTFF values by performance group
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

plt.title("TTFF to Relevant AOI (All 5 Tasks)")
plt.ylabel("TTFF (ms)")
plt.xlabel("Performance Group")

plt.tight_layout()
plt.show()

# ============================================================
# MANN-WHITNEY U TEST
# ============================================================

# Compare TTFF values between high and low performers
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