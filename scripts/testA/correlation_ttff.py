import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
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
    Load a CSV or TSV file and standardize its column names.
    """
    path = os.path.join(DATA_PATH, name)
    df = pd.read_csv(path, sep=sep, engine="python")

    # Clean column names by removing spaces and parentheses
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("(", "")
        .str.replace(")", "")
    )

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

# Load AOI metric files for the selected visualization types
treemap = load_file("treemap_metrics.csv")
line = load_file("line_metrics.csv", sep="\t")
pie = load_file("pie_metrics.csv")
stackedbar = load_file("stackedbar_metrics.csv")
stackedarea = load_file("stackedarea_metrics.csv", sep="\t")

datasets = [
    treemap,
    line,
    pie,
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
# EXTRACT TTFF
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


# Collect TTFF values from all available datasets
ttff_list = []

for df in datasets:
    extracted = extract_ttff(df)

    if extracted is not None:
        ttff_list.append(extracted)

# ============================================================
# COMBINE TTFF DATA
# ============================================================

# Combine TTFF values across all selected visualization types
all_ttff = pd.concat(ttff_list, ignore_index=True)

# ============================================================
# CLEAN TTFF DATA
# ============================================================

# Convert TTFF values to numeric values
all_ttff["TTFF"] = pd.to_numeric(
    all_ttff["TTFF"],
    errors="coerce"
)

# Remove missing or invalid TTFF values
all_ttff = all_ttff.dropna(subset=["TTFF"])
all_ttff = all_ttff[all_ttff["TTFF"] >= 0]

# Remove extreme TTFF values above 25 seconds
all_ttff = all_ttff[all_ttff["TTFF"] <= 25000]

# ============================================================
# LOAD PERFORMANCE DATA
# ============================================================

# Load answer data and standardize participant IDs
answers = load_file("answers.csv")
answers["Participant"] = answers["Participant"].apply(normalize)

# Extract one performance score per participant
scores = answers.groupby("Participant")["Score"].first().reset_index()

# ============================================================
# MERGE TTFF WITH PERFORMANCE
# ============================================================

# Merge TTFF metrics with participant scores
df = all_ttff.merge(scores, on="Participant")

# ============================================================
# CLEAN FINAL DATA
# ============================================================

# Ensure both variables are numeric
df["TTFF"] = pd.to_numeric(df["TTFF"], errors="coerce")
df["Score"] = pd.to_numeric(df["Score"], errors="coerce")

# Remove rows with missing TTFF or score values
df = df.dropna(subset=["TTFF", "Score"])

# ============================================================
# SPEARMAN CORRELATION
# ============================================================

# Compute Spearman correlation between TTFF and performance
corr, p = spearmanr(df["TTFF"], df["Score"])

print("\n==============================")
print(" SPEARMAN CORRELATION (TTFF vs PERFORMANCE)")
print("==============================")
print(f"r = {corr:.3f}")
print(f"p = {p:.5f}")

if p < 0.05:
    print("→ Significant correlation")
else:
    print("→ No significant correlation")

# ============================================================
# SCATTER PLOT
# ============================================================

plt.figure(figsize=(7, 5))

# Add jitter to reduce overlap between points
np.random.seed(42)
x_jitter = df["TTFF"] + np.random.normal(
    0,
    100,
    size=len(df)
)

# Plot TTFF against performance score
plt.scatter(
    x_jitter,
    df["Score"],
    alpha=0.5,
    s=50
)

# Add labels and title
plt.xlabel("TTFF (ms)")
plt.ylabel("Performance (Score)")
plt.title("TTFF vs Performance")

# Add correlation statistics to the plot
plt.text(
    df["TTFF"].min(),
    df["Score"].max() - 0.5,
    f"Spearman r = {corr:.2f}\np = {p:.3f}",
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.6)
)

# Add a light grid for readability
plt.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()