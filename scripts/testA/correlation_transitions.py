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
    df = pd.read_csv(
        os.path.join(DATA_PATH, name),
        sep=sep,
        engine="python"
    )

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
    p = str(p).strip().replace("\ufeff", "")

    if p.startswith("Participant"):
        return p

    if p.startswith("P"):
        return "Participant" + p[1:]

    return p

# ============================================================
# LOAD DATA
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

# Standardize participant IDs in all datasets
for df in datasets:
    df["Participant"] = df.iloc[:, 0].apply(normalize)

# ============================================================
# EXTRACT TRANSITIONS
# ============================================================

def extract_transitions(df):
    """
    Extract the transition count column from a metric dataframe.
    """
    # Search for a column containing transition information
    cols = [c for c in df.columns if "Transitions" in c]

    if len(cols) == 0:
        return None

    col = cols[0]

    # Keep only participant ID and transition count
    out = df[["Participant", col]].copy()
    out.columns = ["Participant", "Transitions"]

    return out


# Collect transition data from all metric files
trans_list = []

for df in datasets:
    extracted = extract_transitions(df)

    if extracted is not None:
        trans_list.append(extracted)

# ============================================================
# COMBINE TRANSITION DATA
# ============================================================

# Combine all transition metrics into one dataframe
all_trans = pd.concat(trans_list, ignore_index=True)

# ============================================================
# CLEAN DATA
# ============================================================

# Convert transitions to numeric values
all_trans["Transitions"] = pd.to_numeric(
    all_trans["Transitions"],
    errors="coerce"
)

# Remove missing or invalid transition values
all_trans = all_trans.dropna(subset=["Transitions"])

# Keep only non-negative transition counts
all_trans = all_trans[all_trans["Transitions"] >= 0]

# ============================================================
# LOAD PERFORMANCE DATA
# ============================================================

# Load answer data and standardize participant IDs
answers = load_file("answers.csv")
answers["Participant"] = answers["Participant"].apply(normalize)

# Extract one performance score per participant
scores = answers.groupby("Participant")["Score"].first().reset_index()

# ============================================================
# MERGE TRANSITIONS WITH PERFORMANCE
# ============================================================

# Merge transition metrics with participant scores
df = all_trans.merge(scores, on="Participant")

# Remove rows with missing values
df = df.dropna(subset=["Transitions", "Score"])

# ============================================================
# SPEARMAN CORRELATION
# ============================================================

# Compute Spearman correlation between transitions and performance
corr, p = spearmanr(df["Transitions"], df["Score"])

print("\n==============================")
print(" SPEARMAN CORRELATION (TRANSITIONS vs PERFORMANCE)")
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

# Add slight jitter to reduce overlap between points
np.random.seed(42)
x_jitter = df["Transitions"] + np.random.normal(
    0,
    0.3,
    size=len(df)
)

# Plot transition count against performance score
plt.scatter(
    x_jitter,
    df["Score"],
    alpha=0.5,
    s=50
)

# Add labels and title
plt.xlabel("Number of Transitions")
plt.ylabel("Performance (Score)")
plt.title("Transitions vs Performance")

# Add correlation statistics to the plot
plt.text(
    df["Transitions"].min(),
    df["Score"].max() - 0.5,
    f"Spearman r = {corr:.2f}\np = {p:.3f}",
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.6)
)

# Add a light grid for readability
plt.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()