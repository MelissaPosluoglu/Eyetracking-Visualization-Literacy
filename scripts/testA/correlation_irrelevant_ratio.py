import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import os

# ============================================================
# PATH
# ============================================================

# Define the base directory and data path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "testA")

# ============================================================
# LOAD FUNCTION
# ============================================================

def load_file(name, sep=","):
    """
    Load a CSV or TSV file and clean the column names.
    """
    path = os.path.join(DATA_PATH, name)
    df = pd.read_csv(path, sep=sep)

    # Standardize column names by removing spaces and parentheses
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("(", "")
        .str.replace(")", "")
    )

    return df

# ============================================================
# LOAD DATA
# ============================================================

# Load AOI metric files for selected visualization types
treemap = load_file("treemap_metrics.csv")
line = load_file("line_metrics.csv", sep="\t")
pie = load_file("pie_metrics.csv")
stackedbar = load_file("stackedbar_metrics.csv")
stackedarea = load_file("stackedarea_metrics.csv", sep="\t")

# ============================================================
# NORMALIZE PARTICIPANT IDS
# ============================================================

def normalize(p):
    """
    Convert participant IDs into a consistent format.

    Example:
    P1 -> Participant1
    Participant1 -> Participant1
    """
    p = str(p).strip()

    if p.startswith("P"):
        return "Participant" + p[1:]

    return p


# Apply participant ID normalization to all metric dataframes
for df in [treemap, line, pie, stackedbar, stackedarea]:
    df["Participant"] = df.iloc[:, 0].apply(normalize)

# ============================================================
# EXTRACT IRRELEVANT RATIO
# ============================================================

def extract_irrelevant(df):
    """
    Keep only participant ID and irrelevant-ratio values.
    """
    return df[["Participant", "Irrelevant_Ratio"]].rename(
        columns={"Irrelevant_Ratio": "Irrelevant"}
    )


treemap = extract_irrelevant(treemap)
line = extract_irrelevant(line)
pie = extract_irrelevant(pie)
stackedbar = extract_irrelevant(stackedbar)
stackedarea = extract_irrelevant(stackedarea)

# ============================================================
# COMBINE AOI METRICS
# ============================================================

# Combine all selected visualization types into one dataframe
all_data = pd.concat(
    [treemap, line, pie, stackedbar, stackedarea],
    ignore_index=True
)

# ============================================================
# LOAD PERFORMANCE DATA
# ============================================================

# Load answer data and extract one score per participant
answers = load_file("answers.csv")
scores = answers.groupby("Participant")["Score"].first().reset_index()

# Merge irrelevant-ratio metrics with participant performance scores
df = all_data.merge(scores, on="Participant")

# Clean and convert relevant columns to numeric values
df = df.dropna()
df["Irrelevant"] = pd.to_numeric(df["Irrelevant"])
df["Score"] = pd.to_numeric(df["Score"])

# ============================================================
# CORRELATION
# ============================================================

# Compute Spearman correlation between irrelevant viewing and performance
corr, p = spearmanr(df["Irrelevant"], df["Score"])

print(f"Spearman r = {corr:.3f}, p = {p:.5f}")

# ============================================================
# SCATTER PLOT
# ============================================================

# Create scatter plot without regression line or binning
plt.figure(figsize=(7, 5))

# Add small jitter to avoid overlapping points
np.random.seed(42)
x_jitter = df["Irrelevant"] + np.random.normal(
    0,
    0.005,
    size=len(df)
)

plt.scatter(
    x_jitter,
    df["Score"],
    alpha=0.5,
    s=50
)

# Add axis labels and title
plt.xlabel("Irrelevant Ratio")
plt.ylabel("Performance (Score)")
plt.title("Irrelevant Ratio vs Performance")

# Add correlation statistics inside the plot
plt.text(
    df["Irrelevant"].min(),
    df["Score"].max() - 0.5,
    f"Spearman r = {corr:.2f}\np = {p:.3f}",
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.6)
)

# Add a light grid for readability
plt.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()