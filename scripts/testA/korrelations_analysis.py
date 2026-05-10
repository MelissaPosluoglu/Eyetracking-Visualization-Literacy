import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# ============================================================
# PATH SETUP
# ============================================================

# Define base, data, and output paths in a robust way
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "testA")
OUTPUT_PATH = os.path.join(BASE_DIR, "results", "testA")

# Create output directory if it does not already exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ============================================================
# LOAD CSV FILES
# ============================================================

def load_csv(name):
    """
    Load a CSV file while automatically detecting the separator.
    """
    path = os.path.join(DATA_PATH, name)
    df = pd.read_csv(path, sep=None, engine="python")

    # Remove leading and trailing spaces from column names
    df.columns = df.columns.str.strip()

    return df


# Load metric files for the selected visualization tasks
q1 = load_csv("treemap_metrics.csv")
q8 = load_csv("line_metrics.csv")
q11 = load_csv("stackedarea_metrics.csv")

# ============================================================
# CLEANING FUNCTIONS
# ============================================================

def clean_ttff(series):
    """
    Replace invalid TTFF values.

    A TTFF value of 0 indicates that the target AOI was not fixated,
    so it is treated as missing.
    """
    return series.replace(0, np.nan)


def safe(df, col):
    """
    Safely access a dataframe column.

    If the column is missing, a helpful error message with all available
    column names is shown.
    """
    if col not in df.columns:
        raise ValueError(
            f"\nColumn '{col}' was not found.\nAvailable columns:\n{df.columns.tolist()}"
        )

    return df[col]

# ============================================================
# PREPARE DATASETS
# ============================================================

# Question 1: Treemap
df1 = pd.DataFrame({
    "TTFF": clean_ttff(safe(q1, "TTFF_Search_ms")),
    "Irrelevant": safe(q1, "Irrelevant_Ratio"),
    "Transitions": safe(q1, "Transitions"),
    "Dwell": safe(q1, "Dwell_Search_ms")
})

# Question 8: Line chart
df8 = pd.DataFrame({
    "TTFF": clean_ttff(safe(q8, "TTFF Feb")),
    "Irrelevant": safe(q8, "Irrelevant Ratio"),
    "Transitions": safe(q8, "Transitions"),
    "Dwell": safe(q8, "Dwell Feb")
})

# Question 11: Stacked area chart
# Convert TTFF from seconds to milliseconds
df11 = pd.DataFrame({
    "TTFF": clean_ttff(safe(q11, "TTFF 2012 (s)") * 1000),
    "Irrelevant": safe(q11, "Irrelevant Ratio"),
    "Transitions": safe(q11, "Transitions"),
    "Dwell": safe(q11, "Dwell 2012 (ms)")
})

# ============================================================
# MERGE ALL TASKS
# ============================================================

# Combine all selected task data into one dataframe
df_all = pd.concat(
    [df1, df8, df11],
    ignore_index=True
)

# Remove rows with missing values
df_all = df_all.dropna()

print("\nData points after cleaning:", len(df_all))

# ============================================================
# SPEARMAN CORRELATIONS
# ============================================================

def compute_corr(x, y, label):
    """
    Compute and print Spearman correlation for two variables.
    """
    r, p = spearmanr(x, y)

    print(f"{label}: r = {r:.3f}, p = {p:.5f}")

    return r, p


print("\n==============================")
print(" SPEARMAN CORRELATIONS")
print("==============================\n")

results = {}

results["TTFF_vs_Irrelevant"] = compute_corr(
    df_all["TTFF"],
    df_all["Irrelevant"],
    "TTFF vs Irrelevant"
)

results["TTFF_vs_Transitions"] = compute_corr(
    df_all["TTFF"],
    df_all["Transitions"],
    "TTFF vs Transitions"
)

results["Irrelevant_vs_Transitions"] = compute_corr(
    df_all["Irrelevant"],
    df_all["Transitions"],
    "Irrelevant vs Transitions"
)

# ============================================================
# SAVE CORRELATION RESULTS
# ============================================================

# Convert correlation results into a dataframe
corr_df = pd.DataFrame(results, index=["r", "p"]).T

# Save correlations as CSV
corr_path = os.path.join(OUTPUT_PATH, "correlations.csv")
corr_df.to_csv(corr_path)

print("\n✔ Saved correlations:", corr_path)

# ============================================================
# SCATTER PLOTS
# ============================================================

def scatter_plot(x, y, xlabel, ylabel, filename):
    """
    Create and save a simple scatter plot.
    """
    plt.figure()

    plt.scatter(x, y)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.tight_layout()

    save_path = os.path.join(OUTPUT_PATH, filename)
    plt.savefig(save_path)
    plt.close()


# Plot TTFF against irrelevant viewing
scatter_plot(
    df_all["TTFF"],
    df_all["Irrelevant"],
    "TTFF (ms)",
    "Irrelevant Ratio",
    "ttff_vs_irrelevant.png"
)

# Plot TTFF against transition count
scatter_plot(
    df_all["TTFF"],
    df_all["Transitions"],
    "TTFF (ms)",
    "Transitions",
    "ttff_vs_transitions.png"
)

# Plot irrelevant viewing against transition count
scatter_plot(
    df_all["Irrelevant"],
    df_all["Transitions"],
    "Irrelevant Ratio",
    "Transitions",
    "irrelevant_vs_transitions.png"
)

print("✔ Plots saved in:", OUTPUT_PATH)

print("\nDone — correlation analysis completed successfully")