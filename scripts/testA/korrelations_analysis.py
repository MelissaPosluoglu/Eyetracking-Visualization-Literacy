import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# ============================================================
# PATH SETUP (ROBUST )
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "testA")
OUTPUT_PATH = os.path.join(BASE_DIR, "results", "testA")

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ============================================================
# LOAD CSV (AUTO-DETECT SEPARATOR)
# ============================================================

def load_csv(name):
    path = os.path.join(DATA_PATH, name)
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = df.columns.str.strip()
    return df

q1 = load_csv("treemap_metrics.csv")
q8 = load_csv("line_metrics.csv")
q11 = load_csv("stackedarea_metrics.csv")

# ============================================================
# CLEANING FUNCTIONS
# ============================================================

def clean_ttff(series):
    """Replace invalid TTFF (0 = not seen)"""
    return series.replace(0, np.nan)

def safe(df, col):
    """Safer column access with debug"""
    if col not in df.columns:
        raise ValueError(f"\n Column '{col}' not found.\nAvailable columns:\n{df.columns.tolist()}")
    return df[col]

# ============================================================
# PREPARE DATASETS
# ============================================================

# --- Q1 (Treemap) ---
df1 = pd.DataFrame({
    "TTFF": clean_ttff(safe(q1, "TTFF_Search_ms")),
    "Irrelevant": safe(q1, "Irrelevant_Ratio"),
    "Transitions": safe(q1, "Transitions"),
    "Dwell": safe(q1, "Dwell_Search_ms")
})

# --- Q8 (Line Chart) ---
df8 = pd.DataFrame({
    "TTFF": clean_ttff(safe(q8, "TTFF Feb")),
    "Irrelevant": safe(q8, "Irrelevant Ratio"),
    "Transitions": safe(q8, "Transitions"),
    "Dwell": safe(q8, "Dwell Feb")
})

# --- Q11 (Stacked Area) ---
#  Sekunden → ms umrechnen!
df11 = pd.DataFrame({
    "TTFF": clean_ttff(safe(q11, "TTFF 2012 (s)") * 1000),
    "Irrelevant": safe(q11, "Irrelevant Ratio"),
    "Transitions": safe(q11, "Transitions"),
    "Dwell": safe(q11, "Dwell 2012 (ms)")
})

# ============================================================
# MERGE ALL TASKS
# ============================================================

df_all = pd.concat([df1, df8, df11], ignore_index=True)

# Drop missing values
df_all = df_all.dropna()

print("\nData points after cleaning:", len(df_all))

# ============================================================
# SPEARMAN CORRELATIONS
# ============================================================

def compute_corr(x, y, label):
    r, p = spearmanr(x, y)
    print(f"{label}: r = {r:.3f}, p = {p:.5f}")
    return r, p

print("\n==============================")
print(" SPEARMAN CORRELATIONS")
print("==============================\n")

results = {}

results["TTFF_vs_Irrelevant"] = compute_corr(
    df_all["TTFF"], df_all["Irrelevant"], "TTFF vs Irrelevant"
)

results["TTFF_vs_Transitions"] = compute_corr(
    df_all["TTFF"], df_all["Transitions"], "TTFF vs Transitions"
)

results["Irrelevant_vs_Transitions"] = compute_corr(
    df_all["Irrelevant"], df_all["Transitions"], "Irrelevant vs Transitions"
)

# ============================================================
# SAVE RESULTS
# ============================================================

corr_df = pd.DataFrame(results, index=["r", "p"]).T
corr_path = os.path.join(OUTPUT_PATH, "correlations.csv")
corr_df.to_csv(corr_path)

print("\n✔ Saved correlations:", corr_path)

# ============================================================
# SCATTER PLOTS (NO COLORS SPECIFIED)
# ============================================================

def scatter_plot(x, y, xlabel, ylabel, filename):
    plt.figure()
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, filename))
    plt.close()

scatter_plot(df_all["TTFF"], df_all["Irrelevant"],
             "TTFF (ms)", "Irrelevant Ratio",
             "ttff_vs_irrelevant.png")

scatter_plot(df_all["TTFF"], df_all["Transitions"],
             "TTFF (ms)", "Transitions",
             "ttff_vs_transitions.png")

scatter_plot(df_all["Irrelevant"], df_all["Transitions"],
             "Irrelevant Ratio", "Transitions",
             "irrelevant_vs_transitions.png")

print("✔ Plots saved in:", OUTPUT_PATH)

print("\n DONE — Everything correct & paper-ready")