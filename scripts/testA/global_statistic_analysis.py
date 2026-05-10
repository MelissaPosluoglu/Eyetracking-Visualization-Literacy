import pandas as pd
import numpy as np
import os

# ============================================================
# PATH SETUP (ROBUST)
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "testA")

# ============================================================
# LOAD DATA (ROBUST)
# ============================================================

def load_csv(name):
    path = os.path.join(DATA_PATH, name)
    return pd.read_csv(path, sep=None, engine="python")

q1 = load_csv("treemap_metrics.csv")
q5 = load_csv("pie_metrics.csv")
q7 = load_csv("stackedbar_metrics.csv")
q8 = load_csv("line_metrics.csv")
q11 = load_csv("stackedarea_metrics.csv")

# ============================================================
# CLEAN FUNCTIONS
# ============================================================

def clean_ttff(series):
    """Remove invalid TTFF values (0 = not seen)"""
    return series.replace(0, np.nan).dropna()

def safe_col(df, name):
    """Fix column issues (spaces, tabs, weird chars)"""
    df.columns = df.columns.str.strip()
    if name not in df.columns:
        raise KeyError(f"Column '{name}' NOT found. Available: {df.columns.tolist()}")
    return df[name]

# ============================================================
# COLLECT RAW DATA (WICHTIGSTER FIX )
# ============================================================

TTFF_all = []
Dwell_rel_all = []
Dwell_ans_all = []
Irrelevant_all = []
Transitions_all = []

def add_task(df, ttff_col, dwell_rel_col, dwell_ans_col, irr_col, trans_col):

    TTFF_all.extend(clean_ttff(safe_col(df, ttff_col)).tolist())
    Dwell_rel_all.extend(safe_col(df, dwell_rel_col).dropna().tolist())
    Dwell_ans_all.extend(safe_col(df, dwell_ans_col).dropna().tolist())
    Irrelevant_all.extend(safe_col(df, irr_col).dropna().tolist())
    Transitions_all.extend(safe_col(df, trans_col).dropna().tolist())

# ============================================================
# APPLY TO ALL TASKS
# ============================================================

# Q1 (Treemap)
add_task(q1,
         "TTFF_Search_ms",
         "Dwell_Search_ms",
         "Dwell_Answers_ms",
         "Irrelevant_Ratio",
         "Transitions")

# Q5 (Pie)
add_task(q5,
         "TTFF Samsung",
         "Dwell Samsung",
         "Dwell Answers",
         "Irrelevant Ratio",
         "Transitions")

# Q7 (Stacked Bar)
add_task(q7,
         "TTFF Seoul",
         "Dwell Seoul",
         "Dwell Answers",
         "Irrelevant Ratio",
         "Transitions")

# Q8 (Line)
add_task(q8,
         "TTFF Feb",
         "Dwell Feb",
         "Dwell Answers",
         "Irrelevant Ratio",
         "Transitions")

# Q11 (Stacked Area → Sekunden → ms)
q11["TTFF_2012_ms"] = safe_col(q11, "TTFF 2012 (s)") * 1000

add_task(q11,
         "TTFF_2012_ms",
         "Dwell 2012 (ms)",
         "Dwell Answers (ms)",
         "Irrelevant Ratio",
         "Transitions")

# ============================================================
# FINAL STATS (JETZT WIRKLICH KORREKT)
# ============================================================

def compute_stats(arr):
    arr = np.array(arr)

    return {
        "Mean": np.mean(arr),
        "Std": np.std(arr),
        "Median": np.median(arr),
        "Min": np.min(arr),
        "Max": np.max(arr)
    }

final_stats = pd.DataFrame({
    "TTFF": compute_stats(TTFF_all),
    "Dwell_rel": compute_stats(Dwell_rel_all),
    "Dwell_ans": compute_stats(Dwell_ans_all),
    "Irrelevant": compute_stats(Irrelevant_all),
    "Transitions": compute_stats(Transitions_all)
})

print("\n==============================")
print(" FINAL OVERALL STATS (CORRECT)")
print("==============================\n")
print(final_stats)

# ============================================================
# OPTIONAL: SAVE
# ============================================================

output_path = os.path.join(DATA_PATH, "final_overall_stats.csv")
final_stats.to_csv(output_path)

print("\n✔ Saved:", output_path)