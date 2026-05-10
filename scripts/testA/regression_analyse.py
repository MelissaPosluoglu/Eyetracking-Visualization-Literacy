import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "testA")
OUTPUT_PATH = os.path.join(BASE_DIR, "results", "testA")

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ============================================================
# LOAD CSV (AUTO DETECT)
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
# CLEANING
# ============================================================

def clean_ttff(series):
    return series.replace(0, np.nan)

def safe(df, col):
    if col not in df.columns:
        raise ValueError(f"\n Column '{col}' not found.\nAvailable:\n{df.columns.tolist()}")
    return df[col]

# ============================================================
# BUILD DATASETS
# ============================================================

# --- Q1 ---
df1 = pd.DataFrame({
    "TTFF": clean_ttff(safe(q1, "TTFF_Search_ms")),
    "Irrelevant": safe(q1, "Irrelevant_Ratio"),
    "Transitions": safe(q1, "Transitions")
})

# --- Q8 ---
df8 = pd.DataFrame({
    "TTFF": clean_ttff(safe(q8, "TTFF Feb")),
    "Irrelevant": safe(q8, "Irrelevant Ratio"),
    "Transitions": safe(q8, "Transitions")
})

# --- Q11 ---
df11 = pd.DataFrame({
    "TTFF": clean_ttff(safe(q11, "TTFF 2012 (s)") * 1000),
    "Irrelevant": safe(q11, "Irrelevant Ratio"),
    "Transitions": safe(q11, "Transitions")
})

# ============================================================
# MERGE
# ============================================================

df_all = pd.concat([df1, df8, df11], ignore_index=True)
df_all = df_all.dropna()

print("\nData points after cleaning:", len(df_all))

# ============================================================
# REGRESSION MODEL
# ============================================================

X = df_all[["TTFF", "Transitions"]]
y = df_all["Irrelevant"]

# Add intercept
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

print("\n==============================")
print(" REGRESSION RESULTS")
print("==============================\n")
print(model.summary())

# Save summary
with open(os.path.join(OUTPUT_PATH, "regression_summary.txt"), "w") as f:
    f.write(model.summary().as_text())

print("\n✔ Regression summary saved")

# ============================================================
# PLOT 1: TTFF vs Irrelevant + Regression Line
# ============================================================

plt.figure()
plt.scatter(df_all["TTFF"], df_all["Irrelevant"])

# Regression line
z = np.polyfit(df_all["TTFF"], df_all["Irrelevant"], 1)
p = np.poly1d(z)
plt.plot(df_all["TTFF"], p(df_all["TTFF"]))

plt.xlabel("TTFF (ms)")
plt.ylabel("Irrelevant Ratio")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "regression_ttff_irrelevant.png"))
plt.close()

# ============================================================
# PLOT 2: Transitions vs Irrelevant
# ============================================================

plt.figure()
plt.scatter(df_all["Transitions"], df_all["Irrelevant"])

z = np.polyfit(df_all["Transitions"], df_all["Irrelevant"], 1)
p = np.poly1d(z)
plt.plot(df_all["Transitions"], p(df_all["Transitions"]))

plt.xlabel("Transitions")
plt.ylabel("Irrelevant Ratio")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "regression_transitions_irrelevant.png"))
plt.close()

print("✔ Plots saved in:", OUTPUT_PATH)

print("\n DONE — Regression ready for paper")