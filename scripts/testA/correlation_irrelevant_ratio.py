import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import os

# ============================================================
# PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "testA")

# ============================================================
# LOAD + CLEAN FUNCTION (ROBUST)
# ============================================================

def load_file(name, sep=","):
    path = os.path.join(DATA_PATH, name)
    df = pd.read_csv(path, sep=sep)

    # Spalten säubern
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("(", "")
        .str.replace(")", "")
    )

    return df

# ============================================================
# LOAD ALL TASKS
# ============================================================

treemap = load_file("treemap_metrics.csv")
line = load_file("line_metrics.csv", sep="\t")
pie = load_file("pie_metrics.csv")
stackedbar = load_file("stackedbar_metrics.csv")
stackedarea = load_file("stackedarea_metrics.csv", sep="\t")

# ============================================================
# PARTICIPANT NORMALIZATION
# ============================================================

def normalize(p):
    p = str(p).strip()
    if p.startswith("P"):
        return "Participant" + p[1:]
    return p

for df in [treemap, line, pie, stackedbar, stackedarea]:
    df["Participant"] = df.iloc[:, 0].apply(normalize)

# ============================================================
# EXTRACT IRRELEVANT RATIO (JETZT EINHEITLICH!)
# ============================================================

def extract_irrelevant(df):
    if "Irrelevant_Ratio" not in df.columns:
        raise ValueError(f"Irrelevant_Ratio fehlt in {df.columns}")

    return df[["Participant", "Irrelevant_Ratio"]].rename(
        columns={"Irrelevant_Ratio": "Irrelevant"}
    )

treemap = extract_irrelevant(treemap)
line = extract_irrelevant(line)
pie = extract_irrelevant(pie)
stackedbar = extract_irrelevant(stackedbar)
stackedarea = extract_irrelevant(stackedarea)

# ============================================================
# COMBINE ALL TASKS
# ============================================================

all_data = pd.concat([treemap, line, pie, stackedbar, stackedarea], ignore_index=True)

# ============================================================
# LOAD PERFORMANCE DATA
# ============================================================

answers = load_file("answers.csv")

# Score pro Participant holen
scores = answers.groupby("Participant")["Score"].first().reset_index()

# ============================================================
# MERGE
# ============================================================

df = all_data.merge(scores, on="Participant")

# ============================================================
# CLEAN
# ============================================================

df = df.dropna(subset=["Irrelevant", "Score"])

df["Irrelevant"] = pd.to_numeric(df["Irrelevant"], errors="coerce")
df["Score"] = pd.to_numeric(df["Score"], errors="coerce")

df = df.dropna()

# ============================================================
# 🔥 SPEARMAN CORRELATION (HAUPTERGEBNIS)
# ============================================================

corr, p = spearmanr(df["Irrelevant"], df["Score"])

print("\n==============================")
print(" SPEARMAN CORRELATION (ALL TASKS)")
print("==============================")
print(f"r = {corr:.3f}")
print(f"p = {p:.5f}")

if p < 0.05:
    print("→ SIGNIFICANT correlation")
else:
    print("→ No significant correlation")

# ============================================================
# 🔥 SCATTERPLOT (SAUBER)
# ============================================================

plt.figure(figsize=(6, 5))

# leichtes jitter für bessere Sichtbarkeit
np.random.seed(42)
x_jitter = df["Irrelevant"] + np.random.normal(0, 0.01, size=len(df))

plt.scatter(x_jitter, df["Score"], alpha=0.6)

plt.xlabel("Irrelevant Attention (Ratio)")
plt.ylabel("Performance (Score)")
plt.title("Irrelevant Attention vs Performance (All Tasks)")

plt.tight_layout()
plt.show()