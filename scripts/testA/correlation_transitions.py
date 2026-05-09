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
# LOAD FUNCTION
# ============================================================

def load_file(name, sep=","):
    df = pd.read_csv(os.path.join(DATA_PATH, name), sep=sep, engine="python")
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("(", "")
        .str.replace(")", "")
    )
    return df

# ============================================================
# NORMALIZE PARTICIPANTS
# ============================================================

def normalize(p):
    p = str(p).strip().replace("\ufeff", "")
    if p.startswith("Participant"):
        return p
    if p.startswith("P"):
        return "Participant" + p[1:]
    return p

# ============================================================
# LOAD DATA
# ============================================================

treemap = load_file("treemap_metrics.csv")
line = load_file("line_metrics.csv", sep="\t")
pie = load_file("pie_metrics.csv")
stackedbar = load_file("stackedbar_metrics.csv")
stackedarea = load_file("stackedarea_metrics.csv", sep="\t")

datasets = [treemap, line, pie, stackedbar, stackedarea]

for df in datasets:
    df["Participant"] = df.iloc[:, 0].apply(normalize)

# ============================================================
# EXTRACT TRANSITIONS
# ============================================================

def extract_transitions(df):
    cols = [c for c in df.columns if "Transitions" in c]

    if len(cols) == 0:
        return None

    col = cols[0]

    out = df[["Participant", col]].copy()
    out.columns = ["Participant", "Transitions"]

    return out

trans_list = []

for df in datasets:
    extracted = extract_transitions(df)
    if extracted is not None:
        trans_list.append(extracted)

# ============================================================
# COMBINE
# ============================================================

all_trans = pd.concat(trans_list, ignore_index=True)

# ============================================================
# CLEAN
# ============================================================

all_trans["Transitions"] = pd.to_numeric(all_trans["Transitions"], errors="coerce")
all_trans = all_trans.dropna(subset=["Transitions"])
all_trans = all_trans[all_trans["Transitions"] >= 0]

# ============================================================
# LOAD ANSWERS
# ============================================================

answers = load_file("answers.csv")
answers["Participant"] = answers["Participant"].apply(normalize)

scores = answers.groupby("Participant")["Score"].first().reset_index()

# ============================================================
# MERGE
# ============================================================

df = all_trans.merge(scores, on="Participant")
df = df.dropna(subset=["Transitions", "Score"])

# ============================================================
# 🔥 SPEARMAN CORRELATION
# ============================================================

corr, p = spearmanr(df["Transitions"], df["Score"])

print("\n==============================")
print(" SPEARMAN CORRELATION (TRANSITIONS vs PERFORMANCE)")
print("==============================")
print(f"r = {corr:.3f}")
print(f"p = {p:.5f}")

if p < 0.05:
    print("→ SIGNIFICANT correlation")
else:
    print("→ No significant correlation")

# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(7, 5))

# leichtes Jitter NUR wenn nötig (Transitions oft überlappen weniger)
np.random.seed(42)
x_jitter = df["Transitions"] + np.random.normal(0, 0.3, size=len(df))

plt.scatter(
    x_jitter,
    df["Score"],
    alpha=0.5,
    s=50
)

# Labels (gleich wie anderer Plot)
plt.xlabel("Number of Transitions")
plt.ylabel("Performance (Score)")
plt.title("Transitions vs Performance")

# Statistik anzeigen (gleiches Format!)
plt.text(
    df["Transitions"].min(),
    df["Score"].max() - 0.5,
    f"Spearman r = {corr:.2f}\np = {p:.3f}",
    fontsize=10,
    bbox=dict(facecolor='white', alpha=0.6)
)

# Cleaner Look
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()