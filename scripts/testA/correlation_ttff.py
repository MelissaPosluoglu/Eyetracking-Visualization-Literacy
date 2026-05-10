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
    path = os.path.join(DATA_PATH, name)
    df = pd.read_csv(path, sep=sep, engine="python")

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

treemap = load_file("treemap_metrics.csv")
line = load_file("line_metrics.csv", sep="\t")
pie = load_file("pie_metrics.csv")
stackedbar = load_file("stackedbar_metrics.csv")
stackedarea = load_file("stackedarea_metrics.csv", sep="\t")

datasets = [treemap, line, pie, stackedbar, stackedarea]

# ============================================================
# NORMALIZE
# ============================================================

for df in datasets:
    df["Participant"] = df.iloc[:, 0].apply(normalize)

# ============================================================
# EXTRACT TTFF (ROBUST)
# ============================================================

def extract_ttff(df):
    ttff_cols = [c for c in df.columns if "TTFF" in c]

    if len(ttff_cols) == 0:
        return None

    col = ttff_cols[0]

    out = df[["Participant", col]].copy()
    out.columns = ["Participant", "TTFF"]

    return out

ttff_list = []

for df in datasets:
    extracted = extract_ttff(df)
    if extracted is not None:
        ttff_list.append(extracted)

# ============================================================
# COMBINE
# ============================================================

all_ttff = pd.concat(ttff_list, ignore_index=True)

# ============================================================
# CLEAN TTFF
# ============================================================

all_ttff["TTFF"] = pd.to_numeric(all_ttff["TTFF"], errors="coerce")

# ungültige entfernen
all_ttff = all_ttff.dropna(subset=["TTFF"])
all_ttff = all_ttff[all_ttff["TTFF"] >= 0]

# optional aber sinnvoll: extremwerte raus
all_ttff = all_ttff[all_ttff["TTFF"] <= 25000]

# ============================================================
# LOAD PERFORMANCE
# ============================================================

answers = load_file("answers.csv")
answers["Participant"] = answers["Participant"].apply(normalize)

scores = answers.groupby("Participant")["Score"].first().reset_index()

# ============================================================
# MERGE
# ============================================================

df = all_ttff.merge(scores, on="Participant")

# ============================================================
# CLEAN FINAL
# ============================================================

df["TTFF"] = pd.to_numeric(df["TTFF"], errors="coerce")
df["Score"] = pd.to_numeric(df["Score"], errors="coerce")

df = df.dropna(subset=["TTFF", "Score"])

# ============================================================
# SPEARMAN CORRELATION
# ============================================================

corr, p = spearmanr(df["TTFF"], df["Score"])

print("\n==============================")
print(" SPEARMAN CORRELATION (TTFF vs PERFORMANCE)")
print("==============================")
print(f"r = {corr:.3f}")
print(f"p = {p:.5f}")

if p < 0.05:
    print("→ SIGNIFICANT correlation")
else:
    print("→ No significant correlation")

# ============================================================
# SCATTERPLOT
# ============================================================

plt.figure(figsize=(7, 5))

# Jitter
np.random.seed(42)
x_jitter = df["TTFF"] + np.random.normal(0, 100, size=len(df))

plt.scatter(
    x_jitter,
    df["Score"],
    alpha=0.5,
    s=50
)

# Labels
plt.xlabel("TTFF (ms)")
plt.ylabel("Performance (Score)")
plt.title("TTFF vs Performance")

# Statistik Box
plt.text(
    df["TTFF"].min(),
    df["Score"].max() - 0.5,
    f"Spearman r = {corr:.2f}\np = {p:.3f}",
    fontsize=10,
    bbox=dict(facecolor='white', alpha=0.6)
)

plt.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()