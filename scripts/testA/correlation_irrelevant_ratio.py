import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    df = pd.read_csv(path, sep=sep)

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

treemap = load_file("treemap_metrics.csv")
line = load_file("line_metrics.csv", sep="\t")
pie = load_file("pie_metrics.csv")
stackedbar = load_file("stackedbar_metrics.csv")
stackedarea = load_file("stackedarea_metrics.csv", sep="\t")

# ============================================================
# NORMALIZE PARTICIPANTS
# ============================================================

def normalize(p):
    p = str(p).strip()
    if p.startswith("P"):
        return "Participant" + p[1:]
    return p

for df in [treemap, line, pie, stackedbar, stackedarea]:
    df["Participant"] = df.iloc[:, 0].apply(normalize)

# ============================================================
# EXTRACT IRRELEVANT
# ============================================================

def extract_irrelevant(df):
    return df[["Participant", "Irrelevant_Ratio"]].rename(
        columns={"Irrelevant_Ratio": "Irrelevant"}
    )

treemap = extract_irrelevant(treemap)
line = extract_irrelevant(line)
pie = extract_irrelevant(pie)
stackedbar = extract_irrelevant(stackedbar)
stackedarea = extract_irrelevant(stackedarea)

# ============================================================
# COMBINE
# ============================================================

all_data = pd.concat([treemap, line, pie, stackedbar, stackedarea], ignore_index=True)

# ============================================================
# PERFORMANCE
# ============================================================

answers = load_file("answers.csv")
scores = answers.groupby("Participant")["Score"].first().reset_index()

df = all_data.merge(scores, on="Participant")

df = df.dropna()
df["Irrelevant"] = pd.to_numeric(df["Irrelevant"])
df["Score"] = pd.to_numeric(df["Score"])

# ============================================================
# CORRELATION
# ============================================================

corr, p = spearmanr(df["Irrelevant"], df["Score"])

print(f"Spearman r = {corr:.3f}, p = {p:.5f}")

# ============================================================
# FINAL CLEAN SCATTER (NO REGRESSION, NO BINNING)
# ============================================================

plt.figure(figsize=(7, 5))

# Jitter (wichtig für gleiche Optik)
np.random.seed(42)
x_jitter = df["Irrelevant"] + np.random.normal(0, 0.005, size=len(df))

plt.scatter(
    x_jitter,
    df["Score"],
    alpha=0.5,
    s=50
)

# Labels
plt.xlabel("Irrelevant Ratio")
plt.ylabel("Performance (Score)")
plt.title("Irrelevant Ratio vs Performance")

#  IDENTISCHE POSITION wie Transitions
plt.text(
    df["Irrelevant"].min(),
    df["Score"].max() - 0.5,
    f"Spearman r = {corr:.2f}\np = {p:.3f}",
    fontsize=10,
    bbox=dict(facecolor='white', alpha=0.6)
)

# IDENTISCHES GRID
plt.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()