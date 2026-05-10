import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
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
    df.columns = df.columns.str.strip()
    return df

# ============================================================
# 🔥 NORMALIZE (WICHTIG!)
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
pie = load_file("pie_metrics.csv")
line = load_file("line_metrics.csv", sep="\t")
stackedbar = load_file("stackedbar_metrics.csv")
stackedarea = load_file("stackedarea_metrics.csv", sep="\t")

datasets = [treemap, pie, line, stackedbar, stackedarea]

# ============================================================
# NORMALIZE PARTICIPANTS
# ============================================================

for df in datasets:
    df["Participant"] = df.iloc[:, 0].apply(normalize)

# ============================================================
# 🔥 EXTRACT IRRELEVANT RATIO (ROBUST)
# ============================================================

def extract_irrelevant(df):
    cols = [c for c in df.columns if "Irrelevant" in c]

    if len(cols) == 0:
        return None

    col = cols[0]

    out = df[["Participant", col]].copy()
    out.columns = ["Participant", "Irrelevant"]

    return out

irr_list = []

for df in datasets:
    extracted = extract_irrelevant(df)
    if extracted is not None:
        irr_list.append(extracted)

# ============================================================
# COMBINE
# ============================================================

all_irr = pd.concat(irr_list, ignore_index=True)

# ============================================================
# CLEAN
# ============================================================

all_irr["Irrelevant"] = pd.to_numeric(all_irr["Irrelevant"], errors="coerce")

all_irr = all_irr.dropna(subset=["Irrelevant"])

# gültiger Bereich [0,1]
all_irr = all_irr[(all_irr["Irrelevant"] >= 0) & (all_irr["Irrelevant"] <= 1)]

# ============================================================
# LOAD ANSWERS
# ============================================================

answers = load_file("answers.csv")
answers["Participant"] = answers["Participant"].apply(normalize)

scores = answers.groupby("Participant")["Score"].first().reset_index()

# ============================================================
# DEBUG
# ============================================================

print("\n=== IRRELEVANT DEBUG ===")
print("Rows:", len(all_irr))
print("Participants:", all_irr["Participant"].nunique())

print("\n=== ANSWERS DEBUG ===")
print("Participants:", scores["Participant"].nunique())

# ============================================================
# INTERSECTION
# ============================================================

common = set(all_irr["Participant"]) & set(scores["Participant"])

print("\n=== INTERSECTION ===")
print("Common participants:", len(common))

if len(common) == 0:
    raise ValueError("❌ KEIN MATCH → IDs falsch!")

# ============================================================
# FILTER + MERGE
# ============================================================

all_irr = all_irr[all_irr["Participant"].isin(common)]
scores = scores[scores["Participant"].isin(common)]

df = all_irr.merge(scores, on="Participant")

df = df.dropna(subset=["Irrelevant", "Score"])

print("\n=== FINAL DATA ===")
print("Rows:", len(df))
print(df.head())

# ============================================================
# GROUPS (MEDIAN SPLIT)
# ============================================================

median = df["Score"].median()

df["Group"] = df["Score"].apply(
    lambda x: "High" if x >= median else "Low"
)

high = df[df["Group"] == "High"]["Irrelevant"]
low = df[df["Group"] == "Low"]["Irrelevant"]

# ============================================================
# BOXPLOT
# ============================================================

plt.figure(figsize=(6, 5))

plt.boxplot([high, low], tick_labels=["High", "Low"])

np.random.seed(42)
plt.scatter(np.random.normal(1, 0.04, len(high)), high, alpha=0.7)
plt.scatter(np.random.normal(2, 0.04, len(low)), low, alpha=0.7)

plt.title("Irrelevant Attention Ratio (All 5 Tasks)")
plt.ylabel("Irrelevant Ratio")
plt.xlabel("Performance Group")

plt.tight_layout()
plt.show()

# ============================================================
# MANN-WHITNEY
# ============================================================

if len(high) >= 2 and len(low) >= 2:
    u, p = mannwhitneyu(high, low, alternative="two-sided")

    print("\n==============================")
    print(" MANN-WHITNEY U TEST")
    print("==============================")
    print(f"U statistic = {u:.3f}")
    print(f"p-value     = {p:.5f}")

    if p < 0.05:
        print("→ SIGNIFICANT difference")
    else:
        print("→ No significant difference")
else:
    print(" Not enough data")