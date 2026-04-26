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
# NORMALIZE (WICHTIG!)
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
# 🔥 EXTRACT TRANSITIONS (ROBUST)
# ============================================================

def extract_transitions(df):
    cols = [c for c in df.columns if "Transitions" in c]

    if len(cols) == 0:
        return None

    # nehme die richtige (nicht "Transitions_per_sec")
    col = [c for c in cols if "per" not in c.lower()]
    if len(col) == 0:
        return None

    col = col[0]

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

# unrealistische Werte raus (optional, aber sinnvoll)
all_trans = all_trans[all_trans["Transitions"] >= 0]
all_trans = all_trans[all_trans["Transitions"] <= 200]

# ============================================================
# LOAD ANSWERS
# ============================================================

answers = load_file("answers.csv")
answers["Participant"] = answers["Participant"].apply(normalize)

scores = answers.groupby("Participant")["Score"].first().reset_index()

# ============================================================
# DEBUG
# ============================================================

print("\n=== TRANSITIONS DEBUG ===")
print("Rows:", len(all_trans))
print("Participants:", all_trans["Participant"].nunique())

print("\n=== ANSWERS DEBUG ===")
print("Participants:", scores["Participant"].nunique())

# ============================================================
# INTERSECTION
# ============================================================

common = set(all_trans["Participant"]) & set(scores["Participant"])

print("\n=== INTERSECTION ===")
print("Common participants:", len(common))

if len(common) == 0:
    raise ValueError("❌ KEIN MATCH → IDs falsch!")

# ============================================================
# FILTER + MERGE
# ============================================================

all_trans = all_trans[all_trans["Participant"].isin(common)]
scores = scores[scores["Participant"].isin(common)]

df = all_trans.merge(scores, on="Participant")

df = df.dropna(subset=["Transitions", "Score"])

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

high = df[df["Group"] == "High"]["Transitions"]
low = df[df["Group"] == "Low"]["Transitions"]

# ============================================================
# BOXPLOT
# ============================================================

plt.figure(figsize=(6, 5))

plt.boxplot([high, low], tick_labels=["High", "Low"])

np.random.seed(42)
plt.scatter(np.random.normal(1, 0.04, len(high)), high, alpha=0.7)
plt.scatter(np.random.normal(2, 0.04, len(low)), low, alpha=0.7)

plt.title("Transitions (All Tasks)")
plt.ylabel("Number of Transitions")
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
    print("❌ Not enough data")