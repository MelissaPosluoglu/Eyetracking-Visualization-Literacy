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
# NORMALIZE PARTICIPANTS
# ============================================================

def normalize(p):
    p = str(p).strip().replace("\ufeff", "").replace(" ", "")
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
# NORMALIZE IDS
# ============================================================

for df in datasets:
    df["Participant"] = df.iloc[:, 0].apply(normalize)

# ============================================================
# 🔥 EXTRACT CORRECT TTFF (WICHTIG!)
# ============================================================

def extract_ttff(df):
    # bevorzugt "Search" → relevante AOI
    preferred_cols = [c for c in df.columns if "TTFF_Search" in c]

    if len(preferred_cols) > 0:
        col = preferred_cols[0]
    else:
        # fallback: erste TTFF Spalte
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
# 🔥 CLEAN TTFF (JETZT RICHTIG!)
# ============================================================

all_ttff["TTFF"] = pd.to_numeric(all_ttff["TTFF"], errors="coerce")

# ❌ entferne NaN
all_ttff = all_ttff.dropna(subset=["TTFF"])

# ❌ entferne 0 → keine Fixation
all_ttff = all_ttff[all_ttff["TTFF"] > 0]

# ❌ entferne unrealistische Werte (>10 Sekunden)
all_ttff = all_ttff[all_ttff["TTFF"] <= 10000]

# ============================================================
# LOAD ANSWERS
# ============================================================

answers = load_file("answers.csv")
answers["Participant"] = answers["Participant"].apply(normalize)

scores = answers.groupby("Participant")["Score"].first().reset_index()

# ============================================================
# DEBUG
# ============================================================

print("\n=== TTFF DEBUG ===")
print("Rows:", len(all_ttff))
print("Participants:", all_ttff["Participant"].nunique())

print("\n=== ANSWERS DEBUG ===")
print("Participants:", scores["Participant"].nunique())

# ============================================================
# MATCH
# ============================================================

common = set(all_ttff["Participant"]) & set(scores["Participant"])

print("\n=== INTERSECTION ===")
print("Common participants:", len(common))

if len(common) == 0:
    raise ValueError("❌ Kein Match → IDs prüfen!")

# ============================================================
# MERGE
# ============================================================

df = all_ttff.merge(scores, on="Participant")

df = df.dropna(subset=["TTFF", "Score"])

print("\n=== FINAL DATA ===")
print("Rows:", len(df))
print(df.head())

# ============================================================
# GROUPS
# ============================================================

median = df["Score"].median()

df["Group"] = df["Score"].apply(
    lambda x: "High" if x >= median else "Low"
)

high = df[df["Group"] == "High"]["TTFF"]
low = df[df["Group"] == "Low"]["TTFF"]

# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(6, 5))

plt.boxplot([high, low], tick_labels=["High", "Low"])

np.random.seed(42)
plt.scatter(np.random.normal(1, 0.04, len(high)), high, alpha=0.7)
plt.scatter(np.random.normal(2, 0.04, len(low)), low, alpha=0.7)

plt.title("TTFF to Relevant AOI (All 5 Tasks)")
plt.ylabel("TTFF (ms)")
plt.xlabel("Performance Group")

plt.tight_layout()
plt.show()

# ============================================================
# TEST
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