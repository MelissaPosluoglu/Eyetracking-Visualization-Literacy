import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import os

# ============================================================
# PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "testA")

# ============================================================
# LOAD DATA
# ============================================================

def load_csv(name):
    path = os.path.join(DATA_PATH, name)
    return pd.read_csv(path, sep=None, engine="python")

q11 = load_csv("stackedarea_metrics.csv")

# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

q11.columns = q11.columns.str.strip()

# ============================================================
# PARTICIPANT NAMING FIX
# ============================================================

# Einheitliche IDs erzeugen (P2 → Participant2)
def normalize_participant(p):
    p = str(p).strip()
    if p.startswith("P"):
        return "Participant" + p[1:]
    return p

q11["Participant"] = q11["P"].apply(normalize_participant)

# ============================================================
# 👉 HIER DEINE ACCURACY DEFINIEREN
# ============================================================

# ⚠️ DAS MUSST DU ANPASSEN falls du echte Scores hast!
# Beispiel (Dummy – bitte ersetzen mit deinen echten Daten)

accuracy_dict = {
    "Participant2": 1,
    "Participant3": 0,
    "Participant4": 1,
    "Participant5": 0,
    "Participant6": 0,
    "Participant7": 1,
    "Participant8": 1,
    "Participant9": 0,
    "Participant13": 0,
    "Participant14": 1,
    "Participant16": 1,
    "Participant18": 0,
    "Participant19": 0,
    "Participant20": 1,
    "Participant21": 1,
    "Participant22": 1,
    "Participant23": 1,
    "Participant24": 1,
    "Participant25": 1,
    "Participant26": 0,
    "Participant27": 0
}

q11["Accuracy"] = q11["Participant"].map(accuracy_dict)

# ============================================================
# GROUPS (HIGH vs LOW)
# ============================================================

q11["Group"] = q11["Accuracy"].apply(lambda x: "High" if x == 1 else "Low")

# ============================================================
# METRICS
# ============================================================

irr = q11["Irrelevant Ratio"]
acc = q11["Accuracy"]

# ============================================================
# 🔥 1. BOXPLOT (WICHTIGSTER PLOT)
# ============================================================

plt.figure(figsize=(6, 5))
sns.boxplot(data=q11, x="Group", y="Irrelevant Ratio")
sns.stripplot(data=q11, x="Group", y="Irrelevant Ratio", color="black", alpha=0.6)

plt.title("Irrelevant Attention by Performance Group (Q11)")
plt.ylabel("Irrelevant Ratio")
plt.xlabel("Performance Group")

plt.tight_layout()
plt.show()

# ============================================================
# 🔥 2. KORRELATION (SPEARMAN)
# ============================================================

corr, p = spearmanr(irr, acc)

print("\n=== SPEARMAN CORRELATION ===")
print(f"r = {corr:.3f}")
print(f"p = {p:.5f}")

# ============================================================
# 🔥 3. SCATTERPLOT
# ============================================================

plt.figure(figsize=(6, 5))
sns.regplot(x=irr, y=acc)

plt.title("Irrelevant Attention vs Performance (Q11)")
plt.xlabel("Irrelevant Ratio")
plt.ylabel("Accuracy")

plt.tight_layout()
plt.show()