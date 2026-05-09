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
# LOAD FUNCTION (ROBUST)
# ============================================================

def load_file(name, sep=None):
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
# LOAD DATA
# ============================================================

treemap = load_file("treemap_metrics.csv")
line = load_file("line_metrics.csv")
pie = load_file("pie_metrics.csv")
stackedbar = load_file("stackedbar_metrics.csv")
stackedarea = load_file("stackedarea_metrics.csv")

answers = load_file("answers.csv")

# ============================================================
# PARTICIPANT NORMALIZATION
# ============================================================

def normalize(p):
    p = str(p).strip().replace("\ufeff", "")

    if p.startswith("Participant"):
        return p

    if p.startswith("P"):
        return "Participant" + p[1:]

    return p

for df in [treemap, line, pie, stackedbar, stackedarea]:
    df["Participant"] = df.iloc[:, 0].apply(normalize)

answers["Participant"] = answers["Participant"].apply(normalize)

# ============================================================
# EXTRACT IRRELEVANT RATIO
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

all_data = pd.concat(
    [treemap, line, pie, stackedbar, stackedarea],
    ignore_index=True
)

# ============================================================
# PERFORMANCE DATA
# ============================================================

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

print("\nData points:", len(df))

# ============================================================
# REGRESSION: SCORE ~ IRRELEVANT
# ============================================================

X = df["Irrelevant"]
y = df["Score"]

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

print("\n==============================")
print(" REGRESSION RESULTS")
print("==============================\n")
print(model.summary())

# Save summary
with open(os.path.join(OUTPUT_PATH, "regression_score_irrelevant.txt"), "w") as f:
    f.write(model.summary().as_text())

# ============================================================
# SCATTER + REGRESSION LINE
# ============================================================

plt.figure()

# jitter für bessere Sichtbarkeit
np.random.seed(42)
x_jitter = df["Irrelevant"] + np.random.normal(0, 0.01, size=len(df))

plt.scatter(x_jitter, df["Score"])

# Regression line
x_vals = np.linspace(df["Irrelevant"].min(), df["Irrelevant"].max(), 100)
x_vals_const = sm.add_constant(x_vals)
y_vals = model.predict(x_vals_const)

plt.plot(x_vals, y_vals)

plt.xlabel("Irrelevant Attention (Ratio)")
plt.ylabel("Performance (Score)")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "regression_score_irrelevant.png"))
plt.close()

print("\n✔ Plot saved")
print("✔ Results saved in:", OUTPUT_PATH)

print("\n🔥 DONE — Regression ready")