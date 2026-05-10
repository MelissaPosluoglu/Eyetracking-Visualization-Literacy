import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ============================================================
# PATH SETUP
# ============================================================

# Define base, data, and output paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "testA")
OUTPUT_PATH = os.path.join(BASE_DIR, "results", "testA")

# Create output directory if it does not already exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ============================================================
# LOAD FUNCTION
# ============================================================

def load_file(name, sep=None):
    """
    Load a CSV or TSV file and standardize column names.
    """
    path = os.path.join(DATA_PATH, name)
    df = pd.read_csv(path, sep=sep, engine="python")

    # Clean column names by removing spaces and parentheses
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

# Load AOI metric files for the selected visualization tasks
treemap = load_file("treemap_metrics.csv")
line = load_file("line_metrics.csv")
pie = load_file("pie_metrics.csv")
stackedbar = load_file("stackedbar_metrics.csv")
stackedarea = load_file("stackedarea_metrics.csv")

# Load performance data
answers = load_file("answers.csv")

# ============================================================
# PARTICIPANT ID NORMALIZATION
# ============================================================

def normalize(p):
    """
    Convert participant IDs into a consistent format.

    Examples:
    P1 -> Participant1
    Participant1 -> Participant1
    """
    p = str(p).strip().replace("\ufeff", "")

    if p.startswith("Participant"):
        return p

    if p.startswith("P"):
        return "Participant" + p[1:]

    return p


# Standardize participant IDs in all metric files
for df in [treemap, line, pie, stackedbar, stackedarea]:
    df["Participant"] = df.iloc[:, 0].apply(normalize)

# Standardize participant IDs in the answer file
answers["Participant"] = answers["Participant"].apply(normalize)

# ============================================================
# EXTRACT IRRELEVANT RATIO
# ============================================================

def extract_irrelevant(df):
    """
    Extract the irrelevant attention ratio from one metric dataframe.
    """
    if "Irrelevant_Ratio" not in df.columns:
        raise ValueError(
            f"Irrelevant_Ratio is missing. Available columns: {df.columns.tolist()}"
        )

    return df[["Participant", "Irrelevant_Ratio"]].rename(
        columns={"Irrelevant_Ratio": "Irrelevant"}
    )


# Keep only participant ID and irrelevant-ratio values
treemap = extract_irrelevant(treemap)
line = extract_irrelevant(line)
pie = extract_irrelevant(pie)
stackedbar = extract_irrelevant(stackedbar)
stackedarea = extract_irrelevant(stackedarea)

# ============================================================
# COMBINE ALL TASKS
# ============================================================

# Combine irrelevant-ratio values from all selected tasks
all_data = pd.concat(
    [treemap, line, pie, stackedbar, stackedarea],
    ignore_index=True
)

# ============================================================
# PERFORMANCE DATA
# ============================================================

# Extract one performance score per participant
scores = answers.groupby("Participant")["Score"].first().reset_index()

# ============================================================
# MERGE DATA
# ============================================================

# Merge attention metrics with performance scores
df = all_data.merge(scores, on="Participant")

# ============================================================
# CLEAN DATA
# ============================================================

# Remove missing values before numeric conversion
df = df.dropna(subset=["Irrelevant", "Score"])

# Convert relevant columns to numeric format
df["Irrelevant"] = pd.to_numeric(df["Irrelevant"], errors="coerce")
df["Score"] = pd.to_numeric(df["Score"], errors="coerce")

# Remove rows that could not be converted to numeric values
df = df.dropna()

print("\nData points:", len(df))

# ============================================================
# REGRESSION: SCORE ~ IRRELEVANT
# ============================================================

# Predictor variable
X = df["Irrelevant"]

# Outcome variable
y = df["Score"]

# Add intercept term
X = sm.add_constant(X)

# Fit ordinary least squares regression model
model = sm.OLS(y, X).fit()

print("\n==============================")
print(" REGRESSION RESULTS")
print("==============================\n")
print(model.summary())

# Save regression summary as text file
with open(os.path.join(OUTPUT_PATH, "regression_score_irrelevant.txt"), "w") as f:
    f.write(model.summary().as_text())

# ============================================================
# SCATTER PLOT WITH REGRESSION LINE
# ============================================================

plt.figure()

# Add jitter to improve visibility of overlapping points
np.random.seed(42)
x_jitter = df["Irrelevant"] + np.random.normal(
    0,
    0.01,
    size=len(df)
)

# Plot observed data points
plt.scatter(
    x_jitter,
    df["Score"]
)

# Create regression line
x_vals = np.linspace(
    df["Irrelevant"].min(),
    df["Irrelevant"].max(),
    100
)

x_vals_const = sm.add_constant(x_vals)
y_vals = model.predict(x_vals_const)

plt.plot(
    x_vals,
    y_vals
)

plt.xlabel("Irrelevant Attention (Ratio)")
plt.ylabel("Performance (Score)")

plt.tight_layout()

# Save plot
plt.savefig(
    os.path.join(OUTPUT_PATH, "regression_score_irrelevant.png")
)

plt.close()

print("\n✔ Plot saved")
print("✔ Results saved in:", OUTPUT_PATH)

print("\nDone — regression analysis completed successfully")