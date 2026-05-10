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
# LOAD CSV FILES
# ============================================================

def load_csv(name):
    """
    Load a CSV file while automatically detecting the separator.
    """
    path = os.path.join(DATA_PATH, name)
    df = pd.read_csv(path, sep=None, engine="python")

    # Remove leading and trailing spaces from column names
    df.columns = df.columns.str.strip()

    return df


# Load selected AOI metric files
q1 = load_csv("treemap_metrics.csv")
q8 = load_csv("line_metrics.csv")
q11 = load_csv("stackedarea_metrics.csv")

# ============================================================
# CLEANING FUNCTIONS
# ============================================================

def clean_ttff(series):
    """
    Replace invalid TTFF values.

    A TTFF value of 0 indicates that the relevant AOI was not fixated,
    so it is treated as missing.
    """
    return series.replace(0, np.nan)


def safe(df, col):
    """
    Safely access a dataframe column.

    If the column is missing, an error message with all available columns
    is shown.
    """
    if col not in df.columns:
        raise ValueError(
            f"\nColumn '{col}' was not found.\nAvailable columns:\n{df.columns.tolist()}"
        )

    return df[col]

# ============================================================
# BUILD DATASETS
# ============================================================

# Question 1: Treemap
df1 = pd.DataFrame({
    "TTFF": clean_ttff(safe(q1, "TTFF_Search_ms")),
    "Irrelevant": safe(q1, "Irrelevant_Ratio"),
    "Transitions": safe(q1, "Transitions")
})

# Question 8: Line chart
df8 = pd.DataFrame({
    "TTFF": clean_ttff(safe(q8, "TTFF Feb")),
    "Irrelevant": safe(q8, "Irrelevant Ratio"),
    "Transitions": safe(q8, "Transitions")
})

# Question 11: Stacked area chart
# Convert TTFF from seconds to milliseconds
df11 = pd.DataFrame({
    "TTFF": clean_ttff(safe(q11, "TTFF 2012 (s)") * 1000),
    "Irrelevant": safe(q11, "Irrelevant Ratio"),
    "Transitions": safe(q11, "Transitions")
})

# ============================================================
# MERGE DATA
# ============================================================

# Combine all selected tasks into one dataframe
df_all = pd.concat(
    [df1, df8, df11],
    ignore_index=True
)

# Remove rows with missing values
df_all = df_all.dropna()

print("\nData points after cleaning:", len(df_all))

# ============================================================
# REGRESSION MODEL
# ============================================================

# Predictor variables
X = df_all[["TTFF", "Transitions"]]

# Outcome variable
y = df_all["Irrelevant"]

# Add intercept term to the regression model
X = sm.add_constant(X)

# Fit ordinary least squares regression model
model = sm.OLS(y, X).fit()

print("\n==============================")
print(" REGRESSION RESULTS")
print("==============================\n")
print(model.summary())

# Save regression summary as text file
with open(os.path.join(OUTPUT_PATH, "regression_summary.txt"), "w") as f:
    f.write(model.summary().as_text())

print("\n✔ Regression summary saved")

# ============================================================
# PLOT 1: TTFF VS IRRELEVANT RATIO
# ============================================================

plt.figure()

# Scatter plot
plt.scatter(df_all["TTFF"], df_all["Irrelevant"])

# Add simple linear regression line for visualization
z = np.polyfit(df_all["TTFF"], df_all["Irrelevant"], 1)
p = np.poly1d(z)

plt.plot(
    df_all["TTFF"],
    p(df_all["TTFF"])
)

plt.xlabel("TTFF (ms)")
plt.ylabel("Irrelevant Ratio")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_PATH, "regression_ttff_irrelevant.png")
)

plt.close()

# ============================================================
# PLOT 2: TRANSITIONS VS IRRELEVANT RATIO
# ============================================================

plt.figure()

# Scatter plot
plt.scatter(df_all["Transitions"], df_all["Irrelevant"])

# Add simple linear regression line for visualization
z = np.polyfit(df_all["Transitions"], df_all["Irrelevant"], 1)
p = np.poly1d(z)

plt.plot(
    df_all["Transitions"],
    p(df_all["Transitions"])
)

plt.xlabel("Transitions")
plt.ylabel("Irrelevant Ratio")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_PATH, "regression_transitions_irrelevant.png")
)

plt.close()

print("✔ Plots saved in:", OUTPUT_PATH)

print("\nDone — regression analysis completed successfully")