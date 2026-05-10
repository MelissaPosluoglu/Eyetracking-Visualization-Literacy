import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load the answers CSV file
# The relative path assumes that the script is located in scripts/testA
df = pd.read_csv("../../data/testA/answers.csv")

# Convert the Seconds column to numeric values
# Invalid values are converted to NaN
df["Seconds"] = pd.to_numeric(df["Seconds"], errors="coerce")

# Define the desired order of visualization types
question_order = [
    "treemap", "histogram", "100stacked", "map", "pie",
    "bubble", "stackedbar", "line", "bar", "area",
    "stackedarea", "scatter"
]

# Convert Question into an ordered categorical variable
df["Question"] = pd.Categorical(
    df["Question"],
    categories=question_order,
    ordered=True
)

# Compute median completion times per question
# This can be used later if sorting by median is needed
median_order = df.groupby("Question")["Seconds"].median().sort_values().index

# Create the figure
plt.figure(figsize=(12, 6))

# Draw boxplots for completion time per visualization
sns.boxplot(
    data=df,
    x="Question",
    y="Seconds",
    order=question_order,
    color="#4C72B0",
    width=0.6,
    showfliers=True,
    medianprops=dict(color="red", linewidth=2)
)

# Add individual participant data points on top of the boxplots
sns.stripplot(
    data=df,
    x="Question",
    y="Seconds",
    order=question_order,
    marker="o",
    edgecolor="black",
    linewidth=1,
    facecolor="none",
    size=6,
    alpha=1
)

# Format axis labels and title
plt.xticks(rotation=45)
plt.xlabel("Visualization")
plt.ylabel("Completion Time (s)")
plt.title("Completion Time per Visualization")

plt.tight_layout()

# Create the results folder if it does not already exist
results_path = os.path.join("..", "results")
os.makedirs(results_path, exist_ok=True)

# Define output filename
save_path = os.path.join(results_path, "completion_time_boxplot.png")

# Save plot as PNG
plt.savefig(save_path, dpi=300)
plt.close()

print("Saved to:", save_path)