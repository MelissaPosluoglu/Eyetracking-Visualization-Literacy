import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load feedback data
df = pd.read_csv("../../data/testA/feedback.csv")

# NASA-TLX related feedback questions
questions = [
    "OverallDifficulty",
    "MentalWorkload",
    "StressLevel",
    "TimePressure",
    "PhysicalDemand"
]

# Calculate percentage distribution for each Likert-scale question
likert_data = {}

for q in questions:
    counts = df[q].value_counts(normalize=True).sort_index()
    likert_data[q] = counts

# Convert distributions into a dataframe and replace missing values with 0
likert_df = pd.DataFrame(likert_data).fillna(0)

# Colors for Likert ratings 1 to 5
colors = ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"]

# Create horizontal stacked bar chart
plt.figure(figsize=(10, 6))

bottom = np.zeros(len(questions))

for i, rating in enumerate(sorted(likert_df.index)):
    values = likert_df.loc[rating]

    plt.barh(
        questions,
        values,
        left=bottom,
        color=colors[i],
        label=f"{rating}"
    )

    bottom += values

# Format plot labels and title
plt.xlabel("Percentage")
plt.gca().xaxis.set_major_formatter(lambda x, _: f"{int(x * 100)}%")
plt.title("NASA-TLX – Likert Distribution")
plt.legend(title="Rating (1–5)", bbox_to_anchor=(1.05, 1))
plt.tight_layout()

# Create results folder if it does not already exist
results_path = os.path.join("..", "results")
os.makedirs(results_path, exist_ok=True)

# Define output filename
save_path = os.path.join(results_path, "cognitive_load_likert.png")

# Save plot as PNG
plt.savefig(save_path, dpi=300)
plt.close()

print("Saved to:", save_path)