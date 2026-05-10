import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# Load answer data
df = pd.read_csv("../../data/testA/answers.csv")

# Create a participant-by-question matrix with correctness values
matrix = df.pivot(
    index="Participant",
    columns="Question",
    values="Correct"
)

# Define the desired order of visualization/question types
question_order = [
    "treemap", "histogram", "100stacked", "map", "pie",
    "bubble", "stackedbar", "line", "bar", "area",
    "stackedarea", "scatter"
]

# Reorder question columns
matrix = matrix[question_order]

# Sort participants by their numeric participant ID
matrix = matrix.sort_index(
    key=lambda x: x.str.extract(r'(\d+)').astype(int)[0]
)

# Define colors for incorrect and correct answers
colors = ["#D62728", "#4C72B0"]  # Incorrect / Correct
cmap = ListedColormap(colors)

# Create heatmap figure
fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(matrix.values, cmap=cmap, aspect="auto")

# Set x-axis labels as Q1, Q2, ...
ax.set_xticks(np.arange(len(matrix.columns)))
ax.set_xticklabels([f"Q{i + 1}" for i in range(len(matrix.columns))])

# Set y-axis labels as participant IDs
ax.set_yticks(np.arange(len(matrix.index)))
ax.set_yticklabels(matrix.index)

# Add axis labels and title
ax.set_xlabel("Questions")
ax.set_ylabel("Participants")
ax.set_title("Participants' Responses to Questions")

# Add grid lines between cells
ax.set_xticks(np.arange(-0.5, len(matrix.columns), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(matrix.index), 1), minor=True)
ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
ax.tick_params(which="minor", bottom=False, left=False)

# Create legend manually
legend_elements = [
    Patch(facecolor="#4C72B0", label="Correct"),
    Patch(facecolor="#D62728", label="Incorrect")
]

ax.legend(
    handles=legend_elements,
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
    borderaxespad=0,
    frameon=True
)

plt.tight_layout()
plt.subplots_adjust(right=0.8)

# Create results directory if it does not exist
save_dir = "../results"
os.makedirs(save_dir, exist_ok=True)

# Define output filename
save_path = os.path.join(save_dir, "participants_responses.png")

# Save heatmap as PNG
plt.savefig(save_path, dpi=300)
plt.close()

print("Saved to:", save_path)