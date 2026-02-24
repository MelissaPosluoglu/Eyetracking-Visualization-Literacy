import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# Datei laden
df = pd.read_csv("../../data/testA/answers.csv")

# Pivot
matrix = df.pivot(
    index="Participant",
    columns="Question",
    values="Correct"
)

question_order = [
    "treemap","histogram","100stacked","map","pie",
    "bubble","stackedbar","line","bar","area",
    "stackedarea","scatter"
]

matrix = matrix[question_order]
matrix = matrix.sort_index()
matrix = matrix.sort_index(
    key=lambda x: x.str.extract(r'(\d+)').astype(int)[0]
)
# Farben
colors = ["#D62728", "#4C72B0"]  # Incorrect / Correct
cmap = ListedColormap(colors)

fig, ax = plt.subplots(figsize=(12,6))
ax.imshow(matrix.values, cmap=cmap, aspect="auto")

ax.set_xticks(np.arange(len(matrix.columns)))
ax.set_xticklabels([f"Q{i+1}" for i in range(len(matrix.columns))])

ax.set_yticks(np.arange(len(matrix.index)))
ax.set_yticklabels(matrix.index)

ax.set_xlabel("Questions")
ax.set_ylabel("Participants")
ax.set_title("Participants' Responses to Questions")

ax.set_xticks(np.arange(-.5, len(matrix.columns), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(matrix.index), 1), minor=True)
ax.grid(which="minor", color="white", linestyle='-', linewidth=1.5)
ax.tick_params(which="minor", bottom=False, left=False)

legend_elements = [
    Patch(facecolor="#4C72B0", label="Correct"),
    Patch(facecolor="#D62728", label="Incorrect")
]

ax.legend(
    handles=legend_elements,
    loc="upper left",
    bbox_to_anchor=(1.02, 1),   # weiter rechts
    borderaxespad=0,
    frameon=True
)

plt.tight_layout()
# -------- RELATIVER SPEICHERPFAD --------
save_dir = "../results"
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, "participants_responses.png")


plt.tight_layout()
plt.subplots_adjust(right=0.8)
plt.savefig(save_path, dpi=300)
plt.close()

print("Saved to:", save_path)