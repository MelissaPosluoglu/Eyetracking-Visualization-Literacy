import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Datei laden
df = pd.read_csv("../../data/testA/feedback.csv")

questions = [
    "OverallDifficulty",
    "MentalWorkload",
    "StressLevel",
    "TimePressure",
    "PhysicalDemand"
]

# Prozentverteilung berechnen
likert_data = {}

for q in questions:
    counts = df[q].value_counts(normalize=True).sort_index()
    likert_data[q] = counts

likert_df = pd.DataFrame(likert_data).fillna(0)

# Farben für 1–5
colors = ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"]

# Plot
plt.figure(figsize=(10,6))

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

plt.xlabel("Percentage")
plt.gca().xaxis.set_major_formatter(lambda x, _: f"{int(x*100)}%")
plt.title("Cognitive Load – Likert Distribution")
plt.legend(title="Rating (1–5)", bbox_to_anchor=(1.05,1))
plt.tight_layout()

# Speichern im results Ordner (relativ)
results_path = os.path.join("..", "results")
os.makedirs(results_path, exist_ok=True)

save_path = os.path.join(results_path, "cognitive_load_likert.png")
plt.savefig(save_path, dpi=300)


print("Saved to:", save_path)