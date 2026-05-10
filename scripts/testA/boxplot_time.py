import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# CSV laden (relativer Pfad, wenn Script in scripts/testA liegt)
df = pd.read_csv("../../data/testA/answers.csv")

# Seconds in numeric umwandeln
df["Seconds"] = pd.to_numeric(df["Seconds"], errors="coerce")

# Fragen-Reihenfolge festlegen
question_order = [
    "treemap","histogram","100stacked","map","pie",
    "bubble","stackedbar","line","bar","area",
    "stackedarea","scatter"
]

df["Question"] = pd.Categorical(df["Question"], categories=question_order, ordered=True)

median_order = df.groupby("Question")["Seconds"].median().sort_values().index
plt.figure(figsize=(12,6))

sns.boxplot(
    data=df,
    x="Question",
    y="Seconds",
    order=question_order,
    color="#4C72B0",
    width=0.6,
    showfliers=True,
    medianprops=dict(color="red", linewidth=2)   #  Median rot
)

sns.stripplot(
    data=df,
    x="Question",
    y="Seconds",
    order=question_order,
    marker="o",
    edgecolor="black",
    linewidth=1,
    facecolor="none",     #  nicht ausgefüllt
    size=6,
    alpha=1
)

plt.xticks(rotation=45)
plt.xlabel("Visualization")
plt.ylabel("Completion Time (s)")
plt.title("Completion Time per Visualization")

plt.tight_layout()
# ----------------------------
# Results-Ordner relativ erstellen
# ----------------------------
results_path = os.path.join("..", "results")
os.makedirs(results_path, exist_ok=True)

# Dateiname
save_path = os.path.join(results_path, "completion_time_boxplot.png")

# Plot speichern
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.close()

print("Saved to:", save_path)