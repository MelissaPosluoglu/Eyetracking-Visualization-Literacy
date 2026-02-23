import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------
# Robust path handling
# ---------------------------------------------------

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_PATH = os.path.join(BASE_DIR, "data", "testA")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "testA")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------
# Load all participant TSV files
# ---------------------------------------------------

import glob

files = glob.glob(os.path.join(DATA_PATH, "Participant*.tsv"))

all_data = []

for file in files:
    participant = os.path.basename(file).replace(".tsv", "")
    df = pd.read_csv(file, sep="\t", low_memory=False)

    # Nur Fixations
    fix = df[df["Eye movement type"] == "Fixation"].copy()

    # Dauer Filter (optional wie bei euch)
    fix = fix[
        (fix["Gaze event duration"] >= 80) &
        (fix["Gaze event duration"] <= 1000)
        ]

    # Question Events finden
    question_events = df[
        (df["Event"].str.contains("URL", na=False)) &
        (df["Event value"].str.contains("Question", na=False))
        ].sort_values("Recording timestamp")

    question_events = question_events.reset_index(drop=True)

    # Segmentieren pro Frage
    for i in range(len(question_events)):
        start = question_events.loc[i, "Recording timestamp"]

        if i < len(question_events) - 1:
            end = question_events.loc[i+1, "Recording timestamp"]
        else:
            end = df["Recording timestamp"].max()

        question_label = f"Q{i+1}"

        question_fix = fix[
            (fix["Recording timestamp"] >= start) &
            (fix["Recording timestamp"] < end)
            ]

        if not question_fix.empty:
            mean_fix = question_fix["Gaze event duration"].mean()

            all_data.append({
                "Participant": participant,
                "Question": question_label,
                "MeanFix": mean_fix
            })

# ---------------------------------------------------
# Create DataFrame
# ---------------------------------------------------

result_df = pd.DataFrame(all_data)

# Reihenfolge Q1–Q12
question_order = [f"Q{i}" for i in range(1, 13)]
result_df["Question"] = pd.Categorical(
    result_df["Question"],
    categories=question_order,
    ordered=True
)

# ---------------------------------------------------
# Plot Boxplot
# ---------------------------------------------------

plt.figure(figsize=(10, 6))

data_to_plot = [
    result_df[result_df["Question"] == q]["MeanFix"]
    for q in question_order
]

box = plt.boxplot(
    data_to_plot,
    labels=question_order,
    patch_artist=True
)

# Styling wie wissenschaftliche Grafik
for patch in box["boxes"]:
    patch.set(facecolor="#6BAED6")

plt.ylabel("Mean Fixation Duration (ms)")
plt.xlabel("Question")
plt.title("Mean Fixation Duration per Question")

plt.tight_layout()

# ---------------------------------------------------
# Save
# ---------------------------------------------------

output_path = os.path.join(
    OUTPUT_DIR,
    "fixation_mean_boxplot_testA.png"
)

plt.savefig(output_path, dpi=300)
plt.close()

print("✅ Fixation Boxplot saved:", output_path)
