import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# ---------------------------------------------------
# Robust path handling
# ---------------------------------------------------

# Define base, data, and output directories
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_PATH = os.path.join(BASE_DIR, "data", "testA")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "testA")

# Create output directory if it does not already exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------
# Load all participant TSV files
# ---------------------------------------------------

# Find all participant eye-tracking files
files = glob.glob(os.path.join(DATA_PATH, "Participant*.tsv"))

# Store mean fixation duration per participant and question
all_data = []

for file in files:
    participant = os.path.basename(file).replace(".tsv", "")

    # Load participant eye-tracking data
    df = pd.read_csv(file, sep="\t", low_memory=False)

    # Keep only fixation events
    fix = df[df["Eye movement type"] == "Fixation"].copy()

    # Keep only fixations within a reasonable duration range
    fix = fix[
        (fix["Gaze event duration"] >= 80) &
        (fix["Gaze event duration"] <= 1000)
    ]

    # Find question-related URL events
    question_events = df[
        (df["Event"].str.contains("URL", na=False)) &
        (df["Event value"].str.contains("Question", na=False))
    ].sort_values("Recording timestamp")

    question_events = question_events.reset_index(drop=True)

    # Segment fixation data by question
    for i in range(len(question_events)):
        start = question_events.loc[i, "Recording timestamp"]

        # Use the next question event as the end of the current question window
        if i < len(question_events) - 1:
            end = question_events.loc[i + 1, "Recording timestamp"]
        else:
            end = df["Recording timestamp"].max()

        question_label = f"Q{i + 1}"

        # Select fixations within the current question window
        question_fix = fix[
            (fix["Recording timestamp"] >= start) &
            (fix["Recording timestamp"] < end)
        ]

        # Compute mean fixation duration for the current question
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

# Define the intended question order from Q1 to Q12
question_order = [f"Q{i}" for i in range(1, 13)]

result_df["Question"] = pd.Categorical(
    result_df["Question"],
    categories=question_order,
    ordered=True
)

# ---------------------------------------------------
# Plot boxplot
# ---------------------------------------------------

plt.figure(figsize=(10, 6))

# Prepare data in the correct question order
data_to_plot = [
    result_df[result_df["Question"] == q]["MeanFix"]
    for q in question_order
]

box = plt.boxplot(
    data_to_plot,
    labels=question_order,
    patch_artist=True
)

# Apply simple scientific-style formatting
for patch in box["boxes"]:
    patch.set(facecolor="#6BAED6")

plt.ylabel("Mean Fixation Duration (ms)")
plt.xlabel("Question")
plt.title("Mean Fixation Duration per Question")

plt.tight_layout()

# ---------------------------------------------------
# Save plot
# ---------------------------------------------------

output_path = os.path.join(
    OUTPUT_DIR,
    "fixation_mean_boxplot_testA.png"
)

# Save figure as high-resolution PNG
plt.savefig(output_path, dpi=300)
plt.close()

print("Fixation boxplot saved:", output_path)