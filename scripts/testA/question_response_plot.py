import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------
# Robust path handling
# ---------------------------------------------------

# Define base, data, and output paths
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_PATH = os.path.join(BASE_DIR, "data", "testA", "answers.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "testA")

# Create output directory if it does not already exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------
# Load data
# ---------------------------------------------------

# Load answer data
df = pd.read_csv(DATA_PATH)

# ---------------------------------------------------
# Ensure correct question order
# ---------------------------------------------------

# Define the intended order of visualization/question types
question_order = [
    "treemap",
    "histogram",
    "100stacked",
    "map",
    "pie",
    "bubble",
    "stackedbar",
    "line",
    "bar",
    "area",
    "stackedarea",
    "scatter"
]

# Convert Question into an ordered categorical variable
df["Question"] = pd.Categorical(
    df["Question"],
    categories=question_order,
    ordered=True
)

# ---------------------------------------------------
# Aggregate responses
# ---------------------------------------------------

# Count unique participants
total_participants = df["Participant"].nunique()

# Count correct and incorrect responses per question
summary = (
    df.groupby("Question")["Correct"]
    .value_counts()
    .unstack(fill_value=0)
)

# Reindex explicitly to keep the defined question order
summary = summary.reindex(question_order)

# Extract correct and incorrect counts
correct_counts = summary.get(1, 0)
incorrect_counts = summary.get(0, 0)

# ---------------------------------------------------
# Plot stacked horizontal bar chart
# ---------------------------------------------------

plt.figure(figsize=(10, 7))

# Plot correct answers
plt.barh(
    summary.index,
    correct_counts,
    color="#6BAED6",
    label="Correct"
)

# Plot incorrect answers stacked after correct answers
plt.barh(
    summary.index,
    incorrect_counts,
    left=correct_counts,
    color="#D62728",
    label="Incorrect"
)

# Format plot
plt.xlabel("Number of Participants")
plt.title("Question Responses – Test A")
plt.xlim(0, total_participants)
plt.legend()
plt.tight_layout()

# ---------------------------------------------------
# Save plot
# ---------------------------------------------------

# Define output filename
output_path = os.path.join(
    OUTPUT_DIR,
    "question_responses_testA.png"
)

# Save figure as high-resolution PNG
plt.savefig(output_path, dpi=300)
plt.close()

print("Plot saved:", output_path)