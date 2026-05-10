import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------
# Robust path handling
# ---------------------------------------------------

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_PATH = os.path.join(BASE_DIR, "data", "testA", "answers.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "testA")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------
# Load Data
# ---------------------------------------------------

df = pd.read_csv(DATA_PATH)

# ---------------------------------------------------
# Ensure correct question order
# ---------------------------------------------------

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

df["Question"] = pd.Categorical(
    df["Question"],
    categories=question_order,
    ordered=True
)

# ---------------------------------------------------
# Aggregate responses
# ---------------------------------------------------

total_participants = df["Participant"].nunique()

summary = (
    df.groupby("Question")["Correct"]
    .value_counts()
    .unstack(fill_value=0)
)

# Reihenfolge explizit festlegen (wichtig!)
summary = summary.reindex(question_order)

correct_counts = summary.get(1, 0)
incorrect_counts = summary.get(0, 0)

# ---------------------------------------------------
# Plot
# ---------------------------------------------------

plt.figure(figsize=(10, 7))

plt.barh(
    summary.index,
    correct_counts,
    color="#6BAED6",
    label="Correct"
)

plt.barh(
    summary.index,
    incorrect_counts,
    left=correct_counts,
    color="#D62728",
    label="Incorrect"
)

plt.xlabel("Number of Participants")
plt.title("Question Responses – Test A")
plt.xlim(0, total_participants)
plt.legend()
plt.tight_layout()

# ---------------------------------------------------
# Save
# ---------------------------------------------------

output_path = os.path.join(
    OUTPUT_DIR,
    "question_responses_testA.png"
)

plt.savefig(output_path, dpi=300)
plt.close()

print("Plot saved:", output_path)
