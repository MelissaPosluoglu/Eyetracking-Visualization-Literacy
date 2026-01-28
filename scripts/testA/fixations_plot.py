import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
import os
import re

# ----------------------------------------------------
# 1. Matplotlib backend configuration
# ----------------------------------------------------
# The "TkAgg" backend ensures that plots open in a separate window
# when using IDEs such as PyCharm or IntelliJ.
matplotlib.use("TkAgg")

# ----------------------------------------------------
# 2. Configuration
# ----------------------------------------------------
# Change this value to the desired participant (e.g., "participant1.tsv", "participant2.tsv", ...).
DATA_FILE = "../data/eyetracking.tsv"
IMAGE_DIR = "../data/"
# Base directory for all results
RESULTS_BASE_DIR = "../results/testA"

# Change this value to the desired participant (e.g., "Participant1", "Participant2", ...).
PARTICIPANT = "Participant13"

# Output directory for the currently selected participant
OUTPUT_DIR = os.path.join(
    RESULTS_BASE_DIR,
    PARTICIPANT.lower()
)

# Create output directory if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------
# 3. Load eye-tracking TSV file
# ----------------------------------------------------
df = pd.read_csv(DATA_FILE, sep="\t", low_memory=False)

# Convert Tobii recording timestamps (usually in milliseconds) to numeric values
df["Recording timestamp"] = pd.to_numeric(
    df["Recording timestamp"], errors="coerce"
)

# ----------------------------------------------------
# 4. Extract URL Start / URL End events for task segmentation
# ----------------------------------------------------
events = df[
    df["Event"].isin(["URL Start", "URL End"])
][["Participant name", "Event", "Event value", "Recording timestamp"]].copy()

# Keep only events for the selected participant
events = events[events["Participant name"] == PARTICIPANT]

# Extract question number from the event value (e.g., "Question 3 – Treemap")
def extract_question_number(text):
    match = re.search(r"Question\s+(\d+)", str(text))
    return int(match.group(1)) if match else None

events["question_id"] = events["Event value"].apply(extract_question_number)
events = events.dropna(subset=["question_id"])

# Separate URL Start and URL End timestamps
starts = events[events["Event"] == "URL Start"] \
    .rename(columns={"Recording timestamp": "start_time"})

ends = events[events["Event"] == "URL End"] \
    .rename(columns={"Recording timestamp": "end_time"})

# Merge start and end times to obtain time windows per question
questions = starts.merge(
    ends,
    on=["Participant name", "Event value", "question_id"],
    how="inner"
)

# Sort questions by their numeric order
questions = questions.sort_values("question_id")

# ----------------------------------------------------
# 5. Extract fixation events
# ----------------------------------------------------
fix = df[
    (df["Participant name"] == PARTICIPANT) &
    (df["Eye movement type"] == "Fixation")
    ].copy()

# ----------------------------------------------------
# 6. Assign each fixation to a question based on timestamps
# ----------------------------------------------------
def assign_question(row, q_df):
    hit = q_df[
        (row["Recording timestamp"] >= q_df["start_time"]) &
        (row["Recording timestamp"] <= q_df["end_time"])
        ]
    if len(hit) == 1:
        return int(hit.iloc[0]["question_id"])
    return None

fix["question_id"] = fix.apply(assign_question, axis=1, q_df=questions)
fix = fix.dropna(subset=["question_id"])
fix["question_id"] = fix["question_id"].astype(int)

# ----------------------------------------------------
# 7. Generate fixation visualizations for all 12 questions
# ----------------------------------------------------
for q in range(1, 13):
    print(f"▶ Processing Question {q}")

    # Select fixations belonging to the current question
    fix_q = fix[fix["question_id"] == q].copy()

    if fix_q.empty:
        print(f"  ⚠ No fixations found for Question {q}")
        continue

    # Load the corresponding stimulus image
    img_path = os.path.join(IMAGE_DIR, f"Question{q}.PNG")
    if not os.path.exists(img_path):
        print(f"  ❌ Stimulus image missing: Question{q}.PNG")
        continue

    img = Image.open(img_path)
    w, h = img.size

    # Convert normalized fixation coordinates to pixel coordinates
    fix_q["X_px"] = fix_q["Fixation point X (MCSnorm)"] * w
    fix_q["Y_px"] = fix_q["Fixation point Y (MCSnorm)"] * h

    # ------------------------------------------------
    # Plot fixations on top of the stimulus image
    # ------------------------------------------------
    plt.figure(figsize=(6, 6))
    plt.imshow(img, extent=[0, w, 0, h])

    plt.scatter(
        fix_q["X_px"],
        h - fix_q["Y_px"],                 # Invert Y-axis to match image coordinates
        s=fix_q["Gaze event duration"] / 5,  # Scale marker size by fixation duration
        c="red",
        alpha=0.6,
        edgecolors="white"
    )

    plt.title(f"{PARTICIPANT} – Question {q}", fontsize=14)
    plt.xlim(0, w)
    plt.ylim(0, h)
    plt.axis("off")
    plt.tight_layout()

    # Save the fixation plot
    out_path = os.path.join(
        OUTPUT_DIR,
        f"{PARTICIPANT}_Question{q}_Fixations.png"
    )
    plt.savefig(out_path, dpi=200)
    plt.show()

print("✅ Fixation visualization for all 12 questions completed.")
