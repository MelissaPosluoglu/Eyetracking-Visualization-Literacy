import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
import os
import re

# ----------------------------------------------------
# 1. Matplotlib backend configuration
# ----------------------------------------------------
matplotlib.use("TkAgg")

# ----------------------------------------------------
# 2. Configuration
# ----------------------------------------------------
# Change this value to the desired participant (e.g., "participant1.tsv", "participant2.tsv", ...).
DATA_FILE = "../data/eyetracking.tsv"
IMAGE_DIR = "../data/"
RESULTS_BASE_DIR = "../results/testA"

# Change this value to the desired participant (e.g., "Participant1", "Participant2", ...).
PARTICIPANT = "Participant13"

# Output directory for the currently selected participant
OUTPUT_DIR = os.path.join(
    RESULTS_BASE_DIR,
    PARTICIPANT.lower(),
    "saccades"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------
# 3. Load eye-tracking TSV file
# ----------------------------------------------------
df = pd.read_csv(DATA_FILE, sep="\t", low_memory=False)

df["Recording timestamp"] = pd.to_numeric(
    df["Recording timestamp"], errors="coerce"
)

# ----------------------------------------------------
# 4. Extract URL Start / URL End events
# ----------------------------------------------------
events = df[
    df["Event"].isin(["URL Start", "URL End"])
][["Participant name", "Event", "Event value", "Recording timestamp"]].copy()

events = events[events["Participant name"] == PARTICIPANT]

def extract_question_number(text):
    match = re.search(r"Question\s+(\d+)", str(text))
    return int(match.group(1)) if match else None

events["question_id"] = events["Event value"].apply(extract_question_number)
events = events.dropna(subset=["question_id"])

starts = events[events["Event"] == "URL Start"] \
    .rename(columns={"Recording timestamp": "start_time"})

ends = events[events["Event"] == "URL End"] \
    .rename(columns={"Recording timestamp": "end_time"})

questions = starts.merge(
    ends,
    on=["Participant name", "Event value", "question_id"],
    how="inner"
).sort_values("question_id")

# ----------------------------------------------------
# 5. Extract saccade events
# ----------------------------------------------------
sacc = df[
    (df["Participant name"] == PARTICIPANT) &
    (df["Eye movement type"] == "Saccade")
    ].copy()

# ----------------------------------------------------
# 6. Assign each saccade to a question
# ----------------------------------------------------
def assign_question(row, q_df):
    hit = q_df[
        (row["Recording timestamp"] >= q_df["start_time"]) &
        (row["Recording timestamp"] <= q_df["end_time"])
        ]
    if len(hit) == 1:
        return int(hit.iloc[0]["question_id"])
    return None

sacc["question_id"] = sacc.apply(assign_question, axis=1, q_df=questions)
sacc = sacc.dropna(subset=["question_id"])
sacc["question_id"] = sacc["question_id"].astype(int)

# ----------------------------------------------------
# 7. Generate saccade visualizations for all 12 questions
# ----------------------------------------------------
for q in range(1, 13):
    print(f"▶ Processing Saccades for Question {q}")

    sacc_q = sacc[sacc["question_id"] == q].copy()

    if sacc_q.empty:
        print(f"  ⚠ No saccades found for Question {q}")
        continue

    img_path = os.path.join(IMAGE_DIR, f"Question{q}.PNG")
    if not os.path.exists(img_path):
        print(f"  ❌ Stimulus image missing: Question{q}.PNG")
        continue

    img = Image.open(img_path)
    w, h = img.size

    # Convert normalized coordinates to pixel space
    sacc_q["X_start_px"] = sacc_q["Saccade start point X (MCSnorm)"] * w
    sacc_q["Y_start_px"] = sacc_q["Saccade start point Y (MCSnorm)"] * h
    sacc_q["X_end_px"] = sacc_q["Saccade end point X (MCSnorm)"] * w
    sacc_q["Y_end_px"] = sacc_q["Saccade end point Y (MCSnorm)"] * h

    # ------------------------------------------------
    # Plot saccades
    # ------------------------------------------------
    plt.figure(figsize=(6, 6))
    plt.imshow(img, extent=[0, w, 0, h])

    for _, row in sacc_q.iterrows():
        plt.plot(
            [row["X_start_px"], row["X_end_px"]],
            [h - row["Y_start_px"], h - row["Y_end_px"]],
            color="blue",
            alpha=0.3,
            linewidth=1
        )

    plt.title(f"{PARTICIPANT} – Question {q} (Saccades)", fontsize=14)
    plt.xlim(0, w)
    plt.ylim(0, h)
    plt.axis("off")
    plt.tight_layout()

    out_path = os.path.join(
        OUTPUT_DIR,
        f"{PARTICIPANT}_Question{q}_Saccades.png"
    )
    plt.savefig(out_path, dpi=200)
    plt.show()

print("✅ Saccade visualization for all 12 questions completed.")
