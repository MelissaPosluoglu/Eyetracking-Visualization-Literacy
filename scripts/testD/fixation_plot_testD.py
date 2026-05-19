import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
import os
import numpy as np

# ----------------------------------------------------
# Non-interactive backend
# ----------------------------------------------------
matplotlib.use("Agg")

# ----------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------

# Select participant and question
PARTICIPANT = "Participant61"
QUESTION_ID = 1

# Define input and output paths
DATA_FILE = os.path.join("..", "..", "data", "testD", f"{PARTICIPANT}.tsv")
IMAGE_PATH = os.path.join("..", "..", "data", "testD", "stimuli", f"Question{QUESTION_ID}.png")
OUTPUT_DIR = os.path.join("..", "..", "results", "testD", PARTICIPANT.lower(), "fixations")

# Create output folder if needed
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fixation duration limits
MIN_FIX_DURATION = 80
MAX_FIX_DURATION = 1000

# Threshold for the upper text area
TOP_TEXT_THRESHOLD = 0.28

# Optional y-shifts for the top text area
TOP_TEXT_SHIFTS = {
    "Participant61": 0.00,
    "Participant62": 0.00,
    "Participant63": 0.00,
    "Participant64": 0.00,
    "Participant65": 0.00,
}

# ----------------------------------------------------
# SHIFT FUNCTION
# ----------------------------------------------------
def apply_top_text_shift(fix_df, participant):
    fix_df = fix_df.copy()

    fix_df["X_shifted"] = fix_df["Fixation point X [MCS norm]"].copy()
    fix_df["Y_shifted"] = fix_df["Fixation point Y [MCS norm]"].copy()

    top_text_y_shift = TOP_TEXT_SHIFTS.get(participant, 0.00)

    mask_top = fix_df["Y_shifted"] < TOP_TEXT_THRESHOLD
    fix_df.loc[mask_top, "Y_shifted"] = (
            fix_df.loc[mask_top, "Y_shifted"] + top_text_y_shift
    )

    fix_df["X_shifted"] = fix_df["X_shifted"].clip(0, 1)
    fix_df["Y_shifted"] = fix_df["Y_shifted"].clip(0, 1)

    return fix_df

# ----------------------------------------------------
# LOAD TSV FILE
# ----------------------------------------------------

# Load participant eye-tracking data
df = pd.read_csv(DATA_FILE, sep="\t", low_memory=False)

# ----------------------------------------------------
# Detect question time window
# ----------------------------------------------------
events = df[
    (df["Event"] == "URLStart") &
    (df["Event value"].astype(str).str.contains("Question", na=False))
    ].sort_values("Recording timestamp [ms]")

# Select the current question event
current_event = events[
    events["Event value"].astype(str).str.contains(f"Question {QUESTION_ID}", na=False)
]

if current_event.empty:
    print(f"Question {QUESTION_ID} not found")
    raise SystemExit

# Start time of the selected question
start_time = current_event["Recording timestamp [ms]"].iloc[0]

next_events = events[events["Recording timestamp [ms]"] > start_time]

if not next_events.empty:
    end_time = next_events["Recording timestamp [ms]"].iloc[0]
else:
    end_time = df["Recording timestamp [ms]"].max()

# ----------------------------------------------------
# EXTRACT AND CLEAN FIXATIONS
# ----------------------------------------------------

# Select fixation rows within the question time window
fix = df[
    (df["Eye movement type"] == "Fixation") &
    (df["Recording timestamp [ms]"] >= start_time) &
    (df["Recording timestamp [ms]"] < end_time)
    ].copy()

# Keep only fixations with realistic durations
fix = fix[
    (fix["Gaze event duration [ms]"] >= MIN_FIX_DURATION) &
    (fix["Gaze event duration [ms]"] <= MAX_FIX_DURATION)
    ]

# Remove duplicate fixation events
if "Eye movement type index" in fix.columns:
    fix = fix.drop_duplicates(subset="Eye movement type index")


# Keep only valid normalized fixation coordinates
fix = fix[
    (fix["Fixation point X [MCS norm]"].between(0, 1)) &
    (fix["Fixation point Y [MCS norm]"].between(0, 1))
    ].copy()

print(f"Clean Fixations for Question {QUESTION_ID}: {len(fix)}")

if fix.empty:
    print("No valid fixations after cleaning")
    raise SystemExit

# ----------------------------------------------------
# LOAD STIMULUS IMAGE
# ----------------------------------------------------

# Load stimulus image and get image size
img = Image.open(IMAGE_PATH)
w, h = img.size

# ----------------------------------------------------
# APPLY COORDINATE SHIFT
# ----------------------------------------------------

# Apply the same top-text shift as in the scanpath script
fix = apply_top_text_shift(fix, PARTICIPANT)

fix["X_px"] = fix["X_shifted"] * w
fix["Y_px"] = fix["Y_shifted"] * h

# ----------------------------------------------------
# SCALE FIXATION SIZE
# ----------------------------------------------------


# Scale circle size based on fixation duration
dur = fix["Gaze event duration [ms]"].to_numpy()
dur_scaled = np.clip(dur, 80, 600)
size = dur_scaled / 6

# ----------------------------------------------------
# Plot
# ----------------------------------------------------
dpi = 100
fig_w = w / dpi
fig_h = h / dpi

fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

ax.imshow(img)
ax.axis("off")

ax.scatter(
    fix["X_px"],
    fix["Y_px"],
    s=size * 1.5,
    color="#cc0000",
    edgecolors="black",
    linewidth=0.6,
    alpha=0.9
)

plt.tight_layout()

# ----------------------------------------------------
# Save
# ----------------------------------------------------
out_path = os.path.join(
    OUTPUT_DIR,
    f"{PARTICIPANT}_Question{QUESTION_ID}_FIXATIONS_CLEAN_SHIFTED.png"
)

plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

print("Clean fixation plot with shifted top-text region saved.")
print("Saved to:", out_path)