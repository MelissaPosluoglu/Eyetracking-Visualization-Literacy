import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
import os
import numpy as np

# ============================================================
# NON-INTERACTIVE BACKEND
# ============================================================

# Use a non-interactive matplotlib backend so the script can run
# without opening a GUI window.
matplotlib.use("Agg")

# ============================================================
# CONFIGURATION
# ============================================================

# Participant and question settings
PARTICIPANT = "Participant17neu"
QUESTION_ID = 5

# Input and output paths
DATA_FILE = os.path.join("..", "..", "data", "testA", f"{PARTICIPANT}.tsv")
IMAGE_PATH = os.path.join("..", "..", "data", "testA", "stimuli", f"Question{QUESTION_ID}.png")
OUTPUT_DIR = os.path.join("..", "..", "results", "testA", PARTICIPANT.lower(), "fixations_clean")

# Create output directory if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fixation filtering thresholds
MIN_FIX_DURATION = 80
MAX_FIX_DURATION = 1000

# Y-threshold used to identify fixations in the top text region
TOP_TEXT_THRESHOLD = 0.28

# Participant-specific correction values for the top text region
TOP_TEXT_SHIFTS = {
    "Participant1": 0.04,
    "Participant4": -0.02,
    "Participant5": -0.04,
    "Participant8": -0.06,
    "Participant10": 0.00,
    "Participant12": -0.08,
    "Participant13": -0.04,
    "Participant14": -0.10,
    "Participant15": -0.07,
    "Participant16": 0.03,
    "Participant20": -0.05,
    "Participant21": 0.00,
}

# ============================================================
# APPLY THE SAME SHIFT LOGIC AS IN THE SCANPATH SCRIPT
# ============================================================

def apply_top_text_shift(fix_df, participant):
    """
    Apply a participant-specific vertical correction to fixations
    located in the top text area of the stimulus.
    """
    fix_df = fix_df.copy()

    # Start with original normalized fixation coordinates
    fix_df["X_shifted"] = fix_df["Fixation point X (MCSnorm)"].copy()
    fix_df["Y_shifted"] = fix_df["Fixation point Y (MCSnorm)"].copy()

    # Look up the vertical shift for the current participant
    top_text_y_shift = TOP_TEXT_SHIFTS.get(participant, 0.00)

    # Apply the shift only to fixations in the upper text area
    mask_top = fix_df["Y_shifted"] < TOP_TEXT_THRESHOLD
    fix_df.loc[mask_top, "Y_shifted"] = (
        fix_df.loc[mask_top, "Y_shifted"] + top_text_y_shift
    )

    # Ensure shifted coordinates remain within normalized bounds
    fix_df["X_shifted"] = fix_df["X_shifted"].clip(0, 1)
    fix_df["Y_shifted"] = fix_df["Y_shifted"].clip(0, 1)

    return fix_df

# ============================================================
# LOAD TSV DATA
# ============================================================

# Load eye-tracking data
df = pd.read_csv(DATA_FILE, sep="\t", low_memory=False)

# ============================================================
# DETECT QUESTION TIME WINDOW
# ============================================================

# Find all question start events
events = df[
    (df["Event"] == "URLStart") &
    (df["Event value"].astype(str).str.contains("Question", na=False))
].sort_values("Recording timestamp")

# Select the event corresponding to the current question
current_event = events[
    events["Event value"].astype(str).str.contains(f"Question {QUESTION_ID}", na=False)
]

if current_event.empty:
    raise RuntimeError(f"Question {QUESTION_ID} not found")

start_time = current_event["Recording timestamp"].iloc[0]

# Use the next question event as the end time, if available
next_events = events[events["Recording timestamp"] > start_time]

if not next_events.empty:
    end_time = next_events["Recording timestamp"].iloc[0]
else:
    end_time = df["Recording timestamp"].max()

# ============================================================
# EXTRACT AND CLEAN FIXATIONS
# ============================================================

# Keep only fixation events within the current question time window
fix = df[
    (df["Eye movement type"] == "Fixation") &
    (df["Recording timestamp"] >= start_time) &
    (df["Recording timestamp"] < end_time)
].copy()

# Remove duplicate fixation entries if an index column is available
if "Eye movement type index" in fix.columns:
    fix = fix.drop_duplicates(subset="Eye movement type index")

# Keep only fixations within the accepted duration range
fix = fix[
    (fix["Gaze event duration"] >= MIN_FIX_DURATION) &
    (fix["Gaze event duration"] <= MAX_FIX_DURATION)
]

# Keep only fixations with valid normalized coordinates
fix = fix[
    (fix["Fixation point X (MCSnorm)"].between(0, 1)) &
    (fix["Fixation point Y (MCSnorm)"].between(0, 1))
].copy()

print(f"Clean fixations for Question {QUESTION_ID}: {len(fix)}")

if fix.empty:
    raise RuntimeError("No valid fixations after cleaning")

# ============================================================
# LOAD STIMULUS IMAGE
# ============================================================

# Load the stimulus image and get its dimensions
img = Image.open(IMAGE_PATH)
w, h = img.size

# ============================================================
# APPLY THE SAME SHIFT LOGIC AS IN THE SCANPATH SCRIPT
# ============================================================

fix = apply_top_text_shift(fix, PARTICIPANT)

# Convert normalized coordinates to pixel coordinates
fix["X_px"] = fix["X_shifted"] * w
fix["Y_px"] = fix["Y_shifted"] * h

# ============================================================
# SCALE FIXATION SIZE BY DURATION
# ============================================================

# Fixation marker size is based on fixation duration
dur = fix["Gaze event duration"].to_numpy()
dur_scaled = np.clip(dur, 80, 600)
size = dur_scaled / 6

# ============================================================
# PLOT FIXATIONS
# ============================================================

# Match figure size to the image dimensions
dpi = 100
fig_w = w / dpi
fig_h = h / dpi

fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

ax.imshow(img)
ax.axis("off")

# Plot fixations on top of the stimulus image
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

# ============================================================
# SAVE OUTPUT
# ============================================================

out_path = os.path.join(
    OUTPUT_DIR,
    f"{PARTICIPANT}_Question{QUESTION_ID}_FIXATIONS_CLEAN_SHIFTED.png"
)

plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

print("Clean fixation plot with the same shift logic as the scanpath was saved.")