import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
import os
import numpy as np


# ----------------------------------------------------
# Non-interactive backend
# ----------------------------------------------------

# Use a non-interactive backend so the script can run without opening a GUI window
matplotlib.use("Agg")

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------

# Participant and question settings
PARTICIPANT = "Participant51"
QUESTION_ID = 12

# Input data file
DATA_FILE = os.path.join(
    "..", "..", "data", "testC", f"{PARTICIPANT}.tsv"
)

# Stimulus image for the selected question
IMAGE_PATH = os.path.join(
    "..", "..", "data", "testB", "stimuli", f"Question{QUESTION_ID}.png"
)

# Output directory for fixation plots
OUTPUT_DIR = os.path.join(
    "..", "..", "results", "testB", PARTICIPANT.lower(), "fixations"
)

# Create output directory if it does not already exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------
# Load TSV data
# ----------------------------------------------------

# Load eye-tracking data
df = pd.read_csv(DATA_FILE, sep="\t", low_memory=False)

# ----------------------------------------------------
# Detect question time window
# ----------------------------------------------------

# Find all question start events
events = df[
    (df["Event"] == "URLStart") &
    (df["Event value"].astype(str).str.contains("Question", na=False))
].sort_values("Recording timestamp [ms]")

# Select the start event for the current question
current_event = events[
    events["Event value"].astype(str).str.contains(f"Question {QUESTION_ID}", na=False)
]

if current_event.empty:
    print(f" Question {QUESTION_ID} not found")
    exit()

start_time = current_event["Recording timestamp [ms]"].iloc[0]

# Use the next question start as the end of the current question window
next_events = events[events["Recording timestamp [ms]"] > start_time]

if not next_events.empty:
    end_time = next_events["Recording timestamp [ms]"].iloc[0]
else:
    end_time = df["Recording timestamp [ms]"].max()

# ----------------------------------------------------
# Extract and clean fixations
# ----------------------------------------------------

# Keep only fixation events within the selected question time window
fix = df[
    (df["Eye movement type"] == "Fixation") &
    (df["Recording timestamp [ms]"] >= start_time) &
    (df["Recording timestamp [ms]"] < end_time)
].copy()

# Keep only fixations within a reasonable duration range
fix = fix[
    (fix["Gaze event duration [ms]"] >= 80) &
    (fix["Gaze event duration [ms]"] <= 1000)
]

# Remove duplicate fixation events
fix = fix.drop_duplicates(subset="Eye movement type index")

print(f"Clean fixations for Question {QUESTION_ID}: {len(fix)}")

if fix.empty:
    print("⚠️ No valid fixations after cleaning")
    exit()

# ----------------------------------------------------
# Load stimulus image
# ----------------------------------------------------

# Load the stimulus image and get its dimensions
img = Image.open(IMAGE_PATH)
w, h = img.size

# Convert normalized fixation coordinates to pixel coordinates
fix["X_px"] = fix["Fixation point X [MCS norm]"] * w
fix["Y_px"] = fix["Fixation point Y [MCS norm]"] * h

# ----------------------------------------------------
# Scale fixation size by duration
# ----------------------------------------------------

# Scale marker size according to fixation duration
dur = fix["Gaze event duration [ms]"].to_numpy()
dur_scaled = np.clip(dur, 80, 600)
size = dur_scaled / 6

# ----------------------------------------------------
# Plot fixations
# ----------------------------------------------------

# Match the figure size to the stimulus image dimensions
dpi = 100
fig_w = w / dpi
fig_h = h / dpi

fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

# Show stimulus image
ax.imshow(img)
ax.axis("off")

# Plot fixation points on top of the stimulus
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
# Save output
# ----------------------------------------------------

# Define output file path
out_path = os.path.join(
    OUTPUT_DIR,
    f"{PARTICIPANT}_Question{QUESTION_ID}_FIXATIONS_CLEAN.png"
)

# Save fixation plot
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

print("Clean fixation plot saved.")
print("Saved to:", out_path)