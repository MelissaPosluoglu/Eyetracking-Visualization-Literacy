import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
import os
import numpy as np

# This script is intended for Participants 1 to 20

# ----------------------------------------------------
# Non-interactive backend
# ----------------------------------------------------

# Use a non-interactive backend so the script can run without opening a GUI window
matplotlib.use("Agg")

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------

# Participant and question settings
PARTICIPANT = "Participant5"
QUESTION_ID = 1

# Input data file
DATA_FILE = os.path.join(
    "..", "..", "data", "testA", f"{PARTICIPANT}.tsv"
)

# Stimulus image for the selected question
IMAGE_PATH = os.path.join(
    "..", "..", "data", "testA", "stimuli", f"Question{QUESTION_ID}.png"
)

# Output directory for saccade visualizations
OUTPUT_DIR = os.path.join(
    "..", "..", "results", "testA",
    PARTICIPANT.lower(), "saccades"
)

# Create output directory if it does not already exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------
# Parameters
# ----------------------------------------------------

# Minimum saccade length used for analysis
# This removes very small movements that are likely measurement noise
ANALYSIS_MIN_PX = 20

# Minimum saccade length shown in the plot
# This improves visual readability but does not affect the analysis statistics
VISUAL_MIN_PX = 60

# Plot appearance settings
ALPHA = 0.25
LINEWIDTH = 1.0

# ----------------------------------------------------
# Load data
# ----------------------------------------------------

# Load eye-tracking data
df = pd.read_csv(DATA_FILE, sep="\t", low_memory=False)

# ----------------------------------------------------
# Determine question time window
# ----------------------------------------------------

# Find URLStart and URLEnd events for the selected question
url_events = df[
    (df["Event"].isin(["URLStart", "URLEnd"])) &
    (df["Event value"].str.contains(
        f"Question {QUESTION_ID}", na=False
    ))
]

if url_events.empty:
    raise RuntimeError(f"No URL events found for Question {QUESTION_ID}")

# Extract start and end timestamps of the question
t_start = url_events[url_events["Event"] == "URLStart"]["Recording timestamp"].min()
t_end = url_events[url_events["Event"] == "URLEnd"]["Recording timestamp"].max()

# ----------------------------------------------------
# Extract fixations within the time window
# ----------------------------------------------------

# Keep only fixation events within the question time window
fix = df[
    (df["Eye movement type"] == "Fixation") &
    (df["Recording timestamp"].between(t_start, t_end))
].copy()

# Keep only fixations with valid normalized coordinates
fix = fix[
    (fix["Fixation point X (MCSnorm)"].between(0, 1)) &
    (fix["Fixation point Y (MCSnorm)"].between(0, 1))
]

# Sort fixations chronologically
fix = fix.sort_values("Recording timestamp").reset_index(drop=True)

if len(fix) < 2:
    raise RuntimeError("Not enough fixations to compute saccades")

# ----------------------------------------------------
# Load stimulus image
# ----------------------------------------------------

# Load the stimulus image and get its size
img = Image.open(IMAGE_PATH)
w, h = img.size

# Convert normalized fixation coordinates to pixel coordinates
fix["X_px"] = fix["Fixation point X (MCSnorm)"] * w
fix["Y_px"] = fix["Fixation point Y (MCSnorm)"] * h

# ----------------------------------------------------
# Compute saccades
# ----------------------------------------------------

saccades = []

# A saccade is approximated as the movement between two consecutive fixations
for i in range(len(fix) - 1):
    x1, y1 = fix.loc[i, ["X_px", "Y_px"]]
    x2, y2 = fix.loc[i + 1, ["X_px", "Y_px"]]

    # Euclidean distance between consecutive fixation points
    dist = np.hypot(x2 - x1, y2 - y1)

    # Analysis filter to remove small noisy movements
    if dist >= ANALYSIS_MIN_PX:
        saccades.append((x1, y1, x2, y2, dist))

if len(saccades) == 0:
    raise RuntimeError("No valid saccades after filtering.")

# ----------------------------------------------------
# Saccade statistics
# ----------------------------------------------------

# Extract saccade lengths
lengths = [s[4] for s in saccades]

print("---- Saccade Statistics ----")
print(f"Number of saccades: {len(lengths)}")
print(f"Mean length (px): {np.mean(lengths):.2f}")
print(f"Median length (px): {np.median(lengths):.2f}")
print(f"Std (px): {np.std(lengths):.2f}")
print("----------------------------")

# ----------------------------------------------------
# Visualization
# ----------------------------------------------------

plt.figure(figsize=(5.5, 9))
plt.imshow(img)

# Draw saccade lines on top of the stimulus image
for x1, y1, x2, y2, dist in saccades:

    # Additional visualization filter for readability
    if dist < VISUAL_MIN_PX:
        continue

    plt.plot(
        [x1, x2],
        [y1, y2],
        color="#d32f2f",
        alpha=ALPHA,
        linewidth=LINEWIDTH,
        zorder=1
    )

plt.title(
    f"{PARTICIPANT} – Saccades (cleaned)\nQuestion {QUESTION_ID}",
    fontsize=13
)

plt.axis("off")
plt.tight_layout(pad=0)

# ----------------------------------------------------
# Save output
# ----------------------------------------------------

# Define output file path
out_path = os.path.join(
    OUTPUT_DIR,
    f"{PARTICIPANT}_Question{QUESTION_ID}_SaccadesClean.png"
)

# Save saccade visualization
plt.savefig(out_path, dpi=300)
plt.close()

print("Clean, filtered saccade visualization saved.")