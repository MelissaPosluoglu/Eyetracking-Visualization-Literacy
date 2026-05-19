import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
import os
import numpy as np

# ----------------------------------------------------
# NON-INTERACTIVE BACKEND
# ----------------------------------------------------

# Save plots without opening a window
matplotlib.use("Agg")

# ----------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------

# Select participant and question
PARTICIPANT = "Participant51"
QUESTION_ID = 2

DATA_FILE = os.path.join("..", "..", "data", "testC", f"{PARTICIPANT}.tsv")
IMAGE_PATH = os.path.join("..", "..", "data", "testC", "stimuli", f"Question{QUESTION_ID}.png")
OUTPUT_DIR = os.path.join("..", "..", "results", "testC", PARTICIPANT.lower(), "saccades")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------
# PARAMETERS
# ----------------------------------------------------

# Minimum saccade length for analysis
ANALYSIS_MIN_PX = 20

# Minimum saccade length for visualization
VISUAL_MIN_PX = 60

# Line transparency and width
ALPHA = 0.25
LINEWIDTH = 1.0

# ----------------------------------------------------
# LOAD DATA
# ----------------------------------------------------
df = pd.read_csv(DATA_FILE, sep="\t", low_memory=False)

# ----------------------------------------------------
# DETECT QUESTION TIME WINDOW
# ----------------------------------------------------

# Select URL events for the current question
question_events = df[
    (df["Event"].isin(["URLStart", "URLEnd"])) &
    (df["Event value"].astype(str).str.contains(f"Question {QUESTION_ID}", na=False))
    ].copy()

# Stop if no events were found
if question_events.empty:
    raise RuntimeError(f"No URL events found for Question {QUESTION_ID}")

# Get question start event
start_rows = question_events[question_events["Event"] == "URLStart"]
if start_rows.empty:
    raise RuntimeError(f"No URLStart found for Question {QUESTION_ID}")


# Define start time
t_start = start_rows["Recording timestamp [ms]"].min()

# Prefer URLEnd as question end time
end_rows = question_events[
    (question_events["Event"] == "URLEnd") &
    (question_events["Recording timestamp [ms]"] > t_start)
    ]

if not end_rows.empty:
    # Use latest URLEnd for this question
    t_end = end_rows["Recording timestamp [ms]"].max()
else:
    # Fallback: use next question start
    all_urlstarts = df[
        (df["Event"] == "URLStart") &
        (df["Event value"].astype(str).str.contains("Question", na=False))
        ].sort_values("Recording timestamp [ms]")

    next_starts = all_urlstarts[all_urlstarts["Recording timestamp [ms]"] > t_start]

    if not next_starts.empty:
        t_end = next_starts["Recording timestamp [ms]"].iloc[0]
    else:
        t_end = df["Recording timestamp [ms]"].max()

# ----------------------------------------------------
# EXTRACT FIXATIONS
# ----------------------------------------------------

# Select fixations within the question time window
fix = df[
    (df["Eye movement type"] == "Fixation") &
    (df["Recording timestamp [ms]"] >= t_start) &
    (df["Recording timestamp [ms]"] < t_end)
    ].copy()

# Keep only valid normalized coordinates
fix = fix[
    (fix["Fixation point X [MCS norm]"].between(0, 1)) &
    (fix["Fixation point Y [MCS norm]"].between(0, 1))
    ].copy()

# Remove duplicate fixation events
if "Eye movement type index" in fix.columns:
    fix = fix.drop_duplicates(subset="Eye movement type index")

# Sort fixations by time
fix = fix.sort_values("Recording timestamp [ms]").reset_index(drop=True)

# Stop if fewer than two fixations exist
if len(fix) < 2:
    raise RuntimeError("Not enough fixations to compute saccades")


# ----------------------------------------------------
# LOAD STIMULUS IMAGE
# ----------------------------------------------------

img = Image.open(IMAGE_PATH)
w, h = img.size


# Convert normalized fixation coordinates to pixels
fix["X_px"] = fix["Fixation point X [MCS norm]"] * w
fix["Y_px"] = fix["Fixation point Y [MCS norm]"] * h

# ----------------------------------------------------
# COMPUTE SACCADES
# ----------------------------------------------------
saccades = []

# Calculate distances between consecutive fixations
for i in range(len(fix) - 1):
    x1, y1 = fix.loc[i, ["X_px", "Y_px"]]
    x2, y2 = fix.loc[i + 1, ["X_px", "Y_px"]]

    # Euclidean distance in pixels
    dist = np.hypot(x2 - x1, y2 - y1)

    # Keep only saccades above analysis threshold
    if dist >= ANALYSIS_MIN_PX:
        saccades.append((x1, y1, x2, y2, dist))

if len(saccades) == 0:
    raise RuntimeError("No valid saccades after filtering.")

# ----------------------------------------------------
# SACCADE STATISTICS
# ----------------------------------------------------

lengths = [s[4] for s in saccades]

# Print basic statistics
print("---- Saccade Statistics ----")
print(f"Number of saccades: {len(lengths)}")
print(f"Mean length (px): {np.mean(lengths):.2f}")
print(f"Median length (px): {np.median(lengths):.2f}")
print(f"Std (px): {np.std(lengths):.2f}")
print("----------------------------")

# ----------------------------------------------------
# VISUALIZATION
# ----------------------------------------------------
plt.figure(figsize=(5.5, 9))
plt.imshow(img)

# Draw saccade lines
for x1, y1, x2, y2, dist in saccades:
    # Skip short saccades in the visualization
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
# Add plot title
plt.title(
    f"{PARTICIPANT} – Saccades (cleaned)\nQuestion {QUESTION_ID}",
    fontsize=13
)

plt.axis("off")
plt.tight_layout(pad=0)

# ----------------------------------------------------
# SAVE OUTPUT
# ----------------------------------------------------
out_path = os.path.join(
    OUTPUT_DIR,
    f"{PARTICIPANT}_Question{QUESTION_ID}_SaccadesClean.png"
)

plt.savefig(out_path, dpi=300)
plt.close()

print("Clean, filtered saccade visualization saved.")
print("Saved to:", out_path)