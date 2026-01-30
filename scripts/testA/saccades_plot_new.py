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
# Configuration
# ----------------------------------------------------
PARTICIPANT = "Participant2"
QUESTION_ID = 4

DATA_FILE = os.path.join(
    "..", "..", "data", "testA", f"{PARTICIPANT}.tsv"
)

IMAGE_PATH = os.path.join(
    "..", "..", "data", "testA", "stimuli", f"Question{QUESTION_ID}.png"
)

OUTPUT_DIR = os.path.join(
    "..", "..", "results", "testA", PARTICIPANT.lower(), "saccades_clean"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------
# Load TSV
# ----------------------------------------------------
df = pd.read_csv(DATA_FILE, sep="\t", low_memory=False)

# ----------------------------------------------------
# Use FIXATIONS (on stimulus only)
# ----------------------------------------------------
fix = df[df["Eye movement type"] == "Fixation"].copy()

fix = fix[
    (fix["Fixation point X (MCSnorm)"].between(0, 1)) &
    (fix["Fixation point Y (MCSnorm)"].between(0, 1))
]

# Sort temporally
fix = fix.sort_values("Recording timestamp").reset_index(drop=True)

# ----------------------------------------------------
# Optional: limit number of fixations (temporal)
# ----------------------------------------------------
MAX_FIX = 800
if len(fix) > MAX_FIX:
    fix = fix.iloc[:MAX_FIX]

# ----------------------------------------------------
# Load stimulus image
# ----------------------------------------------------
img = Image.open(IMAGE_PATH)
w, h = img.size

fix["X_px"] = fix["Fixation point X (MCSnorm)"] * w
fix["Y_px"] = fix["Fixation point Y (MCSnorm)"] * h

# ----------------------------------------------------
# Temporal normalization (for color coding)
# ----------------------------------------------------
t = fix["Recording timestamp"].to_numpy()
t_norm = (t - t.min()) / (t.max() - t.min())

# ----------------------------------------------------
# Visualization
# ----------------------------------------------------
plt.figure(figsize=(5.5, 9))
plt.imshow(img)
plt.gca().invert_yaxis()  # screen coordinate system

# Plot saccades (subtle)
for i in range(len(fix) - 1):
    plt.plot(
        [fix.iloc[i]["X_px"], fix.iloc[i + 1]["X_px"]],
        [fix.iloc[i]["Y_px"], fix.iloc[i + 1]["Y_px"]],
        color="black",
        alpha=0.05,
        linewidth=0.8,
        zorder=1
    )

# Plot fixations (time-coded)
sc = plt.scatter(
    fix["X_px"],
    fix["Y_px"],
    c=t_norm,
    cmap="plasma",
    s=18,
    alpha=0.85,
    edgecolors="none",
    zorder=2
)

# Colorbar = time progression
cbar = plt.colorbar(sc, fraction=0.035, pad=0.02)
cbar.set_label("Time progression", fontsize=10)

plt.title(
    f"{PARTICIPANT} – Gaze Path (Question {QUESTION_ID})",
    fontsize=13
)

# Axes in pixels (scientific & reproducible)
plt.xlabel("X (pixels)")
plt.ylabel("Y (pixels)")
plt.xlim(0, w)
plt.ylim(h, 0)

plt.tight_layout()

# ----------------------------------------------------
# Save
# ----------------------------------------------------
out_path = os.path.join(
    OUTPUT_DIR,
    f"{PARTICIPANT}_Question{QUESTION_ID}_GazePath.png"
)

plt.savefig(out_path, dpi=300)
plt.close()

print("✅ Clean, structured gaze visualization saved.")
