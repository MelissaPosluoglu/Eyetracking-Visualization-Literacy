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
PARTICIPANT = "Participant4"
QUESTION_ID = 10

DATA_FILE = os.path.join("..", "..", "data", "testA", f"{PARTICIPANT}.tsv")
IMAGE_PATH = os.path.join("..", "..", "data", "testA", "stimuli", f"Question{QUESTION_ID}.png")
OUTPUT_DIR = os.path.join("..", "..", "results", "testA", PARTICIPANT.lower(), "debug_clean")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------
# Load TSV
# ----------------------------------------------------
df = pd.read_csv(DATA_FILE, sep="\t", low_memory=False)

fix = df[df["Eye movement type"] == "Fixation"].copy()
print(f"Total fixations in file: {len(fix)}")

# OPTIONAL: limit for visualization (keeps temporal order!)
MAX_FIX = 6000
if len(fix) > MAX_FIX:
    fix = fix.iloc[:MAX_FIX]

# ----------------------------------------------------
# Load stimulus
# ----------------------------------------------------
img = Image.open(IMAGE_PATH)
w, h = img.size

fix["X_px"] = fix["Fixation point X (MCSnorm)"] * w
fix["Y_px"] = fix["Fixation point Y (MCSnorm)"] * h

# ----------------------------------------------------
# Normalize time & duration
# ----------------------------------------------------
t = fix["Recording timestamp"].to_numpy()
t_norm = (t - t.min()) / (t.max() - t.min())

dur = fix["Gaze event duration"].to_numpy()
dur_clipped = np.clip(dur, 40, 400)   # cap extremes
size = dur_clipped / 12

# ----------------------------------------------------
# Plot
# ----------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 9))

ax.imshow(img, extent=[0, w, h, 0])
ax.set_xlim(0, w)
ax.set_ylim(h, 0)

sc = ax.scatter(
    fix["X_px"],
    fix["Y_px"],
    s=size,
    c=t_norm,
    cmap="viridis",
    alpha=0.25,          # KEY: calm visualization
    edgecolors="none"
)

ax.set_title(
    f"{PARTICIPANT} – All Fixations (Clean Debug, Question {QUESTION_ID})",
    fontsize=13
)

ax.set_xlabel("X (pixels)")
ax.set_ylabel("Y (pixels)")

# Subtle colorbar
cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.015)
cbar.set_label("Time progression")

# ----------------------------------------------------
# Save
# ----------------------------------------------------
out_path = os.path.join(
    OUTPUT_DIR,
    f"{PARTICIPANT}_Question{QUESTION_ID}_ALL_FIXATIONS_CLEAN.png"
)

plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

print("✅ Clean debug fixation plot saved.")
