import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
import os
import numpy as np
from scipy.ndimage import gaussian_filter


# ----------------------------------------------------
# Use non-interactive backend (NO GUI)
# ----------------------------------------------------
matplotlib.use("Agg")

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------
PARTICIPANT = "Participant13"

DATA_FILE = os.path.join(
    "..", "..", "data", "testA", f"{PARTICIPANT}.tsv"
)

IMAGE_PATH = os.path.join(
    "..", "..", "data", "testA", "stimuli", "Question3.png"
)

OUTPUT_DIR = os.path.join(
    "..", "..", "results", "testA", PARTICIPANT.lower(), "debug"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------
# Load TSV
# ----------------------------------------------------
df = pd.read_csv(DATA_FILE, sep="\t", low_memory=False)

fix = df[df["Eye movement type"] == "Fixation"].copy()
print(f"Total fixations in file: {len(fix)}")

# OPTIONAL: subsample for performance (recommended)
fix = fix.sample(n=8000, random_state=42)

# ----------------------------------------------------
# Load stimulus image
# ----------------------------------------------------
img = Image.open(IMAGE_PATH)
w, h = img.size

fix["X_px"] = fix["Fixation point X (MCSnorm)"] * w
fix["Y_px"] = fix["Fixation point Y (MCSnorm)"] * h

# ----------------------------------------------------
# Fixation Density Map (Heatmap)
# ----------------------------------------------------

heatmap, xedges, yedges = np.histogram2d(
    fix["X_px"],
    h - fix["Y_px"],
    bins=[300, 300],
    range=[[0, w], [0, h]]
)

# Smooth density
heatmap = gaussian_filter(heatmap, sigma=10)

out_path = os.path.join(
    OUTPUT_DIR,
    f"{PARTICIPANT}_FIXATION_DENSITY_Q3.png"
)


plt.figure(figsize=(6, 6))
plt.imshow(img, extent=[0, w, 0, h])

plt.imshow(
    heatmap.T,
    extent=[0, w, 0, h],
    origin="lower",
    cmap="inferno",
    alpha=0.6
)

plt.title(f"{PARTICIPANT} – Fixation Density (Question 3)", fontsize=14)
plt.axis("off")
plt.tight_layout()

plt.savefig(out_path, dpi=220)
plt.close()


print("✅ DEBUG plot saved successfully.")
