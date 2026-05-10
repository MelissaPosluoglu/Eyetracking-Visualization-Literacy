import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
import os
import numpy as np
from scipy.ndimage import gaussian_filter

# ----------------------------------------------------
# Use a non-interactive backend (no GUI required)
# ----------------------------------------------------
matplotlib.use("Agg")

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------

# Participant to analyze
PARTICIPANT = "Participant13"

# Input file paths
DATA_FILE = os.path.join(
    "..", "..", "data", "testA", f"{PARTICIPANT}.tsv"
)

IMAGE_PATH = os.path.join(
    "..", "..", "data", "testA", "stimuli", "Question3.png"
)

# Output directory for debug plots
OUTPUT_DIR = os.path.join(
    "..", "..", "results", "testA", PARTICIPANT.lower(), "debug"
)

# Create output directory if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------
# Load TSV data
# ----------------------------------------------------

# Load eye-tracking data
df = pd.read_csv(DATA_FILE, sep="\t", low_memory=False)

# Keep only fixation events
fix = df[df["Eye movement type"] == "Fixation"].copy()
print(f"Total fixations in file: {len(fix)}")

# Optional subsampling for faster processing
fix = fix.sample(n=8000, random_state=42)

# ----------------------------------------------------
# Load stimulus image
# ----------------------------------------------------

# Load the stimulus image and get its dimensions
img = Image.open(IMAGE_PATH)
w, h = img.size

# Convert normalized fixation coordinates to pixel coordinates
fix["X_px"] = fix["Fixation point X (MCSnorm)"] * w
fix["Y_px"] = fix["Fixation point Y (MCSnorm)"] * h

# ----------------------------------------------------
# Create fixation density map (heatmap)
# ----------------------------------------------------

# Create a 2D histogram of fixation locations
heatmap, xedges, yedges = np.histogram2d(
    fix["X_px"],
    h - fix["Y_px"],
    bins=[300, 300],
    range=[[0, w], [0, h]]
)

# Smooth the density map for better visualization
heatmap = gaussian_filter(heatmap, sigma=10)

# Output path for the heatmap image
out_path = os.path.join(
    OUTPUT_DIR,
    f"{PARTICIPANT}_FIXATION_DENSITY_Q3.png"
)

# ----------------------------------------------------
# Plot heatmap on top of stimulus image
# ----------------------------------------------------

plt.figure(figsize=(6, 6))
plt.imshow(img, extent=[0, w, 0, h])

# Overlay the smoothed heatmap
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

# Save the figure
plt.savefig(out_path, dpi=220)
plt.close()

print("✅ Debug plot saved successfully.")