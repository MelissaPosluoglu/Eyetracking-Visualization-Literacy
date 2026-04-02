import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
import os
import numpy as np

# DAS IST FÜR PARTICIPANT 1 BIS 20

# ----------------------------------------------------
# Non-interactive backend
# ----------------------------------------------------
matplotlib.use("Agg")

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------
PARTICIPANT = ("Participant5")
QUESTION_ID = 1

DATA_FILE = os.path.join(
    "..", "..", "data", "testA", f"{PARTICIPANT}.tsv"
)

IMAGE_PATH = os.path.join(
    "..", "..", "data", "testA", "stimuli", f"Question{QUESTION_ID}.png"
)

OUTPUT_DIR = os.path.join(
    "..", "..", "results", "testA",
    PARTICIPANT.lower(), "saccades"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------
# Parameters
# ----------------------------------------------------
ANALYSIS_MIN_PX = 20      # Filter gegen Messrauschen
VISUAL_MIN_PX   = 60      # Nur für Lesbarkeit
ALPHA = 0.25
LINEWIDTH = 1.0

# ----------------------------------------------------
# Load data
# ----------------------------------------------------
df = pd.read_csv(DATA_FILE, sep="\t", low_memory=False)

# ----------------------------------------------------
# URL window bestimmen
# ----------------------------------------------------
url_events = df[
    (df["Event"].isin(["URLStart", "URLEnd"])) &
    (df["Event value"].str.contains(
        f"Question {QUESTION_ID}", na=False
    ))
    ]

if url_events.empty:
    raise RuntimeError(f"No URL events found for Question {QUESTION_ID}")

t_start = url_events[url_events["Event"] == "URLStart"]["Recording timestamp"].min()
t_end   = url_events[url_events["Event"] == "URLEnd"]["Recording timestamp"].max()

# ----------------------------------------------------
# Fixationen im Zeitfenster
# ----------------------------------------------------
fix = df[
    (df["Eye movement type"] == "Fixation") &
    (df["Recording timestamp"].between(t_start, t_end))
    ].copy()

fix = fix[
    (fix["Fixation point X (MCSnorm)"].between(0, 1)) &
    (fix["Fixation point Y (MCSnorm)"].between(0, 1))
    ]

fix = fix.sort_values("Recording timestamp").reset_index(drop=True)

if len(fix) < 2:
    raise RuntimeError("Not enough fixations to compute saccades")

# ----------------------------------------------------
# Stimulus laden
# ----------------------------------------------------
img = Image.open(IMAGE_PATH)
w, h = img.size

fix["X_px"] = fix["Fixation point X (MCSnorm)"] * w
fix["Y_px"] = fix["Fixation point Y (MCSnorm)"] * h

# ----------------------------------------------------
# Sakkaden berechnen
# ----------------------------------------------------
saccades = []

for i in range(len(fix) - 1):
    x1, y1 = fix.loc[i, ["X_px", "Y_px"]]
    x2, y2 = fix.loc[i + 1, ["X_px", "Y_px"]]

    dist = np.hypot(x2 - x1, y2 - y1)

    # Analysefilter (Rauschen entfernen)
    if dist >= ANALYSIS_MIN_PX:
        saccades.append((x1, y1, x2, y2, dist))

if len(saccades) == 0:
    raise RuntimeError("No valid saccades after filtering.")

# ----------------------------------------------------
# Analysekennwerte (wissenschaftlich korrekt)
# ----------------------------------------------------
lengths = [s[4] for s in saccades]

print("---- Saccade Statistics ----")
print(f"Number of saccades: {len(lengths)}")
print(f"Mean length (px): {np.mean(lengths):.2f}")
print(f"Median length (px): {np.median(lengths):.2f}")
print(f"Std (px): {np.std(lengths):.2f}")
print("----------------------------")

# ----------------------------------------------------
# Visualisierung
# ----------------------------------------------------
plt.figure(figsize=(5.5, 9))
plt.imshow(img)   # kein invert_yaxis mehr!

for x1, y1, x2, y2, dist in saccades:

    # zusätzlicher Visualisierungsfilter
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
# Save
# ----------------------------------------------------
out_path = os.path.join(
    OUTPUT_DIR,
    f"{PARTICIPANT}_Question{QUESTION_ID}_SaccadesClean.png"
)

plt.savefig(out_path, dpi=300)
plt.close()

print("✅ Clean, filtered saccade visualization saved.")