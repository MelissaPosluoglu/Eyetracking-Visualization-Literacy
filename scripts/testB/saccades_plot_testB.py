import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
import os
import numpy as np

# DAS IST FÜR PARTICIPANT 20-30


# ----------------------------------------------------
# Non-interactive backend
# ----------------------------------------------------
matplotlib.use("Agg")

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------
PARTICIPANT = "Participant41"
QUESTION_ID = 12

DATA_FILE = os.path.join("..","..", "..", "data", "testB", f"{PARTICIPANT}.tsv")
IMAGE_PATH = os.path.join("..","..", "..", "data", "testB", "stimuli", f"Question{QUESTION_ID}.png")
OUTPUT_DIR = os.path.join("..","..", "..", "results", "testB", PARTICIPANT.lower(), "saccades")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------
# Parameters
# ----------------------------------------------------
ANALYSIS_MIN_PX = 20
VISUAL_MIN_PX = 60
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
    (df["Event value"].astype(str).str.contains(f"Question {QUESTION_ID}", na=False))
    ].copy()

if url_events.empty:
    raise RuntimeError(f"No URL events found for Question {QUESTION_ID}")

t_start = url_events[url_events["Event"] == "URLStart"]["Recording timestamp [ms]"].min()
t_end = url_events[url_events["Event"] == "URLEnd"]["Recording timestamp [ms]"].max()

# ----------------------------------------------------
# Fixationen im Zeitfenster
# ----------------------------------------------------
fix = df[
    (df["Eye movement type"] == "Fixation") &
    (df["Recording timestamp [ms]"].between(t_start, t_end))
    ].copy()

fix = fix[
    (fix["Fixation point X [MCS norm]"].between(0, 1)) &
    (fix["Fixation point Y [MCS norm]"].between(0, 1))
    ]

fix = fix.sort_values("Recording timestamp [ms]").reset_index(drop=True)

if len(fix) < 2:
    raise RuntimeError("Not enough fixations to compute saccades")

# ----------------------------------------------------
# Stimulus laden
# ----------------------------------------------------
img = Image.open(IMAGE_PATH)
w, h = img.size

fix["X_px"] = fix["Fixation point X [MCS norm]"] * w
fix["Y_px"] = fix["Fixation point Y [MCS norm]"] * h

# ----------------------------------------------------
# Sakkaden berechnen
# ----------------------------------------------------
saccades = []

for i in range(len(fix) - 1):
    x1, y1 = fix.loc[i, ["X_px", "Y_px"]]
    x2, y2 = fix.loc[i + 1, ["X_px", "Y_px"]]

    dist = np.hypot(x2 - x1, y2 - y1)

    if dist >= ANALYSIS_MIN_PX:
        saccades.append((x1, y1, x2, y2, dist))

if len(saccades) == 0:
    raise RuntimeError("No valid saccades after filtering.")

# ----------------------------------------------------
# Analysekennwerte
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
plt.imshow(img)

for x1, y1, x2, y2, dist in saccades:
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
print("📁 Saved to:", out_path)