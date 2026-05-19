import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ============================================================
# SETTINGS
# ============================================================

PARTICIPANT = "Participant51"
QUESTION_ID = 11

# Minimum normalized movement distance used to filter very small saccades
ANALYSIS_MIN_NORM = 0.002

MIN_FIX_DURATION = 80
MAX_FIX_DURATION = 1000
MAX_FIXATIONS = 80

LINEWIDTH = 1.6
LINE_ALPHA = 0.9
FIX_SIZE = 10

# ============================================================
# OPTIONAL PLOT OFFSET
# ============================================================

# Participant-specific vertical shifts for the top text area.
# This correction is applied only for plotting, not for metric computation.
TOP_TEXT_SHIFTS = {
    "Participant51": 0.00,
    "Participant52": 0.00,
    "Participant53": 0.00,
    "Participant54": 0.00,
    "Participant55": 0.00,
}

TOP_TEXT_THRESHOLD = 0.28

# ============================================================
# PATHS
# ============================================================
# Define project, data, stimulus, and output paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "testC")
STIM_PATH = os.path.join(DATA_PATH, "stimuli")

def get_output_dir(participant, question_id):
    """
    Create and return all output paths for one participant and question.
    """
    participant_dir = os.path.join(BASE_DIR, "results", "testC", participant.lower())
    output_dir = os.path.join(participant_dir, "scanpath", f"q{question_id}")
    os.makedirs(output_dir, exist_ok=True)

    return {
        "dir": output_dir,
        "plot": os.path.join(output_dir, "scanpath.png"),
        "csv": os.path.join(output_dir, "scanpath_report.csv"),
        "fix": os.path.join(output_dir, "scanpath_fixations.csv")
    }

# ============================================================
# FIXATION PROCESSING
# ============================================================

def reduce_fixations(fix):
    """
   Clean fixation data by removing duplicates, filtering fixation duration,
   and sorting fixations chronologically.
   """
    fix = fix.copy()
    # Remove duplicate fixation events if an index column is available
    if "Eye movement type index" in fix.columns:
        fix = fix.drop_duplicates(subset="Eye movement type index")
    # Keep only fixations within the accepted duration range
    if "Gaze event duration [ms]" in fix.columns:
        fix = fix[
            (fix["Gaze event duration [ms]"] >= MIN_FIX_DURATION) &
            (fix["Gaze event duration [ms]"] <= MAX_FIX_DURATION)
            ]
    # Sort fixations by timestamp
    fix = fix.sort_values("Recording timestamp [ms]")

    return fix.reset_index(drop=True)


def reduce_for_plot(fix):
    """
    Reduce the number of plotted fixations for readability.

    The full fixation sequence is still used for metric computation.
    """
    fix = fix.copy()

    if len(fix) > MAX_FIXATIONS:
        indices = np.linspace(0, len(fix) - 1, MAX_FIXATIONS).astype(int)
        fix = fix.iloc[indices]

    return fix.reset_index(drop=True)

# ============================================================
# HELPERS
# ============================================================

def get_fixations_for_question(df, question_label):
    """
    Extract valid fixations for one question using the URLStart and URLEnd events.
    """
    url_events = df[
        (df["Event"].isin(["URLStart", "URLEnd"])) &
        (df["Event value"] == question_label)
        ]
    # Skip if start or end event is missing
    if len(url_events) < 2:
        return None, None

    t_start = url_events[url_events["Event"] == "URLStart"]["Recording timestamp [ms]"].min()
    t_end   = url_events[url_events["Event"] == "URLEnd"]["Recording timestamp [ms]"].max()

    # Duration in seconds
    duration_sec = (t_end - t_start) / 1000.0

    # Select fixation events within the question time window
    fix = df[
        (df["Eye movement type"] == "Fixation") &
        (df["Recording timestamp [ms]"].between(t_start, t_end))
        ].copy()

    # Keep only valid normalized coordinates
    fix = fix[
        (fix["Fixation point X [MCS norm]"].between(0, 1)) &
        (fix["Fixation point Y [MCS norm]"].between(0, 1))
        ]

    fix = reduce_fixations(fix)

    if len(fix) < 2:
        return None, None

    return fix, duration_sec

# ============================================================
# METRICS
# ============================================================

def compute_metrics(fix):
    """
   Compute scanpath-based metrics from consecutive fixation movements.

   Metrics include:
   - Scanpath length
   - Vertical movement ratio
   - Regression rate
   - Directional entropy
   """
    dx = np.diff(fix["Fixation point X [MCS norm]"].to_numpy())
    dy = np.diff(fix["Fixation point Y [MCS norm]"].to_numpy())

    distances = np.sqrt(dx**2 + dy**2)
    # Remove very small movements that are likely noise
    valid = distances >= ANALYSIS_MIN_NORM
    dx = dx[valid]
    dy = dy[valid]
    distances = distances[valid]

    if len(distances) == 0:
        return None
    # Compare vertical and horizontal movement components
    vertical = np.sum(np.abs(dy))
    horizontal = np.sum(np.abs(dx))

    # Proportion of leftward movements
    vertical_ratio = vertical / (vertical + horizontal + 1e-12)
    regression_rate = np.sum(dx < 0) / len(dx)

    # Directional entropy based on movement angles
    angles = np.arctan2(dy, dx)
    hist, _ = np.histogram(angles, bins=8)
    prob = hist / (np.sum(hist) + 1e-12)

    entropy = -np.sum(prob * np.log2(prob + 1e-12))

    return distances, vertical_ratio, regression_rate, entropy

# ============================================================
# PLOT
# ============================================================

def save_scanpath_plot(participant, fix_analysis, qid, plot_path):
    """
     Create and save a time-coded scanpath plot on top of the stimulus image.

     The optional participant-specific top-text shift is applied only here.
     """
    img_path = os.path.join(STIM_PATH, f"Question{qid}.png")
    if not os.path.exists(img_path):
        print("Stimulus fehlt")
        return None

    # Reduce fixations only for visualization
    fix_plot = reduce_for_plot(fix_analysis)

    if fix_plot is None or len(fix_plot) < 2:
        return None

    img = Image.open(img_path)
    w, h = img.size

    fix_plot = fix_plot.copy()

    # Start with original normalized coordinates
    fix_plot["X_shifted"] = fix_plot["Fixation point X [MCS norm]"].copy()
    fix_plot["Y_shifted"] = fix_plot["Fixation point Y [MCS norm]"].copy()

    # Apply vertical correction only to the top text area
    top_text_y_shift = TOP_TEXT_SHIFTS.get(participant, 0.00)
    mask_top = fix_plot["Y_shifted"] < TOP_TEXT_THRESHOLD
    fix_plot.loc[mask_top, "Y_shifted"] = (
            fix_plot.loc[mask_top, "Y_shifted"] + top_text_y_shift
    )
    # Keep corrected coordinates within valid normalized bounds
    fix_plot["X_shifted"] = fix_plot["X_shifted"].clip(0, 1)
    fix_plot["Y_shifted"] = fix_plot["Y_shifted"].clip(0, 1)

    # Convert normalized coordinates to pixels
    fix_plot["X_px"] = fix_plot["X_shifted"] * w
    fix_plot["Y_px"] = fix_plot["Y_shifted"] * h

    n = len(fix_plot)
    cmap = plt.cm.plasma

    plt.figure(figsize=(6, 9))
    plt.imshow(img)

    # Draw time-coded movement lines between consecutive fixations
    for i in range(n - 1):
        x1, y1 = fix_plot.loc[i, ["X_px", "Y_px"]]
        x2, y2 = fix_plot.loc[i + 1, ["X_px", "Y_px"]]

        dx = fix_plot.loc[i + 1, "X_shifted"] - fix_plot.loc[i, "X_shifted"]
        dy = fix_plot.loc[i + 1, "Y_shifted"] - fix_plot.loc[i, "Y_shifted"]

        dist = math.sqrt(dx * dx + dy * dy)
        if dist < ANALYSIS_MIN_NORM:
            continue

        plt.plot(
            [x1, x2], [y1, y2],
            color=cmap(i / max(n - 1, 1)),
            linewidth=LINEWIDTH,
            alpha=LINE_ALPHA
        )
    # Plot fixation points using the same time-coded color scale
    plt.scatter(
        fix_plot["X_px"],
        fix_plot["Y_px"],
        c=np.linspace(0, 1, n),
        cmap="plasma",
        s=FIX_SIZE,
        alpha=0.9
    )
    # Mark first and last fixation
    plt.scatter(fix_plot.loc[0, "X_px"], fix_plot.loc[0, "Y_px"], s=80, marker="o")
    plt.scatter(fix_plot.loc[n - 1, "X_px"], fix_plot.loc[n - 1, "Y_px"], s=80, marker="X")

    plt.title(f"{participant} – Time-coded Scanpath (Q{qid})")
    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.savefig(plot_path, dpi=300)
    plt.close()

    return fix_plot

# ============================================================
# MAIN
# ============================================================

# Create output paths
paths = get_output_dir(PARTICIPANT, QUESTION_ID)

# Load participant data
file_path = os.path.join(DATA_PATH, f"{PARTICIPANT}.tsv")
df = pd.read_csv(file_path, sep="\t", low_memory=False)

# Find the selected question
question_rows = df[
    (df["Event"] == "URLStart") &
    (df["Event value"].astype(str).str.contains(f"Question {QUESTION_ID}", na=False))
    ]

if question_rows.empty:
    raise RuntimeError("Question nicht gefunden")

q_label = question_rows.iloc[0]["Event value"]

# Extract fixations for the selected question
fix, duration = get_fixations_for_question(df, q_label)

if fix is None:
    raise RuntimeError("Keine Fixationen")

# Compute scanpath metrics
metrics = compute_metrics(fix)
if metrics is None:
    raise RuntimeError("Keine gueltigen Sakkaden")

distances, vr, rr, ent = metrics

# Create summary dataframe
result = pd.DataFrame([{
    "Participant": PARTICIPANT,
    "Question_ID": QUESTION_ID,
    "Duration_sec": round(duration, 2),
    "Fixations_used": len(fix),
    "Saccades": len(distances),
    "Scanpath_Length": round(np.sum(distances), 4),
    "Vertical_Ratio": round(vr, 4),
    "Regression_Rate": round(rr, 4),
    "Entropy": round(ent, 4)
}])

print(result)

# Save scanpath plot
save_scanpath_plot(PARTICIPANT, fix, QUESTION_ID, paths["plot"])

# Save metric summary
result.to_csv(paths["csv"], index=False)

# Save fixation coordinates in the same simple format as before
fix_export = fix[[
    "Recording timestamp [ms]",
    "Fixation point X [MCS norm]",
    "Fixation point Y [MCS norm]"
]].copy()

fix_export.columns = ["t", "x", "y"]
fix_export.to_csv(paths["fix"], index=False)

print("\n Finale Fixationen:", len(fix))
print("Gespeichert in:", paths["dir"])