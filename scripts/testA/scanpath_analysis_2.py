import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ============================================================
# SETTINGS
# ============================================================

PARTICIPANT = "Participant24"
QUESTION_ID = 7

ANALYSIS_MIN_NORM = 0.002

MIN_FIX_DURATION = 80        # 🔥 gleich wie Hauptanalyse
MAX_FIXATIONS = 80           # 🔥 Zielbereich (perfekt für Scanpath)

LINEWIDTH = 1.6
LINE_ALPHA = 0.9
FIX_SIZE = 10

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "testA")
STIM_PATH = os.path.join(DATA_PATH, "stimuli")

def get_output_dir(participant, question_id):
    participant_dir = os.path.join(BASE_DIR, "results", "testA", participant.lower())
    output_dir = os.path.join(participant_dir, "scanpath", f"q{question_id}")
    os.makedirs(output_dir, exist_ok=True)

    return {
        "dir": output_dir,
        "plot": os.path.join(output_dir, "scanpath.png"),
        "csv": os.path.join(output_dir, "scanpath_report.csv"),
        "fix": os.path.join(output_dir, "scanpath_fixations.csv")
    }

# ============================================================
# 🔥 FINAL FIXATION REDUCTION (WISSENSCHAFTLICH SAUBER)
# ============================================================

def reduce_fixations(fix):

    # 1. 🔥 ECHTE FIXATIONEN (wie in PDF!)
    if "Eye movement type index" in fix.columns:
        fix = fix.drop_duplicates(subset="Eye movement type index")

    # 2. Dauerfilter (identisch zur Hauptanalyse)
    if "Gaze event duration [ms]" in fix.columns:
        fix = fix[
            (fix["Gaze event duration [ms]"] >= MIN_FIX_DURATION) &
            (fix["Gaze event duration [ms]"] <= 1000)
            ]

    # 3. Sortieren (WICHTIG für Scanpath!)
    fix = fix.sort_values("Recording timestamp [ms]")

    # 4. 🔥 SMART SAMPLING (gleichmäßige Auswahl statt skip)
    if len(fix) > MAX_FIXATIONS:
        indices = np.linspace(0, len(fix) - 1, MAX_FIXATIONS).astype(int)
        fix = fix.iloc[indices]

    return fix.reset_index(drop=True)

# ============================================================
# HELPERS
# ============================================================

def get_fixations_for_question(df, question_label):

    url_events = df[
        (df["Event"].isin(["URLStart", "URLEnd"])) &
        (df["Event value"] == question_label)
        ]

    if len(url_events) < 2:
        return None, None

    t_start = url_events[url_events["Event"] == "URLStart"]["Recording timestamp [ms]"].min()
    t_end   = url_events[url_events["Event"] == "URLEnd"]["Recording timestamp [ms]"].max()

    duration_sec = (t_end - t_start) / 1000.0

    fix = df[
        (df["Eye movement type"] == "Fixation") &
        (df["Recording timestamp [ms]"].between(t_start, t_end))
        ].copy()

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

    dx = np.diff(fix["Fixation point X [MCS norm]"])
    dy = np.diff(fix["Fixation point Y [MCS norm]"])

    distances = np.sqrt(dx**2 + dy**2)

    valid = distances >= ANALYSIS_MIN_NORM
    dx, dy, distances = dx[valid], dy[valid], distances[valid]

    if len(distances) == 0:
        return None

    vertical = np.sum(np.abs(dy))
    horizontal = np.sum(np.abs(dx))

    vertical_ratio = vertical / (vertical + horizontal + 1e-12)
    regression_rate = np.sum(dx < 0) / len(dx)

    angles = np.arctan2(dy, dx)
    hist, _ = np.histogram(angles, bins=8)
    prob = hist / (np.sum(hist) + 1e-12)

    entropy = -np.sum(prob * np.log2(prob + 1e-12))

    return distances, vertical_ratio, regression_rate, entropy

# ============================================================
# PLOT
# ============================================================

def save_scanpath_plot(participant, df, qid, qlabel, plot_path):

    img_path = os.path.join(STIM_PATH, f"Question{qid}.png")
    if not os.path.exists(img_path):
        print("Stimulus fehlt")
        return None

    fix, _ = get_fixations_for_question(df, qlabel)
    if fix is None:
        return None

    img = Image.open(img_path)
    w, h = img.size

    fix = fix.copy()
    fix["X_px"] = fix["Fixation point X [MCS norm]"] * w
    fix["Y_px"] = fix["Fixation point Y [MCS norm]"] * h

    n = len(fix)
    cmap = plt.cm.plasma

    plt.figure(figsize=(6, 9))
    plt.imshow(img)

    for i in range(n - 1):

        x1, y1 = fix.loc[i, ["X_px", "Y_px"]]
        x2, y2 = fix.loc[i + 1, ["X_px", "Y_px"]]

        dx = fix.loc[i + 1, "Fixation point X [MCS norm]"] - fix.loc[i, "Fixation point X [MCS norm]"]
        dy = fix.loc[i + 1, "Fixation point Y [MCS norm]"] - fix.loc[i, "Fixation point Y [MCS norm]"]

        dist = math.sqrt(dx*dx + dy*dy)
        if dist < ANALYSIS_MIN_NORM:
            continue

        plt.plot([x1, x2], [y1, y2],
                 color=cmap(i / max(n - 1, 1)),
                 linewidth=LINEWIDTH,
                 alpha=LINE_ALPHA)

    plt.scatter(
        fix["X_px"],
        fix["Y_px"],
        c=np.linspace(0, 1, n),
        cmap="plasma",
        s=FIX_SIZE,
        alpha=0.9
    )

    plt.scatter(fix.loc[0, "X_px"], fix.loc[0, "Y_px"], s=80, marker="o")
    plt.scatter(fix.loc[n-1, "X_px"], fix.loc[n-1, "Y_px"], s=80, marker="X")

    plt.title(f"{participant} – Time-coded Scanpath (Q{qid})")
    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.savefig(plot_path, dpi=300)
    plt.close()

    return plot_path

# ============================================================
# MAIN
# ============================================================

paths = get_output_dir(PARTICIPANT, QUESTION_ID)

file_path = os.path.join(DATA_PATH, f"{PARTICIPANT}.tsv")
df = pd.read_csv(file_path, sep="\t", low_memory=False)

question_rows = df[
    (df["Event"] == "URLStart") &
    (df["Event value"].astype(str).str.contains(f"Question {QUESTION_ID}", na=False))
    ]

if question_rows.empty:
    raise RuntimeError("Question nicht gefunden")

q_label = question_rows.iloc[0]["Event value"]

fix, duration = get_fixations_for_question(df, q_label)

if fix is None:
    raise RuntimeError("Keine Fixationen")

metrics = compute_metrics(fix)
distances, vr, rr, ent = metrics

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

# speichern
save_scanpath_plot(PARTICIPANT, df, QUESTION_ID, q_label, paths["plot"])
result.to_csv(paths["csv"], index=False)

# 🔥 MINIMAL & SINNVOLL
fix_export = fix[[
    "Recording timestamp [ms]",
    "Fixation point X [MCS norm]",
    "Fixation point Y [MCS norm]"
]].copy()

fix_export.columns = ["t", "x", "y"]
fix_export.to_csv(paths["fix"], index=False)

print("\n✅ Finale Fixationen:", len(fix))
print("📁 Gespeichert in:", paths["dir"])