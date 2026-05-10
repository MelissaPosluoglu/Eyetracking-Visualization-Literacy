import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_PATH = os.path.join(BASE_DIR, "data", "testD")
STIM_PATH = os.path.join(DATA_PATH, "stimuli")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "testD")

os.makedirs(OUTPUT_DIR, exist_ok=True)

PARTICIPANTS = ["Participant61"]

# ============================================================
# SCATTER SETTINGS
# ============================================================

PLOT = {
    "x1": 0.38,
    "x2": 0.63,
    "y1": 0.285,
    "y2": 0.69
}

GRID_COLS = 4
GRID_ROWS = 4

# ============================================================
# UI AOIs
# ============================================================

def get_ui_aois():
    return [
        {"name": "question", "x1": 0.27, "y1": 0.02, "x2": 0.73, "y2": 0.17, "type": "relevant"},
        {"name": "answers",  "x1": 0.32, "y1": 0.74, "x2": 0.68, "y2": 0.96, "type": "relevant"},
        {"name": "background", "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0, "type": "irrelevant"},
    ]

# ============================================================
# HELPERS
# ============================================================

def extract_question_id(event_value):
    m = re.search(r"(\d+)", str(event_value))
    return int(m.group(1)) if m else None


def get_fixations_for_question(df, question_label):

    if "Recording timestamp" in df.columns:
        ts_col = "Recording timestamp"
        x_col = "Fixation point X (MCSnorm)"
        y_col = "Fixation point Y (MCSnorm)"
        dur_col = "Gaze event duration"
    else:
        ts_col = "Recording timestamp [ms]"
        x_col = "Fixation point X [MCS norm]"
        y_col = "Fixation point Y [MCS norm]"
        dur_col = "Gaze event duration [ms]"

    url_events = df[
        (df["Event"].isin(["URLStart", "URLEnd"])) &
        (df["Event value"] == question_label)
        ]

    if len(url_events) < 2:
        return None, None

    t_start = url_events[url_events["Event"] == "URLStart"][ts_col].min()
    t_end = url_events[url_events["Event"] == "URLEnd"][ts_col].max()

    duration = (t_end - t_start) / 1000

    fix = df[
        (df["Eye movement type"] == "Fixation") &
        (df[ts_col].between(t_start, t_end))
        ].copy()

    fix = fix[
        (fix[x_col].between(0, 1)) &
        (fix[y_col].between(0, 1))
        ]

    fix = fix.rename(columns={
        ts_col: "Recording timestamp",
        x_col: "Fixation point X",
        y_col: "Fixation point Y",
        dur_col: "Gaze event duration"
    })

    return fix.sort_values("Recording timestamp").reset_index(drop=True), duration

# ============================================================
# AOI MAPPING
# ============================================================

def map_aois(fix):

    names, types = [], []

    dx = (PLOT["x2"] - PLOT["x1"]) / GRID_COLS
    dy = (PLOT["y2"] - PLOT["y1"]) / GRID_ROWS

    for _, row in fix.iterrows():

        x = row["Fixation point X"]
        y = row["Fixation point Y"]

        if PLOT["x1"] <= x <= PLOT["x2"] and PLOT["y1"] <= y <= PLOT["y2"]:

            col = int((x - PLOT["x1"]) / dx)
            row_idx = int((y - PLOT["y1"]) / dy)

            col = min(col, GRID_COLS - 1)
            row_idx = min(row_idx, GRID_ROWS - 1)

            names.append(f"grid_{row_idx}_{col}")
            types.append("relevant")
            continue

        assigned = False
        for aoi in get_ui_aois():
            if aoi["name"] == "background":
                continue

            if aoi["x1"] <= x <= aoi["x2"] and aoi["y1"] <= y <= aoi["y2"]:
                names.append(aoi["name"])
                types.append(aoi["type"])
                assigned = True
                break

        if assigned:
            continue

        names.append("background")
        types.append("irrelevant")

    fix["AOI"] = names
    fix["AOI_type"] = types

    return fix

# ============================================================
# METRICS
# ============================================================

def compute_metrics(fix, duration):

    dwell = fix.groupby("AOI")["Gaze event duration"].sum()
    total_dwell = dwell.sum()

    def ttff(target):
        subset = fix[fix["AOI"] == target]
        if len(subset) == 0:
            return np.nan
        return (subset["Recording timestamp"].iloc[0] - fix["Recording timestamp"].iloc[0]) / 1000

    seq = fix["AOI"].tolist()
    seq_clean = [seq[i] for i in range(len(seq)) if i == 0 or seq[i] != seq[i - 1]]

    transitions = list(zip(seq_clean[:-1], seq_clean[1:]))
    trans_df = pd.DataFrame(transitions, columns=["from", "to"])
    matrix = pd.crosstab(trans_df["from"], trans_df["to"]) if len(trans_df) > 0 else pd.DataFrame()

    irrelevant = fix[fix["AOI_type"] == "irrelevant"]["Gaze event duration"].sum()
    irrelevant_ratio = irrelevant / total_dwell if total_dwell > 0 else 0

    return {
        "TTFF_answers": ttff("answers"),
        "Transitions": len(seq_clean),
        "Irrelevant_Ratio": irrelevant_ratio,
        "Transition_Matrix": matrix
    }

# ============================================================
# PNG OVERLAY
# ============================================================

def plot_aoi_overlay(participant):

    img_path = os.path.join(STIM_PATH, "Question12.png")
    img = Image.open(img_path)
    w, h = img.size

    plt.figure(figsize=(6, 9))
    plt.imshow(img)

    x1, x2 = PLOT["x1"], PLOT["x2"]
    y1, y2 = PLOT["y1"], PLOT["y2"]

    dx = (x2 - x1) / GRID_COLS
    dy = (y2 - y1) / GRID_ROWS

    for i in range(GRID_COLS + 1):
        x = (x1 + i * dx) * w
        plt.plot([x, x], [y1 * h, y2 * h], linestyle=(0, (4, 4)), color="#1f4aff")

    for j in range(GRID_ROWS + 1):
        y = (y1 + j * dy) * h
        plt.plot([x1 * w, x2 * w], [y, y], linestyle=(0, (4, 4)), color="#1f4aff")

    for aoi in get_ui_aois():
        rect = plt.Rectangle(
            (aoi["x1"] * w, aoi["y1"] * h),
            (aoi["x2"] - aoi["x1"]) * w,
            (aoi["y2"] - aoi["y1"]) * h,
            linewidth=2,
            edgecolor="blue" if aoi["type"] == "relevant" else "gray",
            facecolor="none",
            linestyle="--"
        )
        plt.gca().add_patch(rect)

    plt.axis("off")

    save_path = os.path.join(OUTPUT_DIR, f"aoi_overlay_q12_{participant}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

# ============================================================
# MARKDOWN
# ============================================================

def create_markdown(participant, metrics):

    text = f"""# Auswertung – {participant} (Frage 12)

TTFF_answers: {metrics['TTFF_answers']:.2f}s  
Transitions: {metrics['Transitions']}  
Irrelevant Ratio: {metrics['Irrelevant_Ratio']*100:.1f}%

Das Verhalten deutet auf eine explorative Strategie hin.
"""

    with open(os.path.join(OUTPUT_DIR, f"{participant}_q12.md"), "w", encoding="utf-8") as f:
        f.write(text)

# ============================================================
# ANALYSIS
# ============================================================

def run_analysis(participant):

    file_path = os.path.join(DATA_PATH, f"{participant}.tsv")
    df = pd.read_csv(file_path, sep="\t", low_memory=False)

    question_labels = df[df["Event"] == "URLStart"]["Event value"].dropna().unique()

    results = []
    matrices = []

    for q_label in question_labels:

        qid = extract_question_id(q_label)

        if qid != 12:
            continue

        fix, duration = get_fixations_for_question(df, q_label)
        if fix is None:
            continue

        fix = map_aois(fix)

        metrics = compute_metrics(fix, duration)

        create_markdown(participant, metrics)

        matrix = metrics.pop("Transition_Matrix")
        if not matrix.empty:
            matrix["Participant"] = participant
            matrix["Question"] = qid
            matrices.append(matrix)

        row = {"Participant": participant, "Question": qid}
        row.update(metrics)
        results.append(row)

    return pd.DataFrame(results), pd.concat(matrices) if len(matrices) > 0 else None

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    all_results = []
    all_matrices = []

    for p in PARTICIPANTS:

        plot_aoi_overlay(p)

        df_res, df_mat = run_analysis(p)

        if df_res is not None and not df_res.empty:
            all_results.append(df_res)

        if df_mat is not None and not df_mat.empty:
            all_matrices.append(df_mat)

    if len(all_results) > 0:
        pd.concat(all_results).to_csv(os.path.join(OUTPUT_DIR, "aoi_metrics_q12.csv"), index=False)

    if len(all_matrices) > 0:
        pd.concat(all_matrices).to_csv(os.path.join(OUTPUT_DIR, "transition_matrix_q12.csv"), index=False)

    print("\n FINAL: Alles läuft für alle Participants")