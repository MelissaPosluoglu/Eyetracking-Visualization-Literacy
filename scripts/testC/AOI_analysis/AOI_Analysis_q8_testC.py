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
DATA_PATH = os.path.join(BASE_DIR, "data", "testC")
STIM_PATH = os.path.join(DATA_PATH, "stimuli")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "testC")

os.makedirs(OUTPUT_DIR, exist_ok=True)

PARTICIPANTS = ["Participant51"]

# ============================================================
# AOIs (Q8 FINAL)
# ============================================================

def get_aois_q8():
    return [
        {"name": "question", "x1": 0.31, "y1": 0.03, "x2": 0.68, "y2": 0.165, "type": "relevant"},
        {"name": "price_50", "x1": 0.395, "y1": 0.23, "x2": 0.414, "y2": 0.27, "type": "relevant"},
        {"name": "month_jan", "x1": 0.415, "y1": 0.17, "x2": 0.425, "y2": 0.6, "type": "relevant"},
        {"name": "month_feb", "x1": 0.427, "y1": 0.17, "x2": 0.44, "y2": 0.6, "type": "relevant"},
        {"name": "month_rest", "x1": 0.445, "y1": 0.17, "x2": 0.65, "y2": 0.6, "type": "relevant"},
        {"name": "answers", "x1": 0.28, "y1": 0.64, "x2": 0.72, "y2": 0.92, "type": "relevant"},
        {"name": "background", "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0, "type": "irrelevant"},
    ]

# ============================================================
# HELPERS
# ============================================================

def normalize_columns(df):
    df = df.copy()
    df.columns = df.columns.str.strip()

    rename_map = {}
    if "Recording timestamp [ms]" in df.columns:
        rename_map["Recording timestamp [ms]"] = "Recording timestamp"
    if "Gaze event duration [ms]" in df.columns:
        rename_map["Gaze event duration [ms]"] = "Gaze event duration"
    if "Fixation point X [MCS norm]" in df.columns:
        rename_map["Fixation point X [MCS norm]"] = "Fixation point X (MCSnorm)"
    if "Fixation point Y [MCS norm]" in df.columns:
        rename_map["Fixation point Y [MCS norm]"] = "Fixation point Y (MCSnorm)"

    df = df.rename(columns=rename_map)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    return df

def extract_question_id(event_value):
    # Extract question number from event label
    m = re.search(r"Question\s+(\d+)", str(event_value))
    return int(m.group(1)) if m else None

def get_fixations_for_question(df, question_label):
    # Get URL start and end events for this question
    url_events = df[
        (df["Event"].isin(["URLStart", "URLEnd"])) &
        (df["Event value"] == question_label)
        ]

    if len(url_events) < 2:
        return None, None

    # Define question time window
    t_start = url_events[url_events["Event"] == "URLStart"]["Recording timestamp"].min()
    t_end = url_events[url_events["Event"] == "URLEnd"]["Recording timestamp"].max()

    if pd.isna(t_end) or (t_end - t_start) > 30000:
        t_end = t_start + 25000

    # Convert duration to seconds
    duration = (t_end - t_start)

    # Select fixation data within this time window
    fix = df[
        (df["Eye movement type"] == "Fixation") &
        (df["Recording timestamp"].between(t_start, t_end))
        ].copy()

    # Keep only valid normalized coordinates
    fix = fix[
        (fix["Fixation point X (MCSnorm)"].between(0, 1)) &
        (fix["Fixation point Y (MCSnorm)"].between(0, 1))
        ]

    # Sort fixations by timestamp
    return fix.sort_values("Recording timestamp").reset_index(drop=True), duration

# ============================================================
# AOI MAPPING
# ============================================================

def map_aois(fix, aois):
    # Check smaller AOIs before the background AOI
    aois_sorted = sorted(
        aois,
        key=lambda a: (a["name"] == "background", (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]))
    )

    aoi_names = []
    aoi_types = []

    for _, row in fix.iterrows():
        x = row["Fixation point X (MCSnorm)"]
        y = row["Fixation point Y (MCSnorm)"]

        assigned = False

        # Assign each fixation to one AOI
        for aoi in aois_sorted:
            if aoi["x1"] <= x <= aoi["x2"] and aoi["y1"] <= y <= aoi["y2"]:
                aoi_names.append(aoi["name"])
                aoi_types.append(aoi["type"])
                assigned = True
                break

        if not assigned:
            aoi_names.append("background")
            aoi_types.append("irrelevant")

    # Add AOI information to fixation data
    fix["AOI"] = aoi_names
    fix["AOI_type"] = aoi_types

    return fix

# ============================================================
# METRICS
# ============================================================

def compute_metrics(fix, duration):
    # Sum dwell time per AOI
    dwell = fix.groupby("AOI")["Gaze event duration"].sum()
    total_dwell = dwell.sum()

    if total_dwell > duration and total_dwell > 0:
        scale = duration / total_dwell
        dwell = dwell * scale
        total_dwell = dwell.sum()

    # Calculate dwell ratios
    dwell_ratio = {k: v / total_dwell for k, v in dwell.items()} if total_dwell > 0 else {}

    def ttff(target):
        subset = fix[fix["AOI"] == target]
        if len(subset) == 0:
            return np.nan
        return subset["Recording timestamp"].iloc[0] - fix["Recording timestamp"].iloc[0]

    # Create AOI sequence
    seq = fix["AOI"].tolist()
    # Remove consecutive duplicate AOIs
    seq_clean = [seq[i] for i in range(len(seq)) if i == 0 or seq[i] != seq[i - 1]]

    if len(seq_clean) >= 2:
        # Build transition matrix
        transitions = list(zip(seq_clean[:-1], seq_clean[1:]))
        trans_df = pd.DataFrame(transitions, columns=["from", "to"])
        matrix = pd.crosstab(trans_df["from"], trans_df["to"])
    else:
        matrix = pd.DataFrame()

    # Calculate transitions per second
    transitions_count = max(len(seq_clean) - 1, 0)

    # Calculate irrelevant dwell ratio
    irrelevant = dwell.get("background", 0)
    irrelevant_ratio = irrelevant / total_dwell if total_dwell > 0 else 0

    return {
        "TTFF_feb": ttff("month_feb"),
        "TTFF_answers": ttff("answers"),

        "Dwell_feb": dwell.get("month_feb", 0),
        "Dwell_answers": dwell.get("answers", 0),

        "Dwell_ratio_feb": dwell_ratio.get("month_feb", 0),

        "Transitions": transitions_count,
        "Transitions_per_sec": transitions_count / (duration / 1000) if duration > 0 else 0,
        "Sequence_length": len(seq_clean),
        "First_AOI": next((a for a in seq_clean if a != "background"), None),

        "Irrelevant_Ratio": irrelevant_ratio,

        "Transition_Matrix": matrix
    }

# ============================================================
# AOI OVERLAY PNG
# ============================================================

def plot_aoi_overlay():
    img_path = os.path.join(STIM_PATH, "Question8.png")

    if not os.path.exists(img_path):
        print("Stimulus fehlt:", img_path)
        return

    img = Image.open(img_path)
    w, h = img.size
    aois = get_aois_q8()

    plt.figure(figsize=(6, 9))
    plt.imshow(img)

    for aoi in aois:
        if aoi["name"] == "month_feb":
            color = "#ff2d2d"
            lw = 0.9
        else:
            color = "#1f4aff" if aoi["type"] == "relevant" else "#999999"
            lw = 0.9

        x1 = aoi["x1"] * w
        y1 = aoi["y1"] * h
        width = (aoi["x2"] - aoi["x1"]) * w
        height = (aoi["y2"] - aoi["y1"]) * h

        rect = plt.Rectangle(
            (x1, y1),
            width,
            height,
            linewidth=lw,
            edgecolor=color,
            facecolor="none",
            linestyle=(0, (3, 3))
        )

        plt.gca().add_patch(rect)

        if aoi["name"] == "month_rest":
            text_x, text_y, ha, va = x1 + 3, y1 + 8, "left", "top"
        elif aoi["name"] in ["month_jan", "price_50"]:
            text_x, text_y, ha, va = x1 - 25, y1 + height / 2, "right", "center"
        elif aoi["name"] == "month_feb":
            text_x, text_y, ha, va = x1 + width / 2, y1 + height + 8, "center", "top"
        elif aoi["name"] in ["question", "answers", "background"]:
            text_x, text_y, ha, va = x1, y1 - 20, "left", "top"
        else:
            text_x, text_y, ha, va = x1 + 2, y1 - 3, "left", "top"

        plt.text(
            text_x,
            text_y,
            aoi["name"],
            color=color,
            fontsize=4,
            alpha=0.7,
            ha=ha,
            va=va
        )

    plt.axis("off")

    save_path = os.path.join(OUTPUT_DIR, "aoi_overlay_q8.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("✔ AOI Overlay gespeichert:", save_path)

# ============================================================
# MAIN
# ============================================================

def run_analysis(participant):
    file_path = os.path.join(DATA_PATH, f"{participant}.tsv")
    df = pd.read_csv(file_path, sep="\t", low_memory=False)
    df = normalize_columns(df)

    question_labels = df[df["Event"] == "URLStart"]["Event value"].dropna().unique()

    results = []
    matrices = []

    for q_label in question_labels:
        qid = extract_question_id(q_label)
        if qid != 8:
            continue

        fix, duration = get_fixations_for_question(df, q_label)
        if fix is None:
            continue

        fix = map_aois(fix, get_aois_q8())

        print("\n==============================")
        print(f"AOI Counts für {participant}")
        print(fix["AOI"].value_counts())

        print("\nDwell Time pro AOI:")
        print(fix.groupby("AOI")["Gaze event duration"].sum())
        print("==============================\n")

        metrics = compute_metrics(fix, duration)

        matrix = metrics.pop("Transition_Matrix")
        if not matrix.empty:
            matrix["Participant"] = participant
            matrix["Question"] = qid
            matrices.append(matrix)

        row = {"Participant": participant, "Question": qid}
        row.update(metrics)
        results.append(row)

    results_df = pd.DataFrame(results)
    matrices_df = pd.concat(matrices) if len(matrices) > 0 else pd.DataFrame()

    return results_df, matrices_df

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    # Create AOI overlay for visual inspection
    plot_aoi_overlay()

    all_results = []
    all_matrices = []

    for p in PARTICIPANTS:
        df_res, df_mat = run_analysis(p)
        all_results.append(df_res)

        if not df_mat.empty:
            all_matrices.append(df_mat)

    pd.concat(all_results).to_csv(os.path.join(OUTPUT_DIR, "aoi_metrics_q8.csv"), index=False)
    pd.concat(all_matrices).to_csv(os.path.join(OUTPUT_DIR, "transition_matrix_q8.csv"), index=False)

    print("\n✔ Alles fertig: CSV + AOI PNG")