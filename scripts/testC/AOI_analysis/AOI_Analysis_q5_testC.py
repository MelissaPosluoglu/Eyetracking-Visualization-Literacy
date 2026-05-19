import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Wedge

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
#  PIE SETTINGS
# ============================================================
# Normalized center position of the pie chart
CENTER_X = 0.5
CENTER_Y = 0.44

# Normalized radius of the pie chart
RADIUS = 0.127

PIE_SEGMENTS = []
current_angle = 0

def add_segment(name, size):
    # Add one pie segment with start and end angle
    global current_angle
    start = current_angle
    end = current_angle + size
    PIE_SEGMENTS.append({"name": name, "start": start, "end": end})
    current_angle = end

# Define pie chart segments
add_segment("others", 115)
add_segment("samsung", 64)
add_segment("xiaomi", 56)
add_segment("apple", 54)
add_segment("oppo", 37)
add_segment("vivo", 33)

# ============================================================
# UI AOIs
# ============================================================

def get_ui_aois():
    # Define non-pie AOIs such as question and answers
    return [
        {"name": "question", "x1": 0.26, "y1": 0.03, "x2": 0.74, "y2": 0.20, "type": "relevant"},
        {"name": "answers",  "x1": 0.28, "y1": 0.67, "x2": 0.72, "y2": 0.95, "type": "relevant"},
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
    # Get start and end events for the selected question

    url_events = df[
        (df["Event"].isin(["URLStart", "URLEnd"])) &
        (df["Event value"] == question_label)
        ]

    if len(url_events) < 2:
        return None, None

    # Define question time window
    t_start = url_events[url_events["Event"] == "URLStart"]["Recording timestamp"].min()
    t_end_real = url_events[url_events["Event"] == "URLEnd"]["Recording timestamp"].max()

    if pd.isna(t_end_real) or (t_end_real - t_start) > 30000:
        t_end_real = t_start + 25000

    duration = (t_end_real - t_start)

    # Select fixations within this time window
    fix = df[
        (df["Eye movement type"] == "Fixation") &
        (df["Recording timestamp"].between(t_start, t_end_real))
        ].copy()

    # Keep only valid normalized fixation coordinates
    fix = fix[
        (fix["Fixation point X (MCSnorm)"].between(0, 1)) &
        (fix["Fixation point Y (MCSnorm)"].between(0, 1))
        ]

    # Sort fixations by time
    return fix.sort_values("Recording timestamp").reset_index(drop=True), duration

# ============================================================
# ANGLE CALCULATION
# ============================================================

def get_angle(x, y):
    # Calculate distance from pie center
    dx = x - CENTER_X
    dy = CENTER_Y - y

    # Convert position to angle
    angle = np.degrees(np.arctan2(dy, dx))
    if angle < 0:
        angle += 360

    return (360 - angle + 90) % 360

# ============================================================
# AOI MAPPING
# ============================================================

def map_aois(fix):

    # Store assigned AOI names and types
    names, types = [], []

    for _, row in fix.iterrows():

        x = row["Fixation point X (MCSnorm)"]
        y = row["Fixation point Y (MCSnorm)"]

        assigned = False

        # Calculate distance from fixation to pie center
        dist = np.sqrt((x - CENTER_X) ** 2 + (y - CENTER_Y) ** 2)

        # Check whether fixation lies inside the pie chart
        if dist <= RADIUS:
            angle = get_angle(x, y)

            # Assign fixation to matching pie segment
            for seg in PIE_SEGMENTS:
                if seg["start"] <= angle < seg["end"]:
                    names.append(seg["name"])
                    types.append("relevant")
                    assigned = True
                    break

        if assigned:
            continue

            # Background is handled at the end
        for aoi in get_ui_aois():
            if aoi["name"] == "background":
                continue

            # Check whether fixation lies inside this AOI
            if aoi["x1"] <= x <= aoi["x2"] and aoi["y1"] <= y <= aoi["y2"]:
                names.append(aoi["name"])
                types.append(aoi["type"])
                assigned = True
                break

        if assigned:
            continue
        # 3. BACKGROUND
        names.append("background")
        types.append("irrelevant")

    # Add AOI information to fixation dataframe
    fix["AOI"] = names
    fix["AOI_type"] = types

    return fix

# ============================================================
# METRICS
# ============================================================

def compute_metrics(fix, duration):

    # Sum dwell time per AOI
    dwell = fix.groupby("AOI")["Gaze event duration"].sum()
    total_dwell = dwell.sum()

    if total_dwell > duration:
        scale = duration / total_dwell
        dwell = dwell * scale
        total_dwell = dwell.sum()

    dwell_ratio = {k: v / total_dwell for k, v in dwell.items()} if total_dwell > 0 else {}

    def ttff(target):

        # Calculate time to first fixation for one AOI
        subset = fix[fix["AOI"] == target]
        if len(subset) == 0:
            return np.nan
        return subset["Recording timestamp"].iloc[0] - fix["Recording timestamp"].iloc[0]

    # Create AOI sequence
    seq = fix["AOI"].tolist()

    # Remove consecutive duplicates
    seq_clean = [seq[i] for i in range(len(seq)) if i == 0 or seq[i] != seq[i - 1]]

    # Build transition matrix
    transitions = list(zip(seq_clean[:-1], seq_clean[1:]))
    trans_df = pd.DataFrame(transitions, columns=["from", "to"])
    matrix = pd.crosstab(trans_df["from"], trans_df["to"]) if len(trans_df) > 0 else pd.DataFrame()

    transitions_per_sec = len(seq_clean) / (duration / 1000) if duration > 0 else 0

    # Calculate irrelevant dwell ratio
    irrelevant = dwell.get("background", 0)
    irrelevant_ratio = irrelevant / total_dwell if total_dwell > 0 else 0

    return {
        "TTFF_samsung": ttff("samsung"),
        "TTFF_answers": ttff("answers"),

        "Dwell_samsung": dwell.get("samsung", 0),
        "Dwell_answers": dwell.get("answers", 0),

        "Dwell_ratio_samsung": dwell_ratio.get("samsung", 0),
        "Dwell_ratio_answers": dwell_ratio.get("answers", 0),

        "Transitions": len(seq_clean),
        "Transitions_per_sec": transitions_per_sec,
        "Sequence_length": len(seq_clean),
        "First_AOI": next((a for a in seq_clean if a != "background"), None),

        "Irrelevant_Ratio": irrelevant_ratio,

        "Transition_Matrix": matrix
    }

# ============================================================
# AOI OVERLAY
# ============================================================

def plot_aoi_overlay():

    img_path = os.path.join(STIM_PATH, "Question5.png")
    img = Image.open(img_path)
    w, h = img.size

    plt.figure(figsize=(6, 9))
    plt.imshow(img)

    for seg in PIE_SEGMENTS:

        # Convert stored angles for matplotlib wedge
        start = (seg["start"] - 90) % 360
        end = (seg["end"] - 90) % 360

        # Draw pie segment outline
        wedge = Wedge(
            (CENTER_X * w, CENTER_Y * h),
            RADIUS * w,
            start,
            end,
            facecolor="none",
            edgecolor="#1f4aff",
            linewidth=1,
            linestyle=(0, (4, 4))
        )
        plt.gca().add_patch(wedge)

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

    save_path = os.path.join(OUTPUT_DIR, "aoi_overlay_q5.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("✔ AOI Overlay with pie and UI AOIs saved:", save_path)

# ============================================================
# ANALYSIS
# ============================================================

def run_analysis(participant):

    file_path = os.path.join(DATA_PATH, f"{participant}.tsv")
    df = pd.read_csv(file_path, sep="\t", low_memory=False)
    df = normalize_columns(df)

    question_labels = df[df["Event"] == "URLStart"]["Event value"].dropna().unique()

    results = []
    matrices = []

    for q_label in question_labels:

        # Analyze only Question 5
        qid = extract_question_id(q_label)
        if qid != 5:
            continue

        fix, duration = get_fixations_for_question(df, q_label)
        if fix is None:
            continue

        fix = map_aois(fix)

        # Print debug information
        print("\n==============================")
        print("AOI Counts fuer", participant)
        print(fix["AOI"].value_counts())

        print("\nDwell Time pro AOI:")
        print(fix.groupby("AOI")["Gaze event duration"].sum())
        print("==============================\n")

        metrics = compute_metrics(fix, duration)

        # Store transition matrix separately
        matrix = metrics.pop("Transition_Matrix")
        if not matrix.empty:
            matrix["Participant"] = participant
            matrix["Question"] = qid
            matrices.append(matrix)

        # Store metric row
        row = {"Participant": participant, "Question": qid}
        row.update(metrics)
        results.append(row)


    # Return results and matrices
    return pd.DataFrame(results), pd.concat(matrices) if len(matrices) > 0 else None

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    plot_aoi_overlay()

    all_results = []
    all_matrices = []

    for p in PARTICIPANTS:
        df_res, df_mat = run_analysis(p)

        if df_res is not None and not df_res.empty:
            all_results.append(df_res)

        if df_mat is not None and not df_mat.empty:
            all_matrices.append(df_mat)

    if len(all_results) > 0:
        pd.concat(all_results).to_csv(
            os.path.join(OUTPUT_DIR, "aoi_metrics_q5.csv"),
            index=False
        )

    if len(all_matrices) > 0:
        pd.concat(all_matrices).to_csv(
            os.path.join(OUTPUT_DIR, "transition_matrix_q5.csv"),
            index=False
        )

    print("\n✔ FINAL: PIE AOI analysis completed successfully")