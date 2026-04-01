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
DATA_PATH = os.path.join(BASE_DIR, "data", "testA")
STIM_PATH = os.path.join(DATA_PATH, "stimuli")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "testA")

os.makedirs(OUTPUT_DIR, exist_ok=True)

PARTICIPANTS = ["Participant1"]

# ============================================================
# 🔥 FINAL PIE SETTINGS (FEINJUSTIERT)
# ============================================================

CENTER_X = 0.5   # leicht nach links
CENTER_Y = 0.44
RADIUS   = 0.127

# Winkel (0° = oben, durch Rotation korrigiert)
PIE_SEGMENTS = []

current_angle = 0

def add_segment(name, size):
    global current_angle
    start = current_angle
    end = current_angle + size
    PIE_SEGMENTS.append({"name": name, "start": start, "end": end})
    current_angle = end

# Reihenfolge im Uhrzeigersinn (wie im Bild!)
add_segment("others", 115)
add_segment("samsung", 64)
add_segment("xiaomi", 56)
add_segment("apple", 54)
add_segment("oppo", 37)
add_segment("vivo", 33)  # leicht angepasst damit = 360

# ============================================================
# UI AOIs
# ============================================================

def get_ui_aois():
    return [
        {"name": "question",   "x1": 0.26, "y1": 0.03, "x2": 0.74, "y2": 0.20, "type": "relevant"},
        {"name": "answers",    "x1": 0.28, "y1": 0.67, "x2": 0.72, "y2": 0.95, "type": "relevant"},
        {"name": "background", "x1": 0.0,  "y1": 0.0,  "x2": 1.0,  "y2": 1.0,  "type": "irrelevant"},
    ]

# ============================================================
# HELPERS
# ============================================================

def extract_question_id(event_value):
    m = re.search(r"Question\s+(\d+)", str(event_value))
    return int(m.group(1)) if m else None


def get_fixations_for_question(df, question_label):

    url_events = df[
        (df["Event"].isin(["URLStart", "URLEnd"])) &
        (df["Event value"] == question_label)
        ]

    if len(url_events) < 2:
        return None, None

    t_start = url_events[url_events["Event"] == "URLStart"]["Recording timestamp"].min()
    t_end   = url_events[url_events["Event"] == "URLEnd"]["Recording timestamp"].max()

    duration = (t_end - t_start) / 1000

    fix = df[
        (df["Eye movement type"] == "Fixation") &
        (df["Recording timestamp"].between(t_start, t_end))
        ].copy()

    fix = fix[
        (fix["Fixation point X (MCSnorm)"].between(0, 1)) &
        (fix["Fixation point Y (MCSnorm)"].between(0, 1))
        ]

    return fix.sort_values("Recording timestamp").reset_index(drop=True), duration

# ============================================================
# 🔥 ANGLE FUNCTION (MIT KORREKTER ROTATION)
# ============================================================

def get_angle(x, y):
    dx = x - CENTER_X
    dy = CENTER_Y - y

    angle = np.degrees(np.arctan2(dy, dx))
    if angle < 0:
        angle += 360

    # 🔥 WICHTIG: Richtung drehen → Uhrzeigersinn
    angle = (360 - angle + 90) % 360

    return angle

# ============================================================
# AOI MAPPING (ECHTE PIE SEGMENTE)
# ============================================================

def map_aois(fix):

    names, types = [], []

    for _, row in fix.iterrows():

        x = row["Fixation point X (MCSnorm)"]
        y = row["Fixation point Y (MCSnorm)"]

        assigned = False

        # ==================================================
        # 1. PIE SEGMENTS (HÖCHSTE PRIORITÄT)
        # ==================================================
        dist = np.sqrt((x - CENTER_X)**2 + (y - CENTER_Y)**2)

        if dist <= RADIUS:

            angle = get_angle(x, y)

            for seg in PIE_SEGMENTS:
                if seg["start"] <= angle < seg["end"]:
                    names.append(seg["name"])
                    types.append("relevant")
                    assigned = True
                    break

        if assigned:
            continue

        # ==================================================
        # 2. UI AOIs (OHNE BACKGROUND!)
        # ==================================================
        for aoi in get_ui_aois():

            if aoi["name"] == "background":
                continue  # ❗ ganz wichtig

            if aoi["x1"] <= x <= aoi["x2"] and aoi["y1"] <= y <= aoi["y2"]:
                names.append(aoi["name"])
                types.append(aoi["type"])
                assigned = True
                break

        if assigned:
            continue

        # ==================================================
        # 3. BACKGROUND (FALLBACK)
        # ==================================================
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
    seq_clean = [seq[i] for i in range(len(seq)) if i == 0 or seq[i] != seq[i-1]]

    transitions = list(zip(seq_clean[:-1], seq_clean[1:]))
    trans_df = pd.DataFrame(transitions, columns=["from", "to"])
    matrix = pd.crosstab(trans_df["from"], trans_df["to"])

    irrelevant = fix[fix["AOI_type"] == "irrelevant"]["Gaze event duration"].sum()
    irrelevant_ratio = irrelevant / total_dwell if total_dwell > 0 else 0

    return {
        "TTFF_samsung": ttff("samsung"),
        "TTFF_answers": ttff("answers"),
        "Dwell_samsung": dwell.get("samsung", 0),
        "Transitions": len(seq_clean),
        "Irrelevant_Ratio": irrelevant_ratio,
        "Transition_Matrix": matrix
    }

# ============================================================
# OVERLAY (VISUAL DEBUG)
# ============================================================

def plot_aoi_overlay():

    img_path = os.path.join(STIM_PATH, "Question5.png")
    img = Image.open(img_path)
    w, h = img.size

    plt.figure(figsize=(6, 9))
    plt.imshow(img)

    # 🔥 PIE SEGMENTS + LABELS
    for seg in PIE_SEGMENTS:

        start = (seg["start"] - 90) % 360
        end   = (seg["end"] - 90) % 360

        wedge = Wedge(
            (CENTER_X * w, CENTER_Y * h),
            RADIUS * w,
            start,
            end,
            facecolor="none",
            edgecolor="#1f4aff",
            linewidth=1,            # 🔥 dünner (vorher 2)
            linestyle=(0, (4, 4))   # 🔥 feinere Striche
        )
        plt.gca().add_patch(wedge)

        # 🔥 Mittelpunkt des Segments
        mid = (seg["start"] + seg["end"]) / 2

        # 👉 gleiche Rotation wie beim Wedge!
        mid_rot = (mid - 90) % 360
        rad = np.radians(mid_rot)

        # Punkt am Rand des Kreises
        x_edge = CENTER_X + RADIUS * np.cos(rad)
        y_edge = CENTER_Y + RADIUS * np.sin(rad)

        # 🔥 Label außerhalb platzieren
        label_r = RADIUS * 1.35
        x_text = CENTER_X + label_r * np.cos(rad)
        y_text = CENTER_Y + label_r * np.sin(rad)


        # 🔥 Text
        plt.text(
            x_text * w,
            y_text * h,
            seg["name"],
            ha="center",
            va="center",
            fontsize=6,
            color="#1f4aff"
        )

    # 🔥 UI AOIs (Rechtecke)
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

        # Label für AOI
        plt.text(
            aoi["x1"] * w,
            aoi["y1"] * h - 10,
            aoi["name"],
            color="blue",
            fontsize=8
        )

    plt.axis("off")

    save_path = os.path.join(OUTPUT_DIR, "aoi_overlay_q5.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("✔ AOI Overlay mit Labels gespeichert:", save_path)

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

        if qid != 5:
            continue

        fix, duration = get_fixations_for_question(df, q_label)
        if fix is None:
            continue

        # ==================================================
        # 🔥 HIER IST DEIN DEBUG BLOCK
        # ==================================================
        fix = map_aois(fix)

        print("\n==============================")
        print("AOI Counts für", participant)
        print(fix["AOI"].value_counts())

        print("\nDwell Time pro AOI:")
        print(fix.groupby("AOI")["Gaze event duration"].sum())
        print("==============================\n")
        # ==================================================

        metrics = compute_metrics(fix, duration)

        matrix = metrics.pop("Transition_Matrix")
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

    print("\n✔ FINAL: PIE AOI Analyse komplett & korrekt")