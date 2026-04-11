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
DATA_PATH = os.path.join(BASE_DIR, "data", "testA")
STIM_PATH = os.path.join(DATA_PATH, "stimuli")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "testA")

os.makedirs(OUTPUT_DIR, exist_ok=True)

PARTICIPANTS = ["Participant23"]

# ============================================================
# AOIs (UNVERÄNDERT)
# ============================================================

def get_aois_q11():
    return [

        {"name": "question", "x1": 0.30, "y1": 0.06, "x2": 0.70, "y2": 0.16, "type": "relevant"},

        {"name": "year_2012",
         "x1": 0.515, "x2": 0.545,
         "y1": 0.24, "y2": 0.60,
         "type": "relevant"},

        {"name": "Rest",
         "x1": 0.415, "x2": 0.515,
         "y1": 0.22, "y2": 0.60,
         "type": "relevant"},

        {"name": "Rest1",
         "x1": 0.545, "x2": 0.602,
         "y1": 0.22, "y2": 0.60,
         "type": "relevant"},

        {"name": "answers",
         "x1": 0.28, "y1": 0.67,
         "x2": 0.72, "y2": 0.95,
         "type": "relevant"},

        {"name": "background",
         "x1": 0.0, "y1": 0.0,
         "x2": 1.0, "y2": 1.0,
         "type": "irrelevant"},
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

    t_start = url_events[url_events["Event"] == "URLStart"]["Recording timestamp [ms]"].min()
    t_end   = url_events[url_events["Event"] == "URLEnd"]["Recording timestamp [ms]"].max()

    duration = (t_end - t_start) / 1000

    fix = df[
        (df["Eye movement type"] == "Fixation") &
        (df["Recording timestamp [ms]"].between(t_start, t_end))
        ].copy()

    fix = fix[
        (fix["Fixation point X [MCS norm]"].between(0, 1)) &
        (fix["Fixation point Y [MCS norm]"].between(0, 1))
        ]

    return fix.sort_values("Recording timestamp [ms]").reset_index(drop=True), duration

# ============================================================
# AOI MAPPING
# ============================================================

def map_aois(fix, aois):

    priority = [
        "year_2012",
        "Rest",
        "Rest1",
        "answers",
        "question",
        "background"
    ]

    aois_sorted = sorted(
        aois,
        key=lambda a: priority.index(a["name"]) if a["name"] in priority else 999
    )

    aoi_names = []
    aoi_types = []

    for _, row in fix.iterrows():
        x = row["Fixation point X [MCS norm]"]
        y = row["Fixation point Y [MCS norm]"]

        assigned = False

        for aoi in aois_sorted:
            if aoi["x1"] <= x <= aoi["x2"] and aoi["y1"] <= y <= aoi["y2"]:
                aoi_names.append(aoi["name"])
                aoi_types.append(aoi["type"])
                assigned = True
                break

        if not assigned:
            aoi_names.append("background")
            aoi_types.append("irrelevant")

    fix["AOI"] = aoi_names
    fix["AOI_type"] = aoi_types

    return fix

# ============================================================
# METRICS
# ============================================================

def compute_metrics(fix, duration):

    dwell = fix.groupby("AOI")["Gaze event duration [ms]"].sum()
    total_dwell = dwell.sum()

    def ttff(target):
        subset = fix[fix["AOI"] == target]
        if len(subset) == 0:
            return np.nan
        return (subset["Recording timestamp [ms]"].iloc[0] - fix["Recording timestamp [ms]"].iloc[0]) / 1000

    seq = fix["AOI"].tolist()
    seq_clean = [seq[i] for i in range(len(seq)) if i == 0 or seq[i] != seq[i - 1]]

    transitions = list(zip(seq_clean[:-1], seq_clean[1:]))
    trans_df = pd.DataFrame(transitions, columns=["from", "to"])
    matrix = pd.crosstab(trans_df["from"], trans_df["to"])

    transitions_per_sec = len(seq_clean) / duration if duration > 0 else 0

    irrelevant = fix[fix["AOI_type"] == "irrelevant"]["Gaze event duration [ms]"].sum()
    irrelevant_ratio = irrelevant / total_dwell if total_dwell > 0 else 0

    return {
        "TTFF_2012": ttff("year_2012"),
        "TTFF_answers": ttff("answers"),

        "Dwell_2012": dwell.get("year_2012", 0),
        "Dwell_answers": dwell.get("answers", 0),

        "Transitions": len(seq_clean),
        "Transitions_per_sec": transitions_per_sec,
        "Sequence_length": len(seq_clean),
        "First_AOI": next((a for a in seq_clean if a != "background"), None),

        "Irrelevant_Ratio": irrelevant_ratio,

        "Transition_Matrix": matrix
    }

# ============================================================
# AOI OVERLAY (Q8 STYLE 🔥)
# ============================================================

def plot_aoi_overlay():

    img_path = os.path.join(STIM_PATH, "Question11.png")

    if not os.path.exists(img_path):
        print("Stimulus fehlt:", img_path)
        return

    img = Image.open(img_path)
    w, h = img.size

    aois = get_aois_q11()

    plt.figure(figsize=(6, 9))
    plt.imshow(img)

    for aoi in aois:

        # 🎯 Farben
        if aoi["name"] == "year_2012":
            color = "#ff2d2d"
            lw = 1.2
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

        # =========================
        # 🔥 SAUBERE LABEL POSITIONEN
        # =========================

        center_y = y1 + height / 2

        if aoi["name"] == "year_2012":
            text_x = x1 + width + 20
            text_y = center_y

        elif aoi["name"] in ["Rest", "Rest1"]:
            text_x = x1 + width + 15
            text_y = center_y -40

        elif aoi["name"] == "question":
            text_x = x1 + width + 30
            text_y = y1 + 10

        elif aoi["name"] == "answers":
            text_x = x1 + width + 30
            text_y = center_y

        elif aoi["name"] == "background":
            # ❗ KEINE LINIE + kein Chaos
            plt.text(10, 20, "background", fontsize=6, color="gray")
            continue

        else:
            text_x = x1 + width + 10
            text_y = center_y

        # 🔥 Linie NUR für relevante AOIs (kein background!)
        if aoi["name"] != "background":
            plt.plot(
                [x1 + width, text_x],
                [center_y, text_y],
                color=color,
                linewidth=0.6
            )

        plt.text(
            text_x,
            text_y,
            aoi["name"],
            color=color,
            fontsize=6,
            ha="left",
            va="center"
        )

    plt.axis("off")

    save_path = os.path.join(OUTPUT_DIR, "aoi_overlay_q11.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("✔ AOI Overlay gespeichert:", save_path)

# ============================================================
# MAIN
# ============================================================

def run_analysis(participant):

    file_path = os.path.join(DATA_PATH, f"{participant}.tsv")
    df = pd.read_csv(file_path, sep="\t", low_memory=False)

    question_labels = df[df["Event"] == "URLStart"]["Event value"].dropna().unique()

    results = []
    matrices = []

    for q_label in question_labels:

        qid = extract_question_id(q_label)

        if qid != 11:
            continue

        fix, duration = get_fixations_for_question(df, q_label)
        if fix is None:
            continue

        aois = get_aois_q11()
        fix = map_aois(fix, aois)

        # 🔥 IDENTISCHE CONSOLE AUSGABE WIE Q5
        print("\n==============================")
        print("AOI Counts fuer", participant)
        print(fix["AOI"].value_counts())

        print("\nDwell Time pro AOI:")
        print(fix.groupby("AOI")["Gaze event duration [ms]"].sum())
        print("==============================\n")

        metrics = compute_metrics(fix, duration)

        # 🔥 Transition Matrix speichern
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
            os.path.join(OUTPUT_DIR, "aoi_metrics_q11.csv"),
            index=False
        )

    if len(all_matrices) > 0:
        pd.concat(all_matrices).to_csv(
            os.path.join(OUTPUT_DIR, "transition_matrix_q11.csv"),
            index=False
        )

    print("\n✔ FINAL: Q11 AOI Analyse komplett & korrekt")