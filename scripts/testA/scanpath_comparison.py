import os
import re
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# PDF (ReportLab)
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet

# ============================================================
# SETTINGS
# ============================================================

PARTICIPANTS = ["Participant5", "Participant9"]  # <- genau 2!
ANALYSIS_MIN_NORM = 0.002

# NUR Strategy-Metriken (ohne reine Intensität)
STRATEGY_METRICS_FOR_SELECTION = [
    "Vertical_Ratio",
    "Regression_Rate",
    "Directional_Entropy",
]

# Plot Settings
LINEWIDTH = 1.6
LINE_ALPHA = 0.9
FIX_SIZE = 10

# ============================================================
# ROBUSTER PROJEKTPFAD
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "testA")
STIM_PATH = os.path.join(DATA_PATH, "stimuli")
OUT_DIR = os.path.join(BASE_DIR, "results", "testA", "strategy_report_best_question")
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# HELPERS
# ============================================================

def extract_question_id(event_value: str):
    s = str(event_value)
    m = re.search(r"Question\s+(\d+)", s)
    return int(m.group(1)) if m else None


def get_fixations_for_question(df: pd.DataFrame, question_label: str):
    url_events = df[
        (df["Event"].isin(["URLStart", "URLEnd"])) &
        (df["Event value"] == question_label)
        ]
    if len(url_events) < 2:
        return None, None

    t_start = url_events[url_events["Event"] == "URLStart"]["Recording timestamp"].min()
    t_end   = url_events[url_events["Event"] == "URLEnd"]["Recording timestamp"].max()

    duration_sec = (t_end - t_start) / 1000.0
    if not np.isfinite(duration_sec) or duration_sec <= 0:
        return None, None

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
        return None, None

    return fix, duration_sec


def compute_strategy_metrics_from_fixations(fix: pd.DataFrame):
    dx = np.diff(fix["Fixation point X (MCSnorm)"].to_numpy())
    dy = np.diff(fix["Fixation point Y (MCSnorm)"].to_numpy())
    distances = np.sqrt(dx**2 + dy**2)

    valid = distances >= ANALYSIS_MIN_NORM
    dx, dy, distances = dx[valid], dy[valid], distances[valid]

    if len(distances) == 0:
        return None

    vertical_movement = np.sum(np.abs(dy))
    horizontal_movement = np.sum(np.abs(dx))
    vertical_ratio = vertical_movement / (vertical_movement + horizontal_movement + 1e-12)

    regression_rate = float(np.sum(dx < 0) / max(len(dx), 1))

    angles = np.arctan2(dy, dx)
    hist, _ = np.histogram(angles, bins=8)
    prob = hist / (np.sum(hist) + 1e-12)
    entropy = float(-np.sum(prob * np.log2(prob + 1e-12)))

    return distances, float(vertical_ratio), float(regression_rate), entropy


def analyze_participant(participant: str) -> pd.DataFrame:
    file_path = os.path.join(DATA_PATH, f"{participant}.tsv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"TSV nicht gefunden: {file_path}")

    df = pd.read_csv(file_path, sep="\t", low_memory=False)
    question_labels = df[df["Event"] == "URLStart"]["Event value"].dropna().unique()

    rows = []
    for q_label in question_labels:
        qid = extract_question_id(q_label)
        if qid is None:
            continue

        fix, duration_sec = get_fixations_for_question(df, q_label)
        if fix is None:
            continue

        metrics = compute_strategy_metrics_from_fixations(fix)
        if metrics is None:
            continue

        distances, vertical_ratio, regression_rate, entropy = metrics

        saccade_count = len(distances)
        scanpath_length_norm = float(np.sum(distances))

        saccades_per_min = saccade_count / (duration_sec / 60.0)
        scanpath_per_min = scanpath_length_norm / (duration_sec / 60.0)

        rows.append({
            "Participant": participant,
            "Question_ID": qid,
            "Question_Label": str(q_label),
            "Duration_sec": float(duration_sec),
            "Saccade_Count": int(saccade_count),
            "Saccades_per_min": float(saccades_per_min),
            "Scanpath_Length_norm": scanpath_length_norm,
            "Scanpath_per_min_norm": float(scanpath_per_min),
            "Vertical_Ratio": float(vertical_ratio),
            "Regression_Rate": float(regression_rate),
            "Directional_Entropy": float(entropy),
        })

    return pd.DataFrame(rows), df


def standardize_diff_score(common: pd.DataFrame, metric_cols: list):
    """
    Score = Summe über Metriken von |A-B| / std (über beide Teilnehmer + alle gemeinsamen Fragen)
    """
    score = np.zeros(len(common), dtype=float)
    for m in metric_cols:
        a = common[f"{m}_A"].to_numpy(dtype=float)
        b = common[f"{m}_B"].to_numpy(dtype=float)
        diff = np.abs(a - b)
        pool = np.concatenate([a, b])
        std = np.std(pool, ddof=1)

        if not np.isfinite(std) or std == 0:
            continue

        score += diff / std
    return score


def save_timecoded_scanpath_plot(participant: str, df_raw: pd.DataFrame, question_id: int, question_label: str, out_dir: str):
    img_path = os.path.join(STIM_PATH, f"Question{question_id}.png")
    if not os.path.exists(img_path):
        print(f"[WARN] Stimulus fehlt: {img_path} -> Plot übersprungen.")
        return None

    fix, _ = get_fixations_for_question(df_raw, question_label)
    if fix is None:
        print(f"[WARN] Nicht genug Fixationen für {participant}, Q{question_id}.")
        return None

    img = Image.open(img_path)
    w, h = img.size

    fix = fix.copy()
    fix["X_px"] = fix["Fixation point X (MCSnorm)"] * w
    fix["Y_px"] = fix["Fixation point Y (MCSnorm)"] * h

    n = len(fix)
    cmap = plt.cm.plasma

    plt.figure(figsize=(6, 9))
    plt.imshow(img)

    for i in range(n - 1):
        x1, y1 = fix.loc[i, ["X_px", "Y_px"]]
        x2, y2 = fix.loc[i + 1, ["X_px", "Y_px"]]

        dx = fix.loc[i + 1, "Fixation point X (MCSnorm)"] - fix.loc[i, "Fixation point X (MCSnorm)"]
        dy = fix.loc[i + 1, "Fixation point Y (MCSnorm)"] - fix.loc[i, "Fixation point Y (MCSnorm)"]
        dist_norm = math.sqrt(dx*dx + dy*dy)

        if dist_norm < ANALYSIS_MIN_NORM:
            continue

        color = cmap(i / max(n - 1, 1))
        plt.plot([x1, x2], [y1, y2], color=color, linewidth=LINEWIDTH, alpha=LINE_ALPHA)

    plt.scatter(fix["X_px"], fix["Y_px"],
                c=np.linspace(0, 1, n), cmap="plasma",
                s=FIX_SIZE, alpha=0.9)

    # Start/End markieren
    plt.scatter([fix.loc[0, "X_px"]], [fix.loc[0, "Y_px"]], s=60, marker="o")
    plt.scatter([fix.loc[n-1, "X_px"]], [fix.loc[n-1, "Y_px"]], s=60, marker="X")

    plt.title(f"{participant} – Time-coded Scanpath (Question {question_id})", fontsize=12)
    plt.axis("off")
    plt.tight_layout(pad=0)

    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{participant}_Q{question_id}_timecoded_scanpath.png")
    plt.savefig(out_file, dpi=300)
    plt.close()

    return out_file


def build_pdf_report(single_row_df: pd.DataFrame, summary_df: pd.DataFrame, out_pdf: str, title: str):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(out_pdf, pagesize=landscape(A4),
                            leftMargin=1.2*cm, rightMargin=1.2*cm,
                            topMargin=1.0*cm, bottomMargin=1.0*cm)

    story = []
    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Mittelwerte pro Teilnehmer (über alle Fragen)", styles["Heading2"]))
    story.append(Spacer(1, 6))

    sum_data = [list(summary_df.reset_index().columns)] + summary_df.reset_index().values.tolist()
    sum_table = Table(sum_data, repeatRows=1)
    sum_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaeaea")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(sum_table)
    story.append(Spacer(1, 14))

    story.append(Paragraph("Beste Vergleichsfrage (datenbasiert gewählt)", styles["Heading2"]))
    story.append(Spacer(1, 6))

    row_data = [list(single_row_df.columns)] + single_row_df.values.tolist()
    row_table = Table(row_data, repeatRows=1)
    row_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaeaea")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(row_table)

    doc.build(story)


# ============================================================
# MAIN
# ============================================================

if len(PARTICIPANTS) != 2:
    raise ValueError("Bitte genau 2 Participants angeben.")

A, B = PARTICIPANTS

dfA_metrics, dfA_raw = analyze_participant(A)
dfB_metrics, dfB_raw = analyze_participant(B)

# Gemeinsame Fragen
common = pd.merge(dfA_metrics, dfB_metrics, on="Question_ID", suffixes=("_A", "_B"), how="inner")
if common.empty:
    raise RuntimeError("Keine gemeinsamen Questions gefunden.")

# Score berechnen und beste Frage auswählen
common["Strategy_Diff_Score"] = standardize_diff_score(common, STRATEGY_METRICS_FOR_SELECTION)
best = common.sort_values("Strategy_Diff_Score", ascending=False).iloc[0]

best_qid = int(best["Question_ID"])
best_label_A = best["Question_Label_A"]
best_label_B = best["Question_Label_B"]

print("\n=== Auto-Auswahl (beste Vergleichsfrage) ===")
print(f"Question_ID: {best_qid}")
print(f"Strategy_Diff_Score: {best['Strategy_Diff_Score']:.3f}")
print(f"A: {best_label_A}")
print(f"B: {best_label_B}")

# Time-coded Plots nur für diese Frage speichern
plots_dir = os.path.join(OUT_DIR, "timecoded_scanpaths_best")
plotA = save_timecoded_scanpath_plot(A, dfA_raw, best_qid, best_label_A, plots_dir)
plotB = save_timecoded_scanpath_plot(B, dfB_raw, best_qid, best_label_B, plots_dir)

print("\nGespeicherte Plots:")
print("A:", plotA)
print("B:", plotB)

# Option B: Mittelwerte über alle Fragen
summary = pd.concat([dfA_metrics, dfB_metrics]).groupby("Participant").mean(numeric_only=True).round(4)

# Tabelle: NUR diese eine Frage
cols_keep = [
    "Question_ID",
    "Duration_sec_A", "Saccades_per_min_A", "Scanpath_per_min_norm_A", "Vertical_Ratio_A", "Regression_Rate_A", "Directional_Entropy_A",
    "Duration_sec_B", "Saccades_per_min_B", "Scanpath_per_min_norm_B", "Vertical_Ratio_B", "Regression_Rate_B", "Directional_Entropy_B",
    "Strategy_Diff_Score"
]

best_table = common[common["Question_ID"] == best_qid][cols_keep].copy()

for c in best_table.columns:
    if best_table[c].dtype != object:
        best_table[c] = best_table[c].astype(float).round(4)

# CSV speichern
best_table.to_csv(os.path.join(OUT_DIR, "best_question_table.csv"), index=False)
summary.to_csv(os.path.join(OUT_DIR, "participant_means.csv"))

# PDF speichern (nur diese eine Frage + Mittelwerte)
pdf_path = os.path.join(OUT_DIR, f"Strategy_Report_BestQuestion_{A}_vs_{B}.pdf")
build_pdf_report(
    single_row_df=best_table,
    summary_df=summary,
    out_pdf=pdf_path,
    title=f"Strategy Report (Best Question) – {A} vs {B}"
)

print("\n=== PDF gespeichert ===")
print(pdf_path)

print("\n=== Fertig ===")