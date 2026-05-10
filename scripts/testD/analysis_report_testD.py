import os
import glob
import pandas as pd

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4

# ---------------------------------------------------
# PATHS
# ---------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "testD")
RESULTS_PATH = os.path.join(BASE_DIR, "results", "testD", "fixation_reports")

os.makedirs(RESULTS_PATH, exist_ok=True)

files = glob.glob(os.path.join(DATA_PATH, "*.tsv"))

scores_df = pd.read_csv(os.path.join(DATA_PATH, "scores.csv"))
answers_df = pd.read_csv(os.path.join(DATA_PATH, "answers.csv"))

# ---------------------------------------------------
# KONSTANTEN
# ---------------------------------------------------

TIMESTAMP = "Recording timestamp [ms]"
DURATION = "Gaze event duration [ms]"
INDEX = "Eye movement type index"

all_results = []

print("Found participants:", len(files))

# ---------------------------------------------------
# HELFER
# ---------------------------------------------------

def find_column(columns, candidates):
    for cand in candidates:
        cand_lower = cand.lower().strip()
        for c in columns:
            if cand_lower == c.lower().strip():
                return c
        for c in columns:
            if cand_lower in c.lower().strip():
                return c
    return None

# ---------------------------------------------------
# VERARBEITUNG
# ---------------------------------------------------

for file in files:
    participant_name = os.path.basename(file).replace(".tsv", "")
    df = pd.read_csv(file, sep="\t", low_memory=False)

    df.columns = df.columns.str.strip()

    timestamp_col = find_column(df.columns, [
        "Recording timestamp [ms]",
        "Recording timestamp"
    ])

    duration_col = find_column(df.columns, [
        "Gaze event duration [ms]",
        "Gaze event duration"
    ])

    index_col = find_column(df.columns, [
        "Eye movement type index"
    ])

    event_col = find_column(df.columns, ["Event"])
    event_value_col = find_column(df.columns, ["Event value"])
    movement_col = find_column(df.columns, ["Eye movement type"])

    if not all([timestamp_col, duration_col, index_col, event_col, event_value_col, movement_col]):
        print(f"Fehlende Spalten in {participant_name}")
        print("Columns:", list(df.columns))
        continue

    df = df.rename(columns={
        timestamp_col: TIMESTAMP,
        duration_col: DURATION,
        index_col: INDEX,
        event_col: "Event",
        event_value_col: "Event value",
        movement_col: "Eye movement type"
    })

    df = df.loc[:, ~df.columns.duplicated()].copy()

    df[TIMESTAMP] = pd.to_numeric(df[TIMESTAMP], errors="coerce")
    df[DURATION] = pd.to_numeric(df[DURATION], errors="coerce")
    df[INDEX] = pd.to_numeric(df[INDEX], errors="coerce")

    df = df.dropna(subset=[TIMESTAMP])

    # -------------------------------
    # Question Events
    # -------------------------------
    question_events = df[
        (df["Event"] == "URLStart") &
        (df["Event value"].astype(str).str.contains("Question", na=False))
        ].copy()

    if question_events.empty:
        print(f"Keine Question-Events in {participant_name}")
        continue

    question_events = question_events.sort_values(TIMESTAMP)

    # -------------------------------
    # Fixationen
    # -------------------------------
    fix = df[df["Eye movement type"] == "Fixation"].copy()

    if fix.empty:
        print(f"Keine Fixations in {participant_name}")
        continue

    fix = fix[pd.to_numeric(fix[DURATION], errors="coerce") > 0].copy()

    if fix.empty:
        print(f"Keine gültigen Fixations nach Duration-Filter in {participant_name}")
        continue

    first_question_time = question_events.iloc[0][TIMESTAMP]
    fix = fix[fix[TIMESTAMP] >= first_question_time].copy()

    if fix.empty:
        print(f"Keine Fixations nach erstem Question-Start in {participant_name}")
        continue

    # -------------------------------
    # Segmentierung
    # -------------------------------
    for i in range(len(question_events)):
        start_time = question_events.iloc[i][TIMESTAMP]

        if i < len(question_events) - 1:
            end_time = question_events.iloc[i + 1][TIMESTAMP]
        else:
            end_time = df[TIMESTAMP].max()

        question_name = str(question_events.iloc[i]["Event value"])

        question_fix = fix[
            (fix[TIMESTAMP] >= start_time) &
            (fix[TIMESTAMP] < end_time)
            ].copy()

        if question_fix.empty:
            continue

        unique_fix = question_fix.drop_duplicates(subset=INDEX)

        if unique_fix.empty:
            continue

        fixation_count = unique_fix[INDEX].nunique()
        mean_fix = unique_fix[DURATION].mean()
        total_dwell = unique_fix[DURATION].sum()

        all_results.append([
            participant_name,
            question_name.split(" (")[0],
            fixation_count,
            round(mean_fix, 2) if pd.notna(mean_fix) else 0,
            round(total_dwell, 2) if pd.notna(total_dwell) else 0
        ])

# ---------------------------------------------------
# DATAFRAME
# ---------------------------------------------------

result_df = pd.DataFrame(all_results, columns=[
    "Participant",
    "Question",
    "Fix_Count",
    "Mean_Fix_ms",
    "Total_Dwell_ms"
])

if result_df.empty:
    raise ValueError("Keine Daten verarbeitet. Prüfe, ob die Test-D TSV-Dateien die erwarteten Event- und Fixation-Spalten enthalten.")

result_df["Visualization"] = result_df["Question"].str.replace(
    r"Question \d+ – ", "", regex=True
)

result_df["Participant_ID"] = result_df["Participant"].str.extract(r'(\d+)').astype(int)

# Test D: 61-65
result_df = result_df[
    (result_df["Participant_ID"] >= 61) &
    (result_df["Participant_ID"] <= 65)
    ].copy()

if result_df.empty:
    raise ValueError("Nach dem Test-D Filter (61-65) sind keine Daten übrig.")

participants = result_df["Participant"].unique()
print("Participants im Report:", participants)

# ---------------------------------------------------
# PDF ERSTELLEN
# ---------------------------------------------------

pdf_file = os.path.join(RESULTS_PATH, "All_Participants_Report_testD.pdf")
doc = SimpleDocTemplate(pdf_file, pagesize=A4)
elements = []
styles = getSampleStyleSheet()

elements.append(Paragraph("Eye Tracking Report – Test D", styles["Heading1"]))
elements.append(Spacer(1, 0.4 * inch))

for p in participants:
    participant_answers = answers_df[answers_df["Participant"] == p]
    score_row = scores_df[scores_df["Participant"] == p]

    participant_score = int(score_row["Score"].values[0]) if not score_row.empty else "N/A"

    duration_row = answers_df[answers_df["Participant"] == p]
    participant_duration = duration_row["TotalDuration"].iloc[0] if (
            not duration_row.empty and "TotalDuration" in answers_df.columns
    ) else "N/A"

    participant_data = result_df[result_df["Participant"] == p]

    elements.append(Paragraph(f"Participant: {p}", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Paragraph(f"<b>Total Score:</b> {participant_score}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Total Duration:</b> {participant_duration} s", styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))

    table_data = [["Question", "Fix Count", "Mean Fix (ms)", "Total Dwell (ms)"]]

    for _, row in participant_data.iterrows():
        table_data.append([
            row["Question"],
            row["Fix_Count"],
            row["Mean_Fix_ms"],
            row["Total_Dwell_ms"]
        ])

    table = Table(table_data, colWidths=[150, 80, 100, 120], repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.grey),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 0.3 * inch))

    answer_table_data = [["Question", "Seconds", "Correct"]]

    for _, row in participant_answers.iterrows():
        symbol = "✓" if row["Correct"] == 1 else "✗"
        answer_table_data.append([row["Question"], row["Seconds"], symbol])

    answer_table = Table(answer_table_data, colWidths=[120, 80, 60], repeatRows=1)
    answer_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.green),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.grey),
    ]))

    elements.append(answer_table)
    elements.append(PageBreak())

doc.build(elements)

print("\n PDF erstellt:", pdf_file)