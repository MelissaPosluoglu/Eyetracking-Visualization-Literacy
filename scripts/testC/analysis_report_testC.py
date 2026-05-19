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
DATA_PATH = os.path.join(BASE_DIR, "data", "testC")
RESULTS_PATH = os.path.join(BASE_DIR, "results", "testC", "fixation_reports")

# Create result folder if needed
os.makedirs(RESULTS_PATH, exist_ok=True)

# Load all participant TSV files
files = glob.glob(os.path.join(DATA_PATH, "*.tsv"))

# Load score and answer files
scores_df = pd.read_csv(os.path.join(DATA_PATH, "scores.csv"))
answers_df = pd.read_csv(os.path.join(DATA_PATH, "answers.csv"))

# ---------------------------------------------------
# CONSTANTS
# ---------------------------------------------------

# Standard column names used after renaming
TIMESTAMP = "Recording timestamp [ms]"
DURATION = "Gaze event duration [ms]"
INDEX = "Eye movement type index"

# Store all calculated fixation results
all_results = []

print("Found participants:", len(files))

# ---------------------------------------------------
# HELPER
# ---------------------------------------------------

def find_column(columns, candidates):
    # Find a column name even if the exact spelling differs
    cols_lower = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        cand_lower = cand.lower().strip()
        # First check exact matches
        for c in columns:
            if cand_lower == c.lower().strip():
                return c

        # Then check partial matches
        for c in columns:
            if cand_lower in c.lower().strip():
                return c
    return None

# ---------------------------------------------------
# PROCESS PARTICIPANT FILES
# ---------------------------------------------------

for file in files:

    # Extract participant name from file name
    participant_name = os.path.basename(file).replace(".tsv", "")

    # Load participant eye-tracking data
    df = pd.read_csv(file, sep="\t", low_memory=False)

    # Remove extra spaces from column names
    df.columns = df.columns.str.strip()

    # Find required columns automatically
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

    # Skip participant if important columns are missing
    if not all([timestamp_col, duration_col, index_col, event_col, event_value_col, movement_col]):
        print(f"Fehlende Spalten in {participant_name}")
        print("Columns:", list(df.columns))
        continue

    # Rename columns to standard names
    df = df.rename(columns={
        timestamp_col: TIMESTAMP,
        duration_col: DURATION,
        index_col: INDEX,
        event_col: "Event",
        event_value_col: "Event value",
        movement_col: "Eye movement type"
    })

    # Remove duplicated columns
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Convert important columns to numeric values
    df[TIMESTAMP] = pd.to_numeric(df[TIMESTAMP], errors="coerce")
    df[DURATION] = pd.to_numeric(df[DURATION], errors="coerce")
    df[INDEX] = pd.to_numeric(df[INDEX], errors="coerce")

    # Remove rows without valid timestamps
    df = df.dropna(subset=[TIMESTAMP])

    # -------------------------------
    # QUESTION EVENTS
    # -------------------------------

    # Select all question start events
    question_events = df[
        (df["Event"] == "URLStart") &
        (df["Event value"].astype(str).str.contains("Question", na=False))
        ].copy()

    if question_events.empty:
        print(f"Keine Question-Events in {participant_name}")
        continue

    # Sort question events by time
    question_events = question_events.sort_values(TIMESTAMP)

    # -------------------------------
    # FIXATION DATA
    # -------------------------------

    # Keep only fixation rows
    fix = df[df["Eye movement type"] == "Fixation"].copy()

    if fix.empty:
        print(f"Keine Fixations in {participant_name}")
        continue

    # Keep only fixations with positive duration
    fix = fix[pd.to_numeric(fix[DURATION], errors="coerce") > 0].copy()

    if fix.empty:
        print(f"Keine gültigen Fixations nach Duration-Filter in {participant_name}")
        continue


    # Remove fixations before the first question
    first_question_time = question_events.iloc[0][TIMESTAMP]
    fix = fix[fix[TIMESTAMP] >= first_question_time].copy()

    if fix.empty:
        print(f"Keine Fixations nach erstem Question-Start in {participant_name}")
        continue

    # -------------------------------
    # SEGMENT FIXATIONS BY QUESTION
    # -------------------------------
    for i in range(len(question_events)):
        # Start time of current question
        start_time = question_events.iloc[i][TIMESTAMP]

        # End time is the next question start
        if i < len(question_events) - 1:
            end_time = question_events.iloc[i + 1][TIMESTAMP]
        else:
            end_time = df[TIMESTAMP].max()

        # Current question label
        question_name = str(question_events.iloc[i]["Event value"])

        # Select fixations within the current question time window
        question_fix = fix[
            (fix[TIMESTAMP] >= start_time) &
            (fix[TIMESTAMP] < end_time)
            ].copy()

        if question_fix.empty:
            continue

        # Remove duplicated fixation events
        unique_fix = question_fix.drop_duplicates(subset=INDEX)

        if unique_fix.empty:
            continue

        # Calculate fixation metrics
        fixation_count = unique_fix[INDEX].nunique()
        mean_fix = unique_fix[DURATION].mean()
        total_dwell = unique_fix[DURATION].sum()

        # Store result row
        all_results.append([
            participant_name,
            question_name.split(" (")[0],
            fixation_count,
            round(mean_fix, 2) if pd.notna(mean_fix) else 0,
            round(total_dwell, 2) if pd.notna(total_dwell) else 0
        ])

# ---------------------------------------------------
# CREATE RESULT DATAFRAME
# ---------------------------------------------------

# Convert collected results into a dataframe
result_df = pd.DataFrame(all_results, columns=[
    "Participant",
    "Question",
    "Fix_Count",
    "Mean_Fix_ms",
    "Total_Dwell_ms"
])

# Stop if no data was processed
if result_df.empty:
    raise ValueError("Keine Daten verarbeitet. Prüfe, ob die Test-C TSV-Dateien die erwarteten Event- und Fixation-Spalten enthalten.")

# Extract visualization name from question label
result_df["Visualization"] = result_df["Question"].str.replace(
    r"Question \d+ – ", "", regex=True
)

# Extract numeric participant ID
result_df["Participant_ID"] = result_df["Participant"].str.extract(r'(\d+)').astype(int)

# Keep only Test C participants
result_df = result_df[
    (result_df["Participant_ID"] >= 51) &
    (result_df["Participant_ID"] <= 55)
    ].copy()

# Stop if no participants remain after filtering
if result_df.empty:
    raise ValueError("Nach dem Test-C Filter (51-55) sind keine Daten übrig.")

# Get participant list for the report
participants = result_df["Participant"].unique()
print("Participants im Report:", participants)

# ---------------------------------------------------
# CREATE PDF REPORT
# ---------------------------------------------------

# Define output PDF path
pdf_file = os.path.join(RESULTS_PATH, "All_Participants_Report_testC.pdf")
# Create PDF document
doc = SimpleDocTemplate(pdf_file, pagesize=A4)
elements = []
styles = getSampleStyleSheet()


# Add report title
elements.append(Paragraph("Eye Tracking Report – Test C", styles["Heading1"]))
elements.append(Spacer(1, 0.4 * inch))

# Create one report section per participant
for p in participants:

    # Select answer and score data for current participant
    participant_answers = answers_df[answers_df["Participant"] == p]
    score_row = scores_df[scores_df["Participant"] == p]

    # Get participant score
    participant_score = int(score_row["Score"].values[0]) if not score_row.empty else "N/A"

    # Get total duration if available
    duration_row = answers_df[answers_df["Participant"] == p]
    participant_duration = duration_row["TotalDuration"].iloc[0] if (
            not duration_row.empty and "TotalDuration" in answers_df.columns
    ) else "N/A"

    # Select fixation metrics for current participant
    participant_data = result_df[result_df["Participant"] == p]

    # Add participant information
    elements.append(Paragraph(f"Participant: {p}", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Paragraph(f"<b>Total Score:</b> {participant_score}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Total Duration:</b> {participant_duration} s", styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))

    # -------------------------------
    # FIXATION METRICS TABLE
    # -------------------------------

    # Create table header
    table_data = [["Question", "Fix Count", "Mean Fix (ms)", "Total Dwell (ms)"]]


    # Add one row per question
    for _, row in participant_data.iterrows():
        table_data.append([
            row["Question"],
            row["Fix_Count"],
            row["Mean_Fix_ms"],
            row["Total_Dwell_ms"]
        ])

    # Create fixation table
    table = Table(table_data, colWidths=[150, 80, 100, 120], repeatRows=1)

    # Style fixation table
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.grey),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 0.3 * inch))

    # -------------------------------
    # ANSWER CORRECTNESS TABLE
    # -------------------------------

    # Create answer table header
    answer_table_data = [["Question", "Seconds", "Correct"]]

    # Add answer rows
    for _, row in participant_answers.iterrows():
        symbol = "✓" if row["Correct"] == 1 else "✗"
        answer_table_data.append([row["Question"], row["Seconds"], symbol])

    # Create answer table
    answer_table = Table(answer_table_data, colWidths=[120, 80, 60], repeatRows=1)

    # Style answer table
    answer_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.green),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.grey),
    ]))

    elements.append(answer_table)


    # Start new page for next participant
    elements.append(PageBreak())


# Build and save PDF report
doc.build(elements)

print("\n PDF erstellt:", pdf_file)