from pathlib import Path
import pandas as pd


INPUT_FILE = Path("../../data/testB/Participant29.tsv")
OUTPUT_FILE = Path("../../data/testB/Participant29.tsv")

# Spalten mit Datum, Uhrzeit oder Zeitstempeln
COLUMNS_TO_REMOVE = [
    "Computer timestamp [ms]",
    "Export date",
    "Recording date",
    "Recording date UTC",
    "Recording start time",
    "Recording start time UTC",
    "Eyetracker timestamp [μs]",
]

# TSV-Datei einlesen
df = pd.read_csv(INPUT_FILE, sep="\t", low_memory=False, encoding="utf-8-sig")

# Nur Spalten löschen, die wirklich in der Datei existieren
existing_columns = [col for col in COLUMNS_TO_REMOVE if col in df.columns]
missing_columns = [col for col in COLUMNS_TO_REMOVE if col not in df.columns]

df_clean = df.drop(columns=existing_columns)

# Neue TSV speichern
df_clean.to_csv(OUTPUT_FILE, sep="\t", index=False, encoding="utf-8-sig")

print("Fertig.")
print(f"Input:  {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")
print(f"Entfernte Spalten: {existing_columns}")

if missing_columns:
    print(f"Nicht gefunden: {missing_columns}")