import ast
import pandas as pd
from pathlib import Path

input_file = Path("RecipeDB3_MasterSheet.csv")

missing_tokens = {
    "",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "unknown",
    "-",
    "--",
}

list_like_columns = {"Ingredients", "Directions", "Keywords"}
dict_like_columns = {"Nutrients"}


def is_missing_value(value, column_name):
    text = str(value).strip()

    # Normal blank / placeholder missing values
    if text.lower() in missing_tokens:
        return True

    # Logical missing for serialized list columns: []
    if column_name in list_like_columns:
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list) and len(parsed) == 0:
                return True
        except Exception:
            pass

    # Logical missing for serialized dict columns: {}
    if column_name in dict_like_columns:
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict) and len(parsed) == 0:
                return True
        except Exception:
            pass

    return False


counts = None
total_rows = 0

for chunk in pd.read_csv(input_file, dtype=str, keep_default_na=False, chunksize=50000):
    total_rows += len(chunk)

    chunk_counts = {}

    for col in chunk.columns:
        chunk_counts[col] = chunk[col].apply(
            lambda x: is_missing_value(x, col)
        ).sum()

    chunk_counts = pd.Series(chunk_counts)

    if counts is None:
        counts = chunk_counts
    else:
        counts = counts.add(chunk_counts, fill_value=0)

print(f"Total rows: {total_rows:,}")
print()
print("Missing values by column, including [] / {} as missing:")
print("-" * 70)
print(f"{'Column':25} {'Missing Count':>15} {'Missing %':>12}")
print("-" * 70)

for col, cnt in counts.astype(int).items():
    pct = cnt / total_rows * 100
    print(f"{col:25} {cnt:15,} {pct:11.2f}%")