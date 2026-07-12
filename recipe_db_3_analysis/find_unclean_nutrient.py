import ast
import csv
import re
from pathlib import Path

import pandas as pd

input_file = Path("RecipeDB3_MasterSheet.csv")
output_file = Path("unclean_nutrients.csv")
summary_file = Path("unclean_nutrients_summary.csv")

expected_nutrient_keys = {
    "calories",
    "fatContent",
    "saturatedFatContent",
    "cholesterolContent",
    "sodiumContent",
    "carbohydrateContent",
    "fiberContent",
    "sugarContent",
    "proteinContent",
}

# Allows clean numeric values like:
# 12
# 12.5
# -1
# 1,234.5
numeric_pattern = re.compile(r"^-?\d{1,3}(,\d{3})*(\.\d+)?$|^-?\d+(\.\d+)?$")


def parse_nutrients(value):
    """
    Nutrients column is Python-dict-like text, not strict JSON.
    Example:
    {'calories': '307.5', 'fatContent': '15.9'}
    """
    try:
        parsed = ast.literal_eval(str(value).strip())
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    return None


def is_clean_numeric(value):
    value = str(value).strip()

    if value == "":
        return False

    return bool(numeric_pattern.match(value))


def check_nutrients(nutrients):
    issues = []

    if nutrients is None:
        issues.append({
            "Issue": "Nutrients not parseable",
            "Nutrient_Key": "",
            "Current_Value": "",
            "Reason": "Nutrients value cannot be parsed as a dictionary",
        })
        return issues

    # Missing expected keys
    missing_keys = expected_nutrient_keys - set(nutrients.keys())

    for key in sorted(missing_keys):
        issues.append({
            "Issue": "Missing nutrient key",
            "Nutrient_Key": key,
            "Current_Value": "",
            "Reason": f"Expected key '{key}' is not present in Nutrients",
        })

    # Non-numeric or unclean values
    for key, value in nutrients.items():
        if not is_clean_numeric(value):
            issues.append({
                "Issue": "Unclean nutrient value",
                "Nutrient_Key": key,
                "Current_Value": value,
                "Reason": "Nutrient value contains text/unit or is not a clean number",
            })

    return issues


output_rows = []

usecols = [
    "Recipe_ID",
    "Recipe Name",
    "Nutrients",
    "Source",
    "URL",
]

for chunk in pd.read_csv(input_file, dtype=str, keep_default_na=False, usecols=usecols, chunksize=50000):
    for _, row in chunk.iterrows():
        nutrients = parse_nutrients(row["Nutrients"])
        issues = check_nutrients(nutrients)

        for issue in issues:
            output_rows.append({
                "Recipe_ID": row["Recipe_ID"],
                "Recipe Name": row["Recipe Name"],
                "Issue": issue["Issue"],
                "Nutrient_Key": issue["Nutrient_Key"],
                "Current_Value": issue["Current_Value"],
                "Reason": issue["Reason"],
                "Nutrients": row["Nutrients"],
                "Source": row["Source"],
                "URL": row["URL"],
            })


# Save detailed CSV
with output_file.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "Recipe_ID",
            "Recipe Name",
            "Issue",
            "Nutrient_Key",
            "Current_Value",
            "Reason",
            "Nutrients",
            "Source",
            "URL",
        ],
    )
    writer.writeheader()
    writer.writerows(output_rows)


# Save summary CSV
summary = pd.DataFrame(output_rows)

if not summary.empty:
    summary_count = (
        summary.groupby(["Issue", "Nutrient_Key"])
        .size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
    )
else:
    summary_count = pd.DataFrame(columns=["Issue", "Nutrient_Key", "Count"])

summary_count.to_csv(summary_file, index=False)

print(f"Total nutrient issues found: {len(output_rows)}")
print(f"Detailed output saved to: {output_file}")
print(f"Summary output saved to: {summary_file}")

print("\nTop issue summary:")
print(summary_count.head(30).to_string(index=False))