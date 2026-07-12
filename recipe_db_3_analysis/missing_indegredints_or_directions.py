import ast
import pandas as pd
from pathlib import Path

input_file = Path("RecipeDB3_MasterSheet.csv")

usecols = [
    "Recipe_ID",
    "Recipe Name",
    "Ingredients",
    "Directions",
    "Source",
    "URL",
]

def is_empty_list(value):
    """
    Returns True if value is:
    - missing / blank
    - []
    - a parsed empty Python list
    """
    if pd.isna(value):
        return True

    value = str(value).strip()

    if value == "" or value.lower() in ["na", "n/a", "nan", "none", "null"]:
        return True

    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list) and len(parsed) == 0:
            return True
    except Exception:
        pass

    return False


empty_records = []

for chunk in pd.read_csv(input_file, dtype=str, keep_default_na=False, usecols=usecols, chunksize=50000):
    for _, row in chunk.iterrows():
        empty_ingredients = is_empty_list(row["Ingredients"])
        empty_directions = is_empty_list(row["Directions"])

        if empty_ingredients or empty_directions:
            empty_records.append({
                "Recipe_ID": row["Recipe_ID"],
                "Recipe Name": row["Recipe Name"],
                "Empty Ingredients": "YES" if empty_ingredients else "NO",
                "Empty Directions": "YES" if empty_directions else "NO",
                "Source": row["Source"],
                "URL": row["URL"],
            })


print(f"Total recipes with empty ingredients or directions: {len(empty_records)}")
print("-" * 120)

for record in empty_records:
    print(f"Recipe_ID: {record['Recipe_ID']}")
    print(f"Recipe Name: {record['Recipe Name']}")
    print(f"Empty Ingredients: {record['Empty Ingredients']}")
    print(f"Empty Directions: {record['Empty Directions']}")
    print(f"Source: {record['Source']}")
    print(f"URL: {record['URL']}")
    print("-" * 120)