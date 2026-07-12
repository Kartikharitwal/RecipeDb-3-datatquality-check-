import pandas as pd
from pathlib import Path

input_file = Path("RecipeDB3_MasterSheet.csv")
output_file = Path("same_name_description_ingredients_different_fields.csv")
summary_file = Path("same_name_description_ingredients_summary.csv")

# Columns used to identify the "same recipe"
identity_cols = [
    "Recipe Name",
    "Description",
    "Ingredients",
]

# Columns to compare after identity match
compare_cols = [
    "Recipe_ID",
    "Cuisine",
    "Category",
    "Prep Time",
    "Cook Time",
    "Total Time",
    "Servings",
    "Directions",
    "Nutrients",
    "Keywords",
    "Source",
    "URL",
    "Image_ID",
    "Image_URL",
    "NA_Image",
    "Language",
    "Ratings",
    "Ratings_Count",
]

usecols = identity_cols + compare_cols

print("Reading CSV...")
df = pd.read_csv(input_file, dtype=str, keep_default_na=False, usecols=usecols)

# Normalize identity columns so small spacing/case differences do not break matching
for col in identity_cols:
    df[col + "_clean"] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )

identity_clean_cols = [col + "_clean" for col in identity_cols]

# Count how many records share same name + description + ingredients
df["identity_count"] = df.groupby(identity_clean_cols)["Recipe_ID"].transform("count")

# Keep only groups where same identity appears more than once
duplicate_identity_df = df[df["identity_count"] > 1].copy()

print(f"Records with same name + description + ingredients: {len(duplicate_identity_df)}")

# Now check whether other fields are different inside each identity group
issue_rows = []
summary_rows = []

for group_key, group in duplicate_identity_df.groupby(identity_clean_cols):
    different_columns = []

    for col in compare_cols:
        if col == "Recipe_ID":
            continue

        unique_values = (
            group[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)
            .nunique()
        )

        if unique_values > 1:
            different_columns.append(col)

    # Keep only groups where some other field is different
    if different_columns:
        recipe_ids = ", ".join(group["Recipe_ID"].astype(str).tolist())

        summary_rows.append({
            "Recipe Name": group["Recipe Name"].iloc[0],
            "Record Count": len(group),
            "Recipe_IDs": recipe_ids,
            "Different Columns": ", ".join(different_columns),
        })

        temp = group.copy()
        temp["Different Columns"] = ", ".join(different_columns)
        issue_rows.append(temp)

if issue_rows:
    result = pd.concat(issue_rows, ignore_index=True)
else:
    result = pd.DataFrame()

summary = pd.DataFrame(summary_rows)

# Remove helper clean columns from detailed output
drop_cols = identity_clean_cols + ["identity_count"]
for col in drop_cols:
    if col in result.columns:
        result = result.drop(columns=[col])

result.to_csv(output_file, index=False)
summary.to_csv(summary_file, index=False)

print(f"Groups found: {len(summary)}")
print(f"Records affected: {len(result)}")

print(f"Detailed output saved to: {output_file}")
print(f"Summary output saved to: {summary_file}")

print("\nFirst 20 groups:")
print(summary.head(20).to_string(index=False))