# RecipeDB Semantic Data Quality Check

This project runs semantic quality checks on a RecipeDB CSV file using Sentence-BERT, zero-shot classification, and language detection.

## Input

Place the dataset file in the project root:

```text
RecipeDB3_random2k.csv
```

Expected columns include:

- `Recipe Name`
- `Ingredients`
- `Directions`
- `Description`
- `Cuisine`
- `Category`
- `Keywords`
- `Language`

## Run

```bash
pip install -r requirements.txt
python semantic_quality_check.py
```

## Outputs

- `semantic_analysis_results.csv`
- `semantic_duplicates.csv`
- `missing_semantic_fields.csv`
