import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from langdetect import detect

############################################################
# Load data
############################################################

df = pd.read_csv("RecipeDB3_random2k.csv")

############################################################
# Fill missing values
############################################################

text_columns = [
    "Recipe Name",
    "Ingredients",
    "Directions",
    "Description",
    "Cuisine",
    "Category",
    "Keywords",
]

for c in text_columns:
    if c in df.columns:
        df[c] = df[c].fillna("").astype(str)

############################################################
# Load models
############################################################

print("Loading Sentence-BERT...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Loading Zero-shot model...")

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

############################################################
# Sentence-BERT Similarity
############################################################

def similarity(a, b):

    emb = model.encode([a, b], convert_to_tensor=True)

    return float(
        util.cos_sim(emb[0], emb[1])
    )

############################################################
# Semantic Similarities
############################################################

df["Title_Ingredients_Similarity"] = df.apply(
    lambda x: similarity(
        x["Recipe Name"],
        x["Ingredients"]
    ),
    axis=1
)

df["Title_Directions_Similarity"] = df.apply(
    lambda x: similarity(
        x["Recipe Name"],
        x["Directions"]
    ),
    axis=1
)

df["Title_Description_Similarity"] = df.apply(
    lambda x: similarity(
        x["Recipe Name"],
        x["Description"]
    ),
    axis=1
)

df["Ingredients_Directions_Similarity"] = df.apply(
    lambda x: similarity(
        x["Ingredients"],
        x["Directions"]
    ),
    axis=1
)

df["Description_Directions_Similarity"] = df.apply(
    lambda x: similarity(
        x["Description"],
        x["Directions"]
    ),
    axis=1
)

############################################################
# Language Validation
############################################################

def detect_language(text):

    try:
        return detect(text)

    except:
        return "unknown"

df["Detected_Language"] = df["Description"].apply(
    detect_language
)

df["Language_Match"] = (
    df["Detected_Language"]
    ==
    df["Language"]
)

############################################################
# Zero-shot Category Validation
############################################################

categories = sorted(
    df["Category"].dropna().unique().tolist()
)

def predict_category(row):

    if row["Description"] == "":

        return ""

    result = classifier(
        row["Description"],
        candidate_labels=categories
    )

    return result["labels"][0]

df["Predicted_Category"] = df.apply(
    predict_category,
    axis=1
)

df["Category_Match"] = (
    df["Predicted_Category"]
    ==
    df["Category"]
)

############################################################
# Zero-shot Cuisine Validation
############################################################

cuisines = sorted(
    df["Cuisine"].dropna().unique().tolist()
)

def predict_cuisine(row):

    if row["Description"] == "":

        return ""

    result = classifier(
        row["Description"],
        candidate_labels=cuisines
    )

    return result["labels"][0]

df["Predicted_Cuisine"] = df.apply(
    predict_cuisine,
    axis=1
)

df["Cuisine_Match"] = (
    df["Predicted_Cuisine"]
    ==
    df["Cuisine"]
)

############################################################
# Duplicate Detection
############################################################

print("Encoding titles...")

title_embeddings = model.encode(
    df["Recipe Name"].tolist(),
    convert_to_tensor=True
)

duplicates = []

for i in range(len(df)):

    sims = util.cos_sim(
        title_embeddings[i],
        title_embeddings
    )[0]

    for j in range(i + 1, len(df)):

        if sims[j] > 0.90:

            duplicates.append(
                (
                    i,
                    j,
                    float(sims[j])
                )
            )

dup_df = pd.DataFrame(
    duplicates,
    columns=[
        "Recipe1",
        "Recipe2",
        "Similarity"
    ]
)

############################################################
# Missing semantic fields
############################################################

semantic_cols = [
    "Recipe Name",
    "Ingredients",
    "Directions",
    "Description"
]

missing_report = df[semantic_cols].isnull().sum()

############################################################
# Save outputs
############################################################

df.to_csv(
    "semantic_analysis_results.csv",
    index=False
)

dup_df.to_csv(
    "semantic_duplicates.csv",
    index=False
)

missing_report.to_csv(
    "missing_semantic_fields.csv"
)

############################################################

print("Completed.")
print("Results saved.")
