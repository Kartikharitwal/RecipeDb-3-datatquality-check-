import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer, util
# from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect

df = pd.read_csv("RecipeDB3_random2k.csv")


text_columns = [
    "Recipe Name",
    "Ingredients",
    "Directions",
    "Description",
    "Cuisine",
    "Category",
    "Keywords"
]

for c in text_columns:
    if c in df.columns:
        df[c] = df[c].fillna("").astype(str)


print("Loading Sentence-BERT...")

# model = SentenceTransformer(
#     "sentence-transformers/all-MiniLM-L6-v2"
# )
model = SentenceTransformer(
    "sentence-transformers/paraphrase-MiniLM-L3-v2"
)

print("Loading Zero-shot model...")

# classifier = pipeline(
#     "zero-shot-classification",
#     model="facebook/bart-large-mnli"
# )

def similarity(a,b):

    emb = model.encode([a,b],convert_to_tensor=True)

    return float(
        util.cos_sim(emb[0],emb[1])
    )


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

categories = sorted(
    df["Category"].dropna().unique().tolist()
)

# def predict_category(row):

#     if row["Description"]=="":

#         return ""

#     result = classifier(
#         row["Description"],
#         candidate_labels=categories
#     )

#     return result["labels"][0]

# df["Predicted_Category"] = df.apply(
#     predict_category,
#     axis=1
# )

# df["Category_Match"] = (
#     df["Predicted_Category"]
#     ==
#     df["Category"]
# )

categories = sorted(df["Category"].dropna().unique().tolist())

category_embeddings = model.encode(
    categories,
    convert_to_tensor=False
)

def predict_category(description):

    if description.strip() == "":
        return ""

    description_embedding = model.encode(
        [description],
        convert_to_tensor=False
    )

    similarities = cosine_similarity(
        description_embedding,
        category_embeddings
    )[0]

    return categories[np.argmax(similarities)]

df["Predicted_Category"] = df["Description"].apply(
    predict_category
)

df["Category_Match"] = (
    df["Predicted_Category"]
    ==
    df["Category"]
)

# cuisines = sorted(
#     df["Cuisine"].dropna().unique().tolist()
# )

# def predict_cuisine(row):

#     if row["Description"]=="":

#         return ""

#     result = classifier(
#         row["Description"],
#         candidate_labels=cuisines
#     )

#     return result["labels"][0]

# df["Predicted_Cuisine"] = df.apply(
#     predict_cuisine,
#     axis=1
# )

# df["Cuisine_Match"] = (
#     df["Predicted_Cuisine"]
#     ==
#     df["Cuisine"]
# )

cuisines = sorted(df["Cuisine"].dropna().unique().tolist())

cuisine_embeddings = model.encode(
    cuisines,
    convert_to_tensor=False
)

def predict_cuisine(description):

    if description.strip() == "":
        return ""

    description_embedding = model.encode(
        [description],
        convert_to_tensor=False
    )

    similarities = cosine_similarity(
        description_embedding,
        cuisine_embeddings
    )[0]

    return cuisines[np.argmax(similarities)]

df["Predicted_Cuisine"] = df["Description"].apply(
    predict_cuisine
)

df["Cuisine_Match"] = (
    df["Predicted_Cuisine"]
    ==
    df["Cuisine"]
)

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

    for j in range(i+1,len(df)):

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

semantic_cols = [
    "Recipe Name",
    "Ingredients",
    "Directions",
    "Description"
]

missing_report = df[semantic_cols].isnull().sum()


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

print("Completed.")
print("Results saved.")
