import pandas as pd
import re
import numpy as np

# Load datasets
movies = pd.read_csv("movies_after_1970_best_picture_nominees.csv")
reviews = pd.read_csv("critic_reviews_filtered_cleaned.csv")

# Filter reviews
allowed_ids = set(movies["movieId"].astype(str).str.strip())
filtered_reviews = reviews[
    reviews["movieId"].astype(str).str.strip().isin(allowed_ids)
].copy()

# Letter grade mapping (to /10 scale)
letter_map = {
    "A+": 10, "A": 9.5, "A-": 9,
    "B+": 8.5, "B": 8, "B-": 7.5,
    "C+": 7, "C": 6.5, "C-": 6,
    "D+": 5.5, "D": 5,
    "F": 4
}

def normalize_score(score):
    if pd.isna(score):
        return np.nan

    score = str(score).strip()

    # 1️⃣ Fraction format (e.g. 3.5/4, 9/10)
    if re.match(r"^\d+(\.\d+)?/\d+(\.\d+)?$", score):
        num, denom = map(float, score.split("/"))
        return (num / denom) * 10

    # 2️⃣ Letter grades
    if score in letter_map:
        return letter_map[score]

    # 3️⃣ Custom -4 to +4 scale
    match = re.search(r"([+-]?\d+)\s*out of\s*(-?\d+)\.\.(\+?\d+)", score)
    if match:
        value = float(match.group(1))
        min_val = float(match.group(2))
        max_val = float(match.group(3))
        return ((value - min_val) / (max_val - min_val)) * 10

    return np.nan  # Unknown format

# Apply normalization
filtered_reviews["normalized_score_10"] = filtered_reviews["originalScore"].apply(normalize_score)

# Save
filtered_reviews.to_csv("critic_reviews_normalized.csv", index=False)

print("Normalization complete.")
print(filtered_reviews["normalized_score_10"].describe())