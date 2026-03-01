import pandas as pd

# Load datasets
best_picture = pd.read_csv("best_picture_after_1970.csv")
movies = pd.read_csv("movies_after_1970_best_picture_nominees.csv")

# --- Clean movie titles for safer matching ---
best_picture["Film_clean"] = best_picture["Film"].str.strip().str.lower()
movies["movieTitle_clean"] = movies["movieTitle"].str.strip().str.lower()

# --- Convert Winner column to proper boolean ---
# In your dataset: True = winner, empty = not winner
best_picture["winner"] = best_picture["Winner"].apply(
    lambda x: True if str(x).strip().lower() == "true" else False
)

# Keep only necessary columns for merging
winner_df = best_picture[["Film_clean", "winner"]].drop_duplicates()

# --- Merge on cleaned movie title ---
merged = movies.merge(
    winner_df,
    left_on="movieTitle_clean",
    right_on="Film_clean",
    how="left"
)

# Replace NaN (if no match found) with False
merged["winner"] = merged["winner"].fillna(False)

# Drop helper columns
merged = merged.drop(columns=["movieTitle_clean", "Film_clean"])

# Save result
merged.to_csv("movies_after_1970_best_picture_nominees_with_winner.csv", index=False)

print("Merge complete.")
