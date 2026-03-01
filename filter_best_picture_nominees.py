import pandas as pd

# Load both CSVs
movies = pd.read_csv("movies_after_1970.csv")
best_picture = pd.read_csv("best_picture_after_1970.csv")

# Extract nominee names from the BEST PICTURE file
nominee_names = (
    best_picture["Film"]
    .astype(str)
    .str.strip()
    .str.lower()
    .dropna()
    .unique()
)
print(nominee_names[:10])  # Print first 10 nominee names for verification
# Normalize movie titles in movies dataset for matching
movies["movieTitle_norm"] = movies["movieTitle"].astype(str).str.strip().str.lower()

# Keep only movies whose title appears in nominee names
movies_nominees_only = movies[movies["movieTitle_norm"].isin(nominee_names)].copy()

# Drop helper column and save
movies_nominees_only.drop(columns=["movieTitle_norm"], inplace=True)
movies_nominees_only.to_csv("movies_after_1970_best_picture_nominees.csv", index=False)

print(
    f"Done. Kept {len(movies_nominees_only)} / {len(movies)} movies. "
    "Saved as 'movies_after_1970_best_picture_nominees.csv'."
)