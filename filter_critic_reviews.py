import pandas as pd

# Load datasets
movies = pd.read_csv("movies_after_1970_best_picture_nominees.csv")
reviews = pd.read_csv("critic_reviews.csv")

# Filter to only movies in nominees dataset
allowed_ids = set(movies["movieId"].astype(str).str.strip())
filtered_reviews = reviews[
    reviews["movieId"].astype(str).str.strip().isin(allowed_ids)
].copy()

# Columns to drop
columns_to_drop = [
    "reviewId",
    "criticName",
    "criticPageUrl",
    "publicationUrl",
    "publicationName",
    "reviewUrl",
    "isRtUrl",
    "creationDate"
]

# Drop columns safely (ignore if missing)
filtered_reviews = filtered_reviews.drop(
    columns=[col for col in columns_to_drop if col in filtered_reviews.columns]
)

# Save cleaned dataset
filtered_reviews.to_csv("critic_reviews_filtered_cleaned.csv", index=False)

print(f"Original reviews: {len(reviews):,}")
print(f"Filtered reviews: {len(filtered_reviews):,}")
print(f"Removed reviews:  {len(reviews) - len(filtered_reviews):,}")
print(f"Remaining columns: {list(filtered_reviews.columns)}")