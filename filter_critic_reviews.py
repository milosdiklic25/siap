import pandas as pd
from pathlib import Path


def main():
    movies_path = Path("movies_after_1970_best_picture_nominees.csv")
    reviews_path = Path("critic_reviews.csv")
    output_path = Path("critic_reviews_filtered_cleaned.csv")

    print("=== Critic Reviews Filter + Clean Script ===")
    print(f"[INFO] Working directory: {Path.cwd()}")
    print(f"[INFO] Movies input file:  {movies_path.resolve()}")
    print(f"[INFO] Reviews input file: {reviews_path.resolve()}")
    print(f"[INFO] Output file:       {output_path.resolve()}")

    if not movies_path.exists():
        raise FileNotFoundError(f"[ERROR] Movies CSV not found: {movies_path.resolve()}")
    if not reviews_path.exists():
        raise FileNotFoundError(f"[ERROR] Reviews CSV not found: {reviews_path.resolve()}")

    # 1) Load datasets
    print("[INFO] Loading datasets...")
    movies = pd.read_csv(movies_path)
    reviews = pd.read_csv(reviews_path)
    print(f"[INFO] Loaded movies:  {len(movies):,} rows, {movies.shape[1]} columns.")
    print(f"[INFO] Loaded reviews: {len(reviews):,} rows, {reviews.shape[1]} columns.")

    # Basic schema sanity checks (no functionality change)
    if "movieId" not in movies.columns:
        raise KeyError(f"[ERROR] Missing required column 'movieId' in movies CSV. Found: {list(movies.columns)}")
    if "movieId" not in reviews.columns:
        raise KeyError(f"[ERROR] Missing required column 'movieId' in reviews CSV. Found: {list(reviews.columns)}")

    # 2) Filter to only movies in nominees dataset (kept identical)
    print("[INFO] Building allowed movieId set from movies dataset...")
    allowed_ids = set(movies["movieId"].astype(str).str.strip())
    print(f"[INFO] Allowed movieIds: {len(allowed_ids):,}")

    print("[INFO] Filtering reviews to allowed movieIds...")
    filtered_reviews = reviews[
        reviews["movieId"].astype(str).str.strip().isin(allowed_ids)
    ].copy()

    print(f"[INFO] Original reviews: {len(reviews):,}")
    print(f"[INFO] Filtered reviews: {len(filtered_reviews):,}")
    print(f"[INFO] Removed reviews:  {len(reviews) - len(filtered_reviews):,}")

    # 3) Drop columns safely (kept identical)
    columns_to_drop = [
        "reviewId",
        "criticName",
        "criticPageUrl",
        "publicationUrl",
        "publicationName",
        "reviewUrl",
        "isRtUrl",
        "creationDate",
    ]

    print("[INFO] Dropping unnecessary columns (ignore if missing)...")
    present_to_drop = [col for col in columns_to_drop if col in filtered_reviews.columns]
    missing_to_drop = [col for col in columns_to_drop if col not in filtered_reviews.columns]
    if present_to_drop:
        print(f"[INFO] Columns dropped: {present_to_drop}")
    if missing_to_drop:
        print(f"[INFO] Columns not present (skipped): {missing_to_drop}")

    filtered_reviews = filtered_reviews.drop(columns=present_to_drop)

    # 4) Save cleaned dataset
    print(f"[INFO] Saving cleaned reviews to {output_path} ...")
    filtered_reviews.to_csv(output_path, index=False)

    # Confirm write + final schema
    if output_path.exists():
        print(f"[INFO] Saved successfully. File size: {output_path.stat().st_size:,} bytes.")
    print(f"[INFO] Remaining columns: {list(filtered_reviews.columns)}")
    print("[DONE] Critic reviews filtering/cleaning complete.")


if __name__ == "__main__":
    main()