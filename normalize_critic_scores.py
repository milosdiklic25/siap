import pandas as pd
import re
import numpy as np
from pathlib import Path


def main():
    movies_path = Path("movies_after_1970_best_picture_nominees.csv")
    reviews_path = Path("critic_reviews_filtered_cleaned.csv")
    output_path = Path("critic_reviews_normalized.csv")

    print("=== Critic Review Score Normalization Script ===")
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
    if "originalScore" not in reviews.columns:
        raise KeyError(f"[ERROR] Missing required column 'originalScore' in reviews CSV. Found: {list(reviews.columns)}")

    # 2) Filter reviews (kept identical logic)
    print("[INFO] Building allowed movieId set from movies dataset...")
    allowed_ids = set(movies["movieId"].astype(str).str.strip())
    print(f"[INFO] Allowed movieIds: {len(allowed_ids):,}")

    print("[INFO] Filtering reviews to allowed movieIds...")
    filtered_reviews = reviews[
        reviews["movieId"].astype(str).str.strip().isin(allowed_ids)
    ].copy()

    print(f"[INFO] Reviews before filter: {len(reviews):,}")
    print(f"[INFO] Reviews after filter:  {len(filtered_reviews):,}")
    print(f"[INFO] Removed reviews:       {len(reviews) - len(filtered_reviews):,}")

    # 3) Letter grade mapping (to /10 scale) (kept identical)
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

        # 1) Fraction format (e.g. 3.5/4, 9/10)
        if re.match(r"^\d+(\.\d+)?/\d+(\.\d+)?$", score):
            num, denom = map(float, score.split("/"))
            return (num / denom) * 10

        # 2) Letter grades
        if score in letter_map:
            return letter_map[score]

        # 3) Custom -4 to +4 scale
        match = re.search(r"([+-]?\d+)\s*out of\s*(-?\d+)\.\.(\+?\d+)", score)
        if match:
            value = float(match.group(1))
            min_val = float(match.group(2))
            max_val = float(match.group(3))
            return ((value - min_val) / (max_val - min_val)) * 10

        return np.nan  # Unknown format

    # 4) Apply normalization (kept identical)
    print("[INFO] Normalizing 'originalScore' to a /10 scale...")
    na_before = int(filtered_reviews["originalScore"].isna().sum())
    filtered_reviews["normalized_score_10"] = filtered_reviews["originalScore"].apply(normalize_score)
    na_after = int(filtered_reviews["normalized_score_10"].isna().sum())

    print(f"[INFO] originalScore NA: {na_before:,}")
    print(f"[INFO] normalized_score_10 NA (includes unknown formats): {na_after:,}")

    # Helpful breakdown (logging only; no behavior change)
    total = len(filtered_reviews)
    parsed = total - na_after
    print(f"[INFO] Parsed scores: {parsed:,} / {total:,} ({(parsed/total*100.0) if total else 0.0:.2f}%)")

    # 5) Save
    print(f"[INFO] Saving normalized reviews to {output_path} ...")
    filtered_reviews.to_csv(output_path, index=False)

    # Confirm write + summary stats
    if output_path.exists():
        print(f"[INFO] Saved successfully. File size: {output_path.stat().st_size:,} bytes.")

    print("[INFO] Normalization complete. Summary stats for normalized_score_10:")
    print(filtered_reviews["normalized_score_10"].describe())


if __name__ == "__main__":
    main()