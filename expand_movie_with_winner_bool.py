import pandas as pd
from pathlib import Path


def main():
    best_picture_path = Path("best_picture_after_1970.csv")
    movies_path = Path("movies_after_1970_best_picture_nominees.csv")
    output_path = Path("movies_after_1970_best_picture_nominees_with_winner.csv")

    print("=== Best Picture Winner Merge Script ===")
    print(f"[INFO] Working directory: {Path.cwd()}")
    print(f"[INFO] Best Picture input file: {best_picture_path.resolve()}")
    print(f"[INFO] Movies input file:       {movies_path.resolve()}")
    print(f"[INFO] Output file:            {output_path.resolve()}")

    if not best_picture_path.exists():
        raise FileNotFoundError(f"[ERROR] Best Picture CSV not found: {best_picture_path.resolve()}")
    if not movies_path.exists():
        raise FileNotFoundError(f"[ERROR] Movies CSV not found: {movies_path.resolve()}")

    # 1) Load datasets
    print("[INFO] Loading datasets...")
    best_picture = pd.read_csv(best_picture_path)
    movies = pd.read_csv(movies_path)
    print(f"[INFO] Loaded best_picture: {len(best_picture):,} rows, {best_picture.shape[1]} columns.")
    print(f"[INFO] Loaded movies:       {len(movies):,} rows, {movies.shape[1]} columns.")

    # Basic schema sanity checks (no functionality change)
    required_bp_cols = ["Film", "Winner"]
    missing_bp = [c for c in required_bp_cols if c not in best_picture.columns]
    if missing_bp:
        raise KeyError(
            f"[ERROR] Missing required columns in best_picture CSV: {missing_bp}. "
            f"Found: {list(best_picture.columns)}"
        )

    required_movies_cols = ["movieTitle"]
    missing_movies = [c for c in required_movies_cols if c not in movies.columns]
    if missing_movies:
        raise KeyError(
            f"[ERROR] Missing required columns in movies CSV: {missing_movies}. "
            f"Found: {list(movies.columns)}"
        )

    # 2) Clean movie titles for safer matching (kept identical)
    print("[INFO] Cleaning title columns for matching (strip/lower)...")
    best_picture["Film_clean"] = best_picture["Film"].str.strip().str.lower()
    movies["movieTitle_clean"] = movies["movieTitle"].str.strip().str.lower()

    # 3) Convert Winner column to proper boolean (kept identical)
    print("[INFO] Converting best_picture['Winner'] to boolean 'winner'...")
    best_picture["winner"] = best_picture["Winner"].apply(
        lambda x: True if str(x).strip().lower() == "true" else False
    )

    # Quick stats
    winner_true = int(best_picture["winner"].sum())
    print(f"[INFO] Winner==True count in best_picture: {winner_true:,} (out of {len(best_picture):,})")

    # 4) Keep only necessary columns for merging (kept identical)
    print("[INFO] Preparing winner lookup table...")
    winner_df = best_picture[["Film_clean", "winner"]].drop_duplicates()
    print(f"[INFO] Winner lookup rows after drop_duplicates: {len(winner_df):,}")

    # 5) Merge on cleaned movie title (kept identical)
    print("[INFO] Merging movies with winner lookup...")
    merged = movies.merge(
        winner_df,
        left_on="movieTitle_clean",
        right_on="Film_clean",
        how="left"
    )

    # 6) Replace NaN (if no match found) with False (kept identical)
    print("[INFO] Filling missing winner values with False...")
    missing_before = int(merged["winner"].isna().sum())
    merged["winner"] = merged["winner"].fillna(False)
    print(f"[INFO] Missing winner before fill: {missing_before:,} | after fill: {int(merged['winner'].isna().sum()):,}")

    # Optional sanity check: how many titles did not match?
    # (This is logging only; does not alter any results.)
    unmatched = int((merged["Film_clean"].isna()).sum())
    print(f"[INFO] Unmatched rows (no Film_clean found on merge): {unmatched:,} / {len(merged):,}")

    # 7) Drop helper columns (kept identical)
    print("[INFO] Dropping helper columns...")
    merged = merged.drop(columns=["movieTitle_clean", "Film_clean"])

    # 8) Save result (kept identical)
    print(f"[INFO] Saving output to {output_path} ...")
    merged.to_csv(output_path, index=False)

    # Confirm write
    if output_path.exists():
        print(f"[INFO] Saved successfully. File size: {output_path.stat().st_size:,} bytes.")
    print("[DONE] Merge complete.")


if __name__ == "__main__":
    main()