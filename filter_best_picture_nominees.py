import pandas as pd
from pathlib import Path


def main():
    movies_path = Path("movies_after_1970.csv")
    best_picture_path = Path("best_picture_after_1970.csv")
    output_path = Path("movies_after_1970_best_picture_nominees.csv")

    print("=== Best Picture Nominee Match Script ===")
    print(f"[INFO] Working directory: {Path.cwd()}")
    print(f"[INFO] Movies input file:       {movies_path.resolve()}")
    print(f"[INFO] Best Picture input file: {best_picture_path.resolve()}")
    print(f"[INFO] Output file:            {output_path.resolve()}")

    if not movies_path.exists():
        raise FileNotFoundError(f"[ERROR] Movies CSV not found: {movies_path.resolve()}")
    if not best_picture_path.exists():
        raise FileNotFoundError(f"[ERROR] Best Picture CSV not found: {best_picture_path.resolve()}")

    # 1) Load both CSVs
    print("[INFO] Loading movies dataset...")
    movies = pd.read_csv(movies_path)
    print(f"[INFO] Loaded movies: {len(movies):,} rows, {movies.shape[1]} columns.")

    print("[INFO] Loading Best Picture dataset...")
    best_picture = pd.read_csv(best_picture_path)
    print(f"[INFO] Loaded best_picture: {len(best_picture):,} rows, {best_picture.shape[1]} columns.")

    # Basic schema sanity checks (do not change functionality)
    required_movies_cols = ["movieTitle"]
    missing_movies = [c for c in required_movies_cols if c not in movies.columns]
    if missing_movies:
        raise KeyError(
            f"[ERROR] Missing required columns in movies CSV: {missing_movies}. "
            f"Found: {list(movies.columns)}"
        )

    required_bp_cols = ["Film"]
    missing_bp = [c for c in required_bp_cols if c not in best_picture.columns]
    if missing_bp:
        raise KeyError(
            f"[ERROR] Missing required columns in best_picture CSV: {missing_bp}. "
            f"Found: {list(best_picture.columns)}"
        )

    # 2) Extract nominee names from BEST PICTURE file (kept identical to your original logic)
    print("[INFO] Extracting nominee names from best_picture['Film'] (strip/lower/unique)...")
    nominee_names = (
        best_picture["Film"]
        .astype(str)
        .str.strip()
        .str.lower()
        .dropna()
        .unique()
    )
    print(f"[INFO] Total unique nominee names extracted: {len(nominee_names):,}")
    print("[INFO] Preview nominee names (first 10):")
    print(nominee_names[:10])

    # 3) Normalize movie titles in movies dataset for matching (kept identical)
    print("[INFO] Normalizing movies['movieTitle'] for matching (strip/lower)...")
    movies["movieTitle_norm"] = movies["movieTitle"].astype(str).str.strip().str.lower()

    # 4) Keep only movies whose title appears in nominee names (kept identical)
    print("[INFO] Filtering movies by nominee titles...")
    movies_nominees_only = movies[movies["movieTitle_norm"].isin(nominee_names)].copy()

    kept = len(movies_nominees_only)
    total = len(movies)
    pct = (kept / total * 100.0) if total else 0.0
    print(f"[INFO] Done. Kept {kept:,} / {total:,} movies ({pct:.2f}%).")

    # 5) Drop helper column and save (kept identical)
    print("[INFO] Dropping helper column and saving output...")
    movies_nominees_only.drop(columns=["movieTitle_norm"], inplace=True)
    movies_nominees_only.to_csv(output_path, index=False)

    # Confirm write
    if output_path.exists():
        print(f"[INFO] Saved successfully. File size: {output_path.stat().st_size:,} bytes.")
    print("[DONE] Saved as 'movies_after_1970_best_picture_nominees.csv'.")


if __name__ == "__main__":
    main()