import pandas as pd
from pathlib import Path

def main():
    # --- Config ---
    INPUT_CSV = Path("movies.csv")
    OUTPUT_CSV = Path("movies_after_1970.csv")
    YEAR_CUTOFF = 1970

    print("=== Movies After 1970 Filter ===")
    print(f"[INFO] Working directory: {Path.cwd()}")
    print(f"[INFO] Input:  {INPUT_CSV.resolve()}")
    print(f"[INFO] Output: {OUTPUT_CSV.resolve()}")
    print(f"[INFO] Year cutoff: {YEAR_CUTOFF}")

    # 1) Load the dataset
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"[ERROR] Input file not found: {INPUT_CSV.resolve()}")

    print("[INFO] Loading dataset...")
    movies = pd.read_csv(INPUT_CSV)
    print(f"[INFO] Loaded {len(movies):,} rows and {movies.shape[1]} columns.")
    print(f"[INFO] Columns: {list(movies.columns)}")

    # Basic column check
    if "movieYear" not in movies.columns:
        raise KeyError("[ERROR] Required column 'movieYear' not found in movies.csv")

    # 2) Ensure movieYear is numeric (helps if the CSV has strings like '1970', 'N/A', etc.)
    print("[INFO] Converting 'movieYear' to numeric...")
    year_na_before = movies["movieYear"].isna().sum()
    movies["movieYear"] = pd.to_numeric(movies["movieYear"], errors="coerce")
    year_na_after = movies["movieYear"].isna().sum()
    print(f"[INFO] movieYear NA before: {year_na_before:,} | after coercion: {year_na_after:,} "
        f"(newly coerced invalid years: {year_na_after - year_na_before:,})")

    # Useful quick stats
    if movies["movieYear"].notna().any():
        print(f"[INFO] movieYear range (overall): {int(movies['movieYear'].min())}–{int(movies['movieYear'].max())}")
    else:
        print("[WARN] All movieYear values are NaN after conversion.")

    # 3) Remove movies older than 1970 (same filter intent)
    print(f"[INFO] Filtering movies with movieYear >= {YEAR_CUTOFF} ...")
    before_rows = len(movies)
    filtered_movies = movies[movies["movieYear"] >= YEAR_CUTOFF]
    after_rows = len(filtered_movies)
    print(f"[INFO] Rows before: {before_rows:,} | after: {after_rows:,} | removed: {before_rows - after_rows:,}")

    # Preview
    print("[INFO] Preview of filtered data (first 5 rows):")

    # 4) Save the new dataset to a CSV file
    print(f"[INFO] Saving to {OUTPUT_CSV} ...")
    filtered_movies.to_csv(OUTPUT_CSV, index=False)

    if OUTPUT_CSV.exists():
        print(f"[INFO] Saved successfully. File size: {OUTPUT_CSV.stat().st_size:,} bytes.")

    print("[DONE] Filtering complete. New dataset saved as 'movies_after_1970.csv'.")
    
if __name__ == "__main__":
    main()