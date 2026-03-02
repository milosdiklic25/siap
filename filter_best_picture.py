import pandas as pd
from pathlib import Path

def main():
    input_path = Path("full_data.csv")
    output_path = Path("best_picture_after_1970.csv")

    print("=== Best Picture Filter Script ===")
    print(f"[INFO] Working directory: {Path.cwd()}")
    print(f"[INFO] Input file:  {input_path.resolve()}")
    print(f"[INFO] Output file: {output_path.resolve()}")

    if not input_path.exists():
        raise FileNotFoundError(f"[ERROR] Input CSV not found: {input_path.resolve()}")

    # 1. Load dataset
    print("[INFO] Loading dataset...")
    data = pd.read_csv(input_path)
    print(f"[INFO] Loaded {len(data):,} rows, {data.shape[1]} columns.")

    # Basic schema sanity checks
    required_cols = ["CanonicalCategory", "Year"]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise KeyError(f"[ERROR] Missing required columns: {missing}. Found: {list(data.columns)}")

    # 2. Clean columns (safe practice)
    print("[INFO] Cleaning columns...")
    # Category cleanup
    data["CanonicalCategory"] = data["CanonicalCategory"].astype(str).str.strip().str.upper()

    # Year cleanup
    year_before_na = data["Year"].isna().sum()
    data["Year"] = pd.to_numeric(data["Year"], errors="coerce")
    year_after_na = data["Year"].isna().sum()
    newly_coerced = year_after_na - year_before_na

    print(f"[INFO] Year NA before: {year_before_na:,} | after coercion: {year_after_na:,} "
          f"(newly coerced invalid years: {newly_coerced:,})")

    # Quick category check
    category_counts = data["CanonicalCategory"].value_counts(dropna=False)
    best_picture_count = int(category_counts.get("BEST PICTURE", 0))
    print(f"[INFO] Rows with CanonicalCategory == 'BEST PICTURE': {best_picture_count:,}")

    # 3. Apply both filters
    print("[INFO] Applying filters: category='BEST PICTURE' AND year>=1970 ...")
    filtered_data = data[
        (data["CanonicalCategory"] == "BEST PICTURE") &
        (data["Year"] >= 1970)
    ]

    print(f"[INFO] Filtered rows: {len(filtered_data):,}")

    if len(filtered_data) > 0:
        min_year = int(filtered_data["Year"].min())
        max_year = int(filtered_data["Year"].max())
        print(f"[INFO] Filtered Year range: {min_year}–{max_year}")
        print("[INFO] Sample (first 5 rows):")
        print(filtered_data.head(5).to_string(index=False))
    else:
        print("[WARN] No rows matched the filters. Output CSV will be empty.")

    # 4. Save to new CSV
    print(f"[INFO] Saving to {output_path} ...")
    filtered_data.to_csv(output_path, index=False)

    # Confirm write
    if output_path.exists():
        print(f"[INFO] Saved successfully. File size: {output_path.stat().st_size:,} bytes.")
    print("[DONE] Filtering complete.")

if __name__ == "__main__":
    main()