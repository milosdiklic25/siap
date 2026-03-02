import pandas as pd
import re
import numpy as np
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def main():
    input_path = Path("critic_reviews_normalized.csv")
    output_path = Path("critic_reviews_normalized_textprocessed.csv")

    print("=== Critic Review Text Preprocessing Script ===")
    print(f"[INFO] Working directory: {Path.cwd()}")
    print(f"[INFO] Input file:  {input_path.resolve()}")
    print(f"[INFO] Output file: {output_path.resolve()}")

    if not input_path.exists():
        raise FileNotFoundError(f"[ERROR] Input CSV not found: {input_path.resolve()}")

    # Download stopwords if not present (safe to call)
    print("[INFO] Ensuring NLTK stopwords are available...")
    nltk.download("stopwords", quiet=True)

    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    print(f"[INFO] Loaded stopwords: {len(stop_words):,}")

    def preprocess_quote(text):
        if pd.isna(text):
            return np.nan

        # 1) lowercase
        text = str(text).lower()

        # 2) remove special characters (keep letters/numbers/spaces)
        text = re.sub(r"[^a-z0-9\s]", " ", text)

        # 3) normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # 4) tokenize
        tokens = text.split()

        # 5) remove stopwords
        tokens = [t for t in tokens if t not in stop_words]

        # 6) stemming
        tokens = [stemmer.stem(t) for t in tokens]

        return " ".join(tokens)

    # 1) Load the normalized reviews
    print("[INFO] Loading normalized reviews...")
    df = pd.read_csv(input_path)
    print(f"[INFO] Loaded {len(df):,} rows, {df.shape[1]} columns.")

    # Basic schema sanity checks (no functionality change)
    if "quote" not in df.columns:
        raise KeyError(f"[ERROR] Missing required column 'quote'. Found: {list(df.columns)}")

    # 2) Process quote column
    print("[INFO] Preprocessing 'quote' text...")
    quote_na_before = int(df["quote"].isna().sum())
    df["quote_processed"] = df["quote"].apply(preprocess_quote)
    quote_processed_na = int(df["quote_processed"].isna().sum())

    print(f"[INFO] quote NA: {quote_na_before:,}")
    print(f"[INFO] quote_processed NA: {quote_processed_na:,}")

    # 3) Save result
    print(f"[INFO] Saving output to {output_path} ...")
    df.to_csv(output_path, index=False)

    # Confirm write + quick preview
    if output_path.exists():
        print(f"[INFO] Saved successfully. File size: {output_path.stat().st_size:,} bytes.")

    print("[DONE] Text preprocessing complete.")
    print("[INFO] Columns now:", list(df.columns))
    print("\n[INFO] Example (first 3 rows):")
    print(df[["quote", "quote_processed"]].head(3).to_string(index=False))


if __name__ == "__main__":
    main()