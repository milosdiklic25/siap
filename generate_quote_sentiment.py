import pandas as pd
import numpy as np
from pathlib import Path

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


def main():
    input_path = Path("critic_reviews_normalized_textprocessed.csv")
    output_path = Path("critic_reviews_normalized_textprocessed_sentiment.csv")

    print("=== VADER Sentiment Analysis Script ===")
    print(f"[INFO] Working directory: {Path.cwd()}")
    print(f"[INFO] Input file:  {input_path.resolve()}")
    print(f"[INFO] Output file: {output_path.resolve()}")

    if not input_path.exists():
        raise FileNotFoundError(f"[ERROR] Input CSV not found: {input_path.resolve()}")

    # Download VADER lexicon if missing (safe to call)
    print("[INFO] Ensuring NLTK VADER lexicon is available...")
    nltk.download("vader_lexicon", quiet=True)

    # Load input file
    print("[INFO] Loading input dataset...")
    df = pd.read_csv(input_path)
    print(f"[INFO] Loaded {len(df):,} rows, {df.shape[1]} columns.")

    # Initialize VADER sentiment analyzer
    print("[INFO] Initializing VADER SentimentIntensityAnalyzer...")
    sia = SentimentIntensityAnalyzer()

    def vader_sentiment(text):
        if pd.isna(text):
            return (np.nan, np.nan)

        scores = sia.polarity_scores(str(text))
        compound = scores["compound"]

        # Common thresholds for VADER
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        return (label, compound)

    # Choose which text column to analyze (kept identical behavior)
    text_col = "quote_processed" if "quote_processed" in df.columns else "quote"
    print(f"[INFO] Using column for sentiment analysis: {text_col}")

    if text_col not in df.columns:
        raise KeyError(
            f"[ERROR] Neither 'quote_processed' nor 'quote' exists in the dataset. "
            f"Found columns: {list(df.columns)}"
        )

    # Apply sentiment
    print("[INFO] Computing VADER sentiment (label + compound)...")
    na_before = int(df[text_col].isna().sum())
    df[["nltk_sentiment_label", "nltk_sentiment_compound"]] = df[text_col].apply(
        lambda x: pd.Series(vader_sentiment(x))
    )
    na_after = int(df["nltk_sentiment_compound"].isna().sum())

    print(f"[INFO] NA in {text_col}: {na_before:,}")
    print(f"[INFO] NA in nltk_sentiment_compound: {na_after:,}")

    # Save output
    print(f"[INFO] Saving output to {output_path} ...")
    df.to_csv(output_path, index=False)

    # Confirm write + summaries
    if output_path.exists():
        print(f"[INFO] Saved successfully. File size: {output_path.stat().st_size:,} bytes.")

    print("[DONE] Sentiment analysis complete.")
    print("[INFO] Top sentiment label/compound combinations:")
    print(df[["nltk_sentiment_label", "nltk_sentiment_compound"]].value_counts().head(10).to_string())

    print("\n[INFO] Compound score summary:")
    print(df["nltk_sentiment_compound"].describe())


if __name__ == "__main__":
    main()