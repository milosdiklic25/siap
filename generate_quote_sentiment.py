import pandas as pd
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER lexicon if missing (safe to call)
nltk.download("vader_lexicon", quiet=True)

# Load input file
df = pd.read_csv("critic_reviews_normalized_textprocessed.csv")

# Initialize VADER sentiment analyzer
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

# Choose which text column to analyze:
# - If you want original quote: use df["quote"]
# - If you want preprocessed text: use df["quote_processed"]
text_col = "quote_processed" if "quote_processed" in df.columns else "quote"

df[["nltk_sentiment_label", "nltk_sentiment_compound"]] = df[text_col].apply(
    lambda x: pd.Series(vader_sentiment(x))
)

# Save output
df.to_csv("critic_reviews_normalized_textprocessed_sentiment.csv", index=False)

print("Sentiment analysis complete.")
print("Using column:", text_col)
print(df[["nltk_sentiment_label", "nltk_sentiment_compound"]].value_counts().head(10))
print("\nCompound score summary:")
print(df["nltk_sentiment_compound"].describe())