import pandas as pd
import re
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords if not present (safe to call)
nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

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

# Load the normalized reviews
df = pd.read_csv("critic_reviews_normalized.csv")

# Process quote column
df["quote_processed"] = df["quote"].apply(preprocess_quote)

# Save result
df.to_csv("critic_reviews_normalized_textprocessed.csv", index=False)

print("Text preprocessing complete.")
print("Columns now:", list(df.columns))
print("\nExample:")
print(df[["quote", "quote_processed"]].head(3))