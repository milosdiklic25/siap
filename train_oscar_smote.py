#!/usr/bin/env python3
"""
Train an Oscar Best Picture winner classifier using SMOTE to address class imbalance.

Inputs (CSV):
1) moview_after_1970_beset_picture_nominees_with_winner.csv
   columns (at least):
   movieId,movieYear,critic_score,critic_sentiment,audience_score,audience_sentiment,
   release_date_theaters,release_date_streaming,rating,original_language,runtime,winner

2) critic_reviews_normalized_textprocessed_sentiment.csv
   columns (at least):
   movieId,reviewState,isFresh,isRotten,isTopCritic,normalized_score_10,
   nltk_sentiment_label,nltk_sentiment_compound

What it does:
- Aggregates review-level info into movie-level features
- Merges with nominees dataset
- Cleans/parses runtime to minutes
- Encodes categoricals
- Time-based split by year (train <= split_year, test > split_year)
- Applies StandardScaler + SMOTE (train only) + SVM (or Logistic Regression)
- Evaluates with classification metrics + "per-year winner pick accuracy"
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler


# ----------------------------
# Helpers
# ----------------------------

def parse_runtime_to_minutes(runtime_val: str) -> float:
    """
    Convert runtime like '2h 13m' to minutes (133).
    Returns np.nan if cannot parse.
    """
    if pd.isna(runtime_val):
        return np.nan
    s = str(runtime_val).strip().lower()

    # Common patterns: "2h 13m", "2h", "133m", "2 h 13 m"
    h_match = re.search(r"(\d+)\s*h", s)
    m_match = re.search(r"(\d+)\s*m", s)

    hours = int(h_match.group(1)) if h_match else 0
    mins = int(m_match.group(1)) if m_match else 0

    if not h_match and not m_match:
        # Sometimes runtime might be plain number of minutes
        num_match = re.fullmatch(r"\d+", s)
        if num_match:
            return float(int(num_match.group(0)))
        return np.nan

    return float(hours * 60 + mins)


def safe_bool_series(x: pd.Series) -> pd.Series:
    """
    Convert True/False-like values to 0/1 ints.
    """
    if x.dtype == bool:
        return x.astype(int)
    # handle strings "True"/"False"
    return x.astype(str).str.lower().map({"true": 1, "false": 0}).fillna(0).astype(int)


def aggregate_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate review-level dataset into movie-level numeric features.
    """
    df = reviews.copy()

    # Ensure expected columns exist (if missing, create safe defaults)
    for col in ["isFresh", "isRotten", "isTopCritic"]:
        if col not in df.columns:
            df[col] = False

    for col in ["normalized_score_10", "nltk_sentiment_compound"]:
        if col not in df.columns:
            df[col] = np.nan

    df["isFresh_i"] = safe_bool_series(df["isFresh"])
    df["isRotten_i"] = safe_bool_series(df["isRotten"])
    df["isTopCritic_i"] = safe_bool_series(df["isTopCritic"])

    # Review state proportions (fresh/rotten often overlap with isFresh/isRotten; keep both)
    # NLTK sentiment label ratios
    if "nltk_sentiment_label" in df.columns:
        df["nltk_pos_i"] = (df["nltk_sentiment_label"].astype(str).str.lower() == "positive").astype(int)
        df["nltk_neu_i"] = (df["nltk_sentiment_label"].astype(str).str.lower() == "neutral").astype(int)
        df["nltk_neg_i"] = (df["nltk_sentiment_label"].astype(str).str.lower() == "negative").astype(int)
    else:
        df["nltk_pos_i"] = 0
        df["nltk_neu_i"] = 0
        df["nltk_neg_i"] = 0

    agg = df.groupby("movieId").agg(
        review_count=("movieId", "size"),
        fresh_count=("isFresh_i", "sum"),
        rotten_count=("isRotten_i", "sum"),
        top_critic_count=("isTopCritic_i", "sum"),
        avg_norm_score_10=("normalized_score_10", "mean"),
        std_norm_score_10=("normalized_score_10", "std"),
        avg_nltk_compound=("nltk_sentiment_compound", "mean"),
        std_nltk_compound=("nltk_sentiment_compound", "std"),
        nltk_pos_count=("nltk_pos_i", "sum"),
        nltk_neu_count=("nltk_neu_i", "sum"),
        nltk_neg_count=("nltk_neg_i", "sum"),
    ).reset_index()

    # Ratios (avoid division by 0)
    eps = 1e-9
    agg["fresh_ratio"] = agg["fresh_count"] / (agg["review_count"] + eps)
    agg["rotten_ratio"] = agg["rotten_count"] / (agg["review_count"] + eps)
    agg["top_critic_ratio"] = agg["top_critic_count"] / (agg["review_count"] + eps)
    agg["nltk_pos_ratio"] = agg["nltk_pos_count"] / (agg["review_count"] + eps)
    agg["nltk_neu_ratio"] = agg["nltk_neu_count"] / (agg["review_count"] + eps)
    agg["nltk_neg_ratio"] = agg["nltk_neg_count"] / (agg["review_count"] + eps)

    # Fill std NaNs for single-review movies
    for col in ["std_norm_score_10", "std_nltk_compound"]:
        agg[col] = agg[col].fillna(0.0)

    return agg


def per_year_top_pick_accuracy(test_df: pd.DataFrame, proba: np.ndarray, year_col: str = "movieYear") -> float:
    """
    For each year, pick the movie with the highest predicted probability of winner
    and check if it is actually the winner. Returns accuracy over years.
    """
    tmp = test_df[[year_col, "winner"]].copy()
    tmp["proba"] = proba

    year_groups = tmp.groupby(year_col)
    correct = 0
    total = 0

    for year, g in year_groups:
        total += 1
        # pick highest proba row
        top_idx = g["proba"].idxmax()
        if bool(tmp.loc[top_idx, "winner"]) is True:
            correct += 1

    return correct / total if total else 0.0


@dataclass
class Config:
    nominees_csv: str
    reviews_csv: str
    split_year: int
    model: str
    smote_k: int
    random_state: int


def build_model_pipeline(cfg: Config):
    """
    Build an imblearn pipeline:
      StandardScaler -> SMOTE -> Classifier
    """
    smote = SMOTE(random_state=cfg.random_state, k_neighbors=cfg.smote_k)

    if cfg.model.lower() == "svm":
        clf = SVC(kernel="linear", probability=True, random_state=cfg.random_state)
    elif cfg.model.lower() in ("logreg", "logistic", "logistic_regression"):
        clf = LogisticRegression(
            max_iter=5000,
            solver="liblinear",
            random_state=cfg.random_state,
        )
    else:
        raise ValueError("Unsupported model. Choose 'svm' or 'logreg'.")

    pipe = ImbPipeline(steps=[
        ("scaler", StandardScaler()),
        ("smote", smote),
        ("clf", clf),
    ])
    return pipe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nominees_csv", default="moview_after_1970_beset_picture_nominees_with_winner.csv")
    parser.add_argument("--reviews_csv", default="critic_reviews_normalized_textprocessed_sentiment.csv")
    parser.add_argument("--split_year", type=int, default=2015, help="Train <= split_year, test > split_year")
    parser.add_argument("--model", choices=["svm", "logreg"], default="svm")
    parser.add_argument("--smote_k", type=int, default=3, help="k_neighbors for SMOTE (small minority => try 3)")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    cfg = Config(
        nominees_csv=args.nominees_csv,
        reviews_csv=args.reviews_csv,
        split_year=args.split_year,
        model=args.model,
        smote_k=args.smote_k,
        random_state=args.random_state,
    )

    # ----------------------------
    # Load
    # ----------------------------
    nominees = pd.read_csv(cfg.nominees_csv)
    reviews = pd.read_csv(cfg.reviews_csv)

    # Basic sanity
    if "movieId" not in nominees.columns or "movieId" not in reviews.columns:
        print("ERROR: Both CSVs must contain movieId.", file=sys.stderr)
        sys.exit(1)

    # ----------------------------
    # Aggregate reviews -> movie features
    # ----------------------------
    review_agg = aggregate_reviews(reviews)

    # ----------------------------
    # Merge with nominees
    # ----------------------------
    df = nominees.merge(review_agg, on="movieId", how="left")

    # If some movies have no reviews in the reviews file, fill with 0s where reasonable
    fill_zero_cols = [
        "review_count", "fresh_count", "rotten_count", "top_critic_count",
        "std_norm_score_10", "std_nltk_compound",
        "fresh_ratio", "rotten_ratio", "top_critic_ratio",
        "nltk_pos_count", "nltk_neu_count", "nltk_neg_count",
        "nltk_pos_ratio", "nltk_neu_ratio", "nltk_neg_ratio",
    ]
    for c in fill_zero_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    # Mean-like columns can be filled with 0 or overall mean; 0 is OK if scaled + model learns it.
    for c in ["avg_norm_score_10", "avg_nltk_compound"]:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    # ----------------------------
    # Parse runtime
    # ----------------------------
    if "runtime" in df.columns:
        df["runtime_min"] = df["runtime"].apply(parse_runtime_to_minutes)
    else:
        df["runtime_min"] = np.nan
    df["runtime_min"] = df["runtime_min"].fillna(df["runtime_min"].median())

    # ----------------------------
    # Target y
    # ----------------------------
    if "winner" not in df.columns:
        print("ERROR: nominees CSV must include 'winner' column.", file=sys.stderr)
        sys.exit(1)

    # Ensure winner is 0/1
    y = safe_bool_series(df["winner"])

    # ----------------------------
    # Feature selection & encoding
    # ----------------------------
    # Choose movie-level numeric features you already have + aggregated review features.
    numeric_cols = []
    for col in [
        "critic_score", "audience_score",
        "runtime_min",
        "review_count", "fresh_ratio", "rotten_ratio", "top_critic_ratio",
        "avg_norm_score_10", "std_norm_score_10",
        "avg_nltk_compound", "std_nltk_compound",
        "nltk_pos_ratio", "nltk_neu_ratio", "nltk_neg_ratio",
    ]:
        if col in df.columns:
            numeric_cols.append(col)

    # Categorical columns -> one-hot
    cat_cols = []
    for col in ["rating", "original_language", "critic_sentiment", "audience_sentiment"]:
        if col in df.columns:
            cat_cols.append(col)

    # Keep year for splitting & per-year evaluation
    if "movieYear" not in df.columns:
        print("ERROR: nominees CSV must include 'movieYear' for time-based split.", file=sys.stderr)
        sys.exit(1)

    # One-hot encode categoricals
    X_num = df[numeric_cols].copy()
    X_cat = pd.get_dummies(df[cat_cols].fillna("UNK"), drop_first=False) if cat_cols else pd.DataFrame(index=df.index)

    X = pd.concat([X_num, X_cat], axis=1).fillna(0.0)

    # ----------------------------
    # Time-based split
    # ----------------------------
    train_mask = df["movieYear"] <= cfg.split_year
    test_mask = df["movieYear"] > cfg.split_year

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        print(
            f"ERROR: Split produced empty train or test. "
            f"Check your years and split_year={cfg.split_year}.",
            file=sys.stderr,
        )
        sys.exit(1)

    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]
    df_test = df.loc[test_mask].copy()

    # ----------------------------
    # Train with SMOTE pipeline
    # ----------------------------
    pipe = build_model_pipeline(cfg)
    pipe.fit(X_train, y_train)

    # Predict probabilities for class 1 (winner)
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X_test)[:, 1]
    else:
        # Fallback (shouldn't happen for SVC(probability=True) or LogisticRegression)
        scores = pipe.decision_function(X_test)
        proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    y_pred = (proba >= 0.5).astype(int)

    # ----------------------------
    # Evaluation
    # ----------------------------
    print("\n=== Split ===")
    print(f"Train years: <= {cfg.split_year}  (n={len(X_train)})")
    print(f"Test years : >  {cfg.split_year}  (n={len(X_test)})")
    print(f"Train positives (winners): {int(y_train.sum())} / {len(y_train)}")
    print(f"Test positives  (winners): {int(y_test.sum())} / {len(y_test)}")

    print("\n=== Confusion Matrix (threshold=0.5) ===")
    print(confusion_matrix(y_test, y_pred))

    print("\n=== Classification Report (threshold=0.5) ===")
    print(classification_report(y_test, y_pred, digits=4))

    # ROC-AUC needs both classes present in y_test
    if len(np.unique(y_test)) == 2:
        auc = roc_auc_score(y_test, proba)
        print(f"ROC-AUC: {auc:.4f}")
    else:
        print("ROC-AUC: cannot compute (only one class present in test set).")

    per_year_acc = per_year_top_pick_accuracy(df_test, proba, year_col="movieYear")
    print(f"\n=== Per-year top-pick accuracy ===")
    print(f"Accuracy: {per_year_acc:.4f}  (choose max proba nominee each year)")

    # Show top prediction per year (useful for your report)
    tmp = df_test[["movieYear", "movieTitle", "winner"]].copy()
    tmp["proba"] = proba
    tmp = tmp.sort_values(["movieYear", "proba"], ascending=[True, False])

    top_each_year = tmp.groupby("movieYear").head(1)
    print("\n=== Top predicted nominee per year (test) ===")
    for _, row in top_each_year.iterrows():
        print(
            f"{int(row['movieYear'])}: {row['movieTitle']} | proba={row['proba']:.3f} | winner={bool(row['winner'])}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()