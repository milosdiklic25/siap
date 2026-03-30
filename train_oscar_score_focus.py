#!/usr/bin/env python3
"""
Train an Oscar Best Picture winner classifier using SMOTE, with stronger focus on
critic_score and audience_score.

Inputs (CSV):
1) movies_after_1970_best_picture_nominees_with_winner.csv
   expected columns (at least):
   movieId,movieTitle,movieYear,critic_score,critic_sentiment,
   audience_score,audience_sentiment,winner

2) critic_reviews_normalized_textprocessed_sentiment.csv
   expected columns (at least):
   movieId,reviewState,isFresh,isRotten,isTopCritic,normalized_score_10,
   nltk_sentiment_label,nltk_sentiment_compound

What this version does:
- Aggregates review-level info into movie-level features
- Merges with nominees dataset
- Creates critic/audience score-derived features
- Uses a smaller, score-centered feature set
- Time-based split by year (train <= split_year, test > split_year)
- Applies StandardScaler + SMOTE (train only) + SVM or Logistic Regression
- Evaluates with classification metrics + per-year winner-pick accuracy
- Prints top learned feature weights for interpretability
"""

from __future__ import annotations

import argparse
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

def safe_bool_series(x: pd.Series) -> pd.Series:
    """
    Convert True/False-like values to 0/1 ints.
    """
    if x.dtype == bool:
        return x.astype(int)
    return (
        x.astype(str)
        .str.lower()
        .map({"true": 1, "false": 0, "1": 1, "0": 0})
        .fillna(0)
        .astype(int)
    )


def safe_numeric_series(x: pd.Series) -> pd.Series:
    """
    Convert a series to numeric, coercing invalid values to NaN.
    """
    return pd.to_numeric(x, errors="coerce")


def aggregate_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate review-level dataset into movie-level features, keeping critic-oriented
    information that can complement critic_score and audience_score.
    """
    df = reviews.copy()

    # Safe defaults if columns are missing
    for col in ["isFresh", "isRotten", "isTopCritic"]:
        if col not in df.columns:
            df[col] = False

    for col in ["normalized_score_10", "nltk_sentiment_compound"]:
        if col not in df.columns:
            df[col] = np.nan

    df["isFresh_i"] = safe_bool_series(df["isFresh"])
    df["isRotten_i"] = safe_bool_series(df["isRotten"])
    df["isTopCritic_i"] = safe_bool_series(df["isTopCritic"])

    if "nltk_sentiment_label" in df.columns:
        sentiment_label = df["nltk_sentiment_label"].astype(str).str.lower()
        df["nltk_pos_i"] = (sentiment_label == "positive").astype(int)
        df["nltk_neu_i"] = (sentiment_label == "neutral").astype(int)
        df["nltk_neg_i"] = (sentiment_label == "negative").astype(int)
    else:
        df["nltk_pos_i"] = 0
        df["nltk_neu_i"] = 0
        df["nltk_neg_i"] = 0

    df["normalized_score_10"] = safe_numeric_series(df["normalized_score_10"])
    df["nltk_sentiment_compound"] = safe_numeric_series(df["nltk_sentiment_compound"])

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

    eps = 1e-9
    agg["fresh_ratio"] = agg["fresh_count"] / (agg["review_count"] + eps)
    agg["rotten_ratio"] = agg["rotten_count"] / (agg["review_count"] + eps)
    agg["top_critic_ratio"] = agg["top_critic_count"] / (agg["review_count"] + eps)
    agg["nltk_pos_ratio"] = agg["nltk_pos_count"] / (agg["review_count"] + eps)
    agg["nltk_neu_ratio"] = agg["nltk_neu_count"] / (agg["review_count"] + eps)
    agg["nltk_neg_ratio"] = agg["nltk_neg_count"] / (agg["review_count"] + eps)

    for col in ["std_norm_score_10", "std_nltk_compound"]:
        agg[col] = agg[col].fillna(0.0)

    return agg


def per_year_top_pick_accuracy(
    test_df: pd.DataFrame,
    proba: np.ndarray,
    year_col: str = "movieYear",
) -> float:
    """
    For each year, pick the movie with the highest predicted probability of winner
    and check if it is actually the winner.
    """
    tmp = test_df[[year_col, "winner"]].copy()
    tmp["proba"] = proba

    correct = 0
    total = 0

    for _, g in tmp.groupby(year_col):
        total += 1
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
    positive_threshold: float


def build_model_pipeline(cfg: Config):
    """
    Build an imblearn pipeline:
      StandardScaler -> SMOTE -> Classifier
    """
    smote = SMOTE(random_state=cfg.random_state, k_neighbors=cfg.smote_k)

    if cfg.model.lower() == "svm":
        clf = SVC(
            kernel="linear",
            probability=True,
            random_state=cfg.random_state,
        )
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


def print_feature_weights(pipe, feature_names: list[str], top_n: int = 20) -> None:
    """
    Print the largest absolute feature weights for linear models.
    """
    clf = pipe.named_steps["clf"]

    if hasattr(clf, "coef_"):
        coef = clf.coef_[0]
        coef_df = pd.DataFrame({
            "feature": feature_names,
            "weight": coef,
            "abs_weight": np.abs(coef),
        }).sort_values("abs_weight", ascending=False)

        print("\n=== Top feature weights ===")
        print(coef_df[["feature", "weight"]].head(top_n).to_string(index=False))
    else:
        print("\n=== Top feature weights ===")
        print("Model does not expose linear coefficients.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nominees_csv",
        default="movies_after_1970_best_picture_nominees_with_winner.csv",
    )
    parser.add_argument(
        "--reviews_csv",
        default="critic_reviews_normalized_textprocessed_sentiment.csv",
    )
    parser.add_argument(
        "--split_year",
        type=int,
        default=2015,
        help="Train <= split_year, test > split_year",
    )
    parser.add_argument(
        "--model",
        choices=["svm", "logreg"],
        default="svm",
    )
    parser.add_argument(
        "--smote_k",
        type=int,
        default=3,
        help="k_neighbors for SMOTE",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--positive_threshold",
        type=float,
        default=0.5,
        help="Probability threshold for classifying a movie as winner",
    )
    args = parser.parse_args()

    cfg = Config(
        nominees_csv=args.nominees_csv,
        reviews_csv=args.reviews_csv,
        split_year=args.split_year,
        model=args.model,
        smote_k=args.smote_k,
        random_state=args.random_state,
        positive_threshold=args.positive_threshold,
    )

    # ----------------------------
    # Load
    # ----------------------------
    nominees = pd.read_csv(cfg.nominees_csv)
    reviews = pd.read_csv(cfg.reviews_csv)

    if "movieId" not in nominees.columns or "movieId" not in reviews.columns:
        print("ERROR: Both CSVs must contain movieId.", file=sys.stderr)
        sys.exit(1)

    # ----------------------------
    # Aggregate review features
    # ----------------------------
    review_agg = aggregate_reviews(reviews)

    # ----------------------------
    # Merge
    # ----------------------------
    df = nominees.merge(review_agg, on="movieId", how="left")

    # ----------------------------
    # Fill review aggregation gaps
    # ----------------------------
    fill_zero_cols = [
        "review_count",
        "fresh_count",
        "rotten_count",
        "top_critic_count",
        "std_norm_score_10",
        "std_nltk_compound",
        "fresh_ratio",
        "rotten_ratio",
        "top_critic_ratio",
        "nltk_pos_count",
        "nltk_neu_count",
        "nltk_neg_count",
        "nltk_pos_ratio",
        "nltk_neu_ratio",
        "nltk_neg_ratio",
    ]
    for col in fill_zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    for col in ["avg_norm_score_10", "avg_nltk_compound"]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # ----------------------------
    # Basic required columns
    # ----------------------------
    if "winner" not in df.columns:
        print("ERROR: nominees CSV must include 'winner' column.", file=sys.stderr)
        sys.exit(1)

    if "movieYear" not in df.columns:
        print("ERROR: nominees CSV must include 'movieYear' column.", file=sys.stderr)
        sys.exit(1)

    if "movieTitle" not in df.columns:
        df["movieTitle"] = df["movieId"].astype(str)

    # ----------------------------
    # Strong focus on critic/audience scores
    # ----------------------------
    if "critic_score" not in df.columns:
        df["critic_score"] = np.nan
    if "audience_score" not in df.columns:
        df["audience_score"] = np.nan

    df["critic_score"] = safe_numeric_series(df["critic_score"])
    df["audience_score"] = safe_numeric_series(df["audience_score"])

    critic_median = df["critic_score"].median()
    audience_median = df["audience_score"].median()

    if pd.isna(critic_median):
        critic_median = 0.0
    if pd.isna(audience_median):
        audience_median = 0.0

    df["critic_score"] = df["critic_score"].fillna(critic_median)
    df["audience_score"] = df["audience_score"].fillna(audience_median)

    # Derived score-focused features
    df["score_mean"] = (df["critic_score"] + df["audience_score"]) / 2.0
    df["score_gap"] = df["critic_score"] - df["audience_score"]
    df["score_abs_gap"] = (df["critic_score"] - df["audience_score"]).abs()
    df["score_product"] = df["critic_score"] * df["audience_score"]
    df["score_harmony"] = 100.0 - df["score_abs_gap"]

    # Optional score-shape features
    df["critic_gt_audience"] = (df["critic_score"] > df["audience_score"]).astype(int)
    df["audience_gt_critic"] = (df["audience_score"] > df["critic_score"]).astype(int)

    # ----------------------------
    # Target
    # ----------------------------
    y = safe_bool_series(df["winner"])

    # ----------------------------
    # Score-centered feature set
    # ----------------------------
    numeric_cols = [
        "critic_score",
        "audience_score",
        "score_mean",
        "score_gap",
        "score_abs_gap",
        "score_product",
        "score_harmony",
        "critic_gt_audience",
        "audience_gt_critic",
    ]

    # Keep only critic-review aggregates that directly support reception quality
    for col in [
        "review_count",
        "fresh_ratio",
        "top_critic_ratio",
        "avg_norm_score_10",
        "avg_nltk_compound",
    ]:
        if col in df.columns:
            numeric_cols.append(col)

    cat_cols = []
    for col in ["critic_sentiment", "audience_sentiment"]:
        if col in df.columns:
            cat_cols.append(col)

    # De-duplicate while preserving order
    numeric_cols = list(dict.fromkeys([c for c in numeric_cols if c in df.columns]))
    cat_cols = list(dict.fromkeys(cat_cols))

    X_num = df[numeric_cols].copy()
    X_cat = (
        pd.get_dummies(df[cat_cols].fillna("UNK"), drop_first=False)
        if cat_cols else
        pd.DataFrame(index=df.index)
    )

    X = pd.concat([X_num, X_cat], axis=1).fillna(0.0)
    feature_names = list(X.columns)

    # ----------------------------
    # Time-based split
    # ----------------------------
    train_mask = df["movieYear"] <= cfg.split_year
    test_mask = df["movieYear"] > cfg.split_year

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        print(
            f"ERROR: Split produced empty train or test. Check split_year={cfg.split_year}.",
            file=sys.stderr,
        )
        sys.exit(1)

    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]
    df_test = df.loc[test_mask].copy()

    # ----------------------------
    # Train
    # ----------------------------
    pipe = build_model_pipeline(cfg)
    pipe.fit(X_train, y_train)

    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X_test)[:, 1]
    else:
        scores = pipe.decision_function(X_test)
        proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    y_pred = (proba >= cfg.positive_threshold).astype(int)

    # ----------------------------
    # Evaluation
    # ----------------------------
    print("\n=== Split ===")
    print(f"Train years: <= {cfg.split_year}  (n={len(X_train)})")
    print(f"Test years : >  {cfg.split_year}  (n={len(X_test)})")
    print(f"Train positives (winners): {int(y_train.sum())} / {len(y_train)}")
    print(f"Test positives  (winners): {int(y_test.sum())} / {len(y_test)}")

    print(f"\n=== Confusion Matrix (threshold={cfg.positive_threshold:.2f}) ===")
    print(confusion_matrix(y_test, y_pred))

    print(f"\n=== Classification Report (threshold={cfg.positive_threshold:.2f}) ===")
    print(classification_report(y_test, y_pred, digits=4))

    if len(np.unique(y_test)) == 2:
        auc = roc_auc_score(y_test, proba)
        print(f"ROC-AUC: {auc:.4f}")
    else:
        print("ROC-AUC: cannot compute (only one class present in test set).")

    per_year_acc = per_year_top_pick_accuracy(df_test, proba, year_col="movieYear")
    print("\n=== Per-year top-pick accuracy ===")
    print(f"Accuracy: {per_year_acc:.4f}  (choose max-proba nominee each year)")

    # ----------------------------
    # Predictions by year
    # ----------------------------
    tmp = df_test[["movieYear", "movieTitle", "winner", "critic_score", "audience_score"]].copy()
    tmp["proba"] = proba
    tmp = tmp.sort_values(["movieYear", "proba"], ascending=[True, False])

    print("\n=== Top predicted nominee per year (test) ===")
    top_each_year = tmp.groupby("movieYear").head(1)
    for _, row in top_each_year.iterrows():
        print(
            f"{int(row['movieYear'])}: {row['movieTitle']} | "
            f"proba={row['proba']:.3f} | "
            f"critic_score={row['critic_score']:.1f} | "
            f"audience_score={row['audience_score']:.1f} | "
            f"winner={bool(row['winner'])}"
        )

    # ----------------------------
    # Inspect model emphasis
    # ----------------------------
    print_feature_weights(pipe, feature_names, top_n=20)

    print("\n=== Features used ===")
    print(feature_names)

    print("\nDone.")


if __name__ == "__main__":
    main()