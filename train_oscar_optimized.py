#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def parse_runtime_to_minutes(runtime_val):
    if pd.isna(runtime_val):
        return np.nan
    s = str(runtime_val).strip().lower()

    h_match = re.search(r"(\d+)\s*h", s)
    m_match = re.search(r"(\d+)\s*m", s)

    hours = int(h_match.group(1)) if h_match else 0
    mins = int(m_match.group(1)) if m_match else 0

    if not h_match and not m_match:
        num_match = re.fullmatch(r"\d+", s)
        if num_match:
            return float(num_match.group(0))
        return np.nan

    return float(hours * 60 + mins)


def safe_bool_series(x: pd.Series) -> pd.Series:
    if x.dtype == bool:
        return x.astype(int)
    return (
        x.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": 1, "false": 0,
            "1": 1, "0": 0,
            "yes": 1, "no": 0
        })
        .fillna(0)
        .astype(int)
    )


def parse_date_features(df: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out[col], errors="coerce")
    out[f"{prefix}_month"] = dt.dt.month
    out[f"{prefix}_dayofyear"] = dt.dt.dayofyear
    out[f"{prefix}_quarter"] = dt.dt.quarter
    return out


def aggregate_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    df = reviews.copy()

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
        lbl = df["nltk_sentiment_label"].astype(str).str.lower()
        df["nltk_pos_i"] = (lbl == "positive").astype(int)
        df["nltk_neu_i"] = (lbl == "neutral").astype(int)
        df["nltk_neg_i"] = (lbl == "negative").astype(int)
    else:
        df["nltk_pos_i"] = 0
        df["nltk_neu_i"] = 0
        df["nltk_neg_i"] = 0

    # top critic weighted score
    df["topcritic_score_10"] = np.where(df["isTopCritic_i"] == 1, df["normalized_score_10"], np.nan)
    df["topcritic_compound"] = np.where(df["isTopCritic_i"] == 1, df["nltk_sentiment_compound"], np.nan)

    agg = df.groupby("movieId").agg(
        review_count=("movieId", "size"),
        fresh_count=("isFresh_i", "sum"),
        rotten_count=("isRotten_i", "sum"),
        top_critic_count=("isTopCritic_i", "sum"),

        avg_norm_score_10=("normalized_score_10", "mean"),
        std_norm_score_10=("normalized_score_10", "std"),
        min_norm_score_10=("normalized_score_10", "min"),
        max_norm_score_10=("normalized_score_10", "max"),

        avg_nltk_compound=("nltk_sentiment_compound", "mean"),
        std_nltk_compound=("nltk_sentiment_compound", "std"),

        topcritic_avg_score_10=("topcritic_score_10", "mean"),
        topcritic_avg_compound=("topcritic_compound", "mean"),

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

    agg["sentiment_balance"] = agg["nltk_pos_ratio"] - agg["nltk_neg_ratio"]
    agg["score_range_10"] = agg["max_norm_score_10"] - agg["min_norm_score_10"]

    for col in [
        "std_norm_score_10", "std_nltk_compound",
        "topcritic_avg_score_10", "topcritic_avg_compound",
        "score_range_10"
    ]:
        agg[col] = agg[col].fillna(0.0)

    return agg


def add_year_relative_features(df: pd.DataFrame, numeric_cols: list[str], year_col="movieYear") -> pd.DataFrame:
    out = df.copy()

    for col in numeric_cols:
        if col not in out.columns:
            continue

        grp = out.groupby(year_col)[col]
        out[f"{col}_year_mean"] = grp.transform("mean")
        out[f"{col}_year_std"] = grp.transform("std").replace(0, np.nan)
        out[f"{col}_year_rank_desc"] = grp.rank(method="average", ascending=False)
        out[f"{col}_minus_year_mean"] = out[col] - out[f"{col}_year_mean"]
        out[f"{col}_zscore_within_year"] = (
            (out[col] - out[f"{col}_year_mean"]) / out[f"{col}_year_std"]
        )

    return out


def per_year_top_pick_accuracy(test_df: pd.DataFrame, proba: np.ndarray, year_col: str = "movieYear") -> float:
    tmp = test_df[[year_col, "winner"]].copy()
    tmp["proba"] = proba

    correct = 0
    total = 0

    for year, g in tmp.groupby(year_col):
        total += 1
        top_idx = g["proba"].idxmax()
        if int(tmp.loc[top_idx, "winner"]) == 1:
            correct += 1

    return correct / total if total else 0.0


def print_top_predictions(df_test: pd.DataFrame, proba: np.ndarray):
    tmp = df_test[["movieYear", "movieTitle", "winner"]].copy()
    tmp["proba"] = proba
    tmp = tmp.sort_values(["movieYear", "proba"], ascending=[True, False])

    top_each_year = tmp.groupby("movieYear").head(1)
    print("\n=== Top predicted nominee per year ===")
    for _, row in top_each_year.iterrows():
        print(
            f"{int(row['movieYear'])}: {row['movieTitle']} | "
            f"proba={row['proba']:.3f} | winner={bool(row['winner'])}"
        )


def build_preprocessor(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in numeric_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                cat_cols,
            ),
        ]
    )
    return preprocessor


def build_model(model_name: str, C: float):
    if model_name == "logreg":
        return LogisticRegression(
            C=C,
            max_iter=5000,
            class_weight="balanced",
            solver="liblinear",
            random_state=42,
        )
    elif model_name == "svm":
        return SVC(
            C=C,
            kernel="linear",
            probability=True,
            class_weight="balanced",
            random_state=42,
        )
    else:
        raise ValueError("Unsupported model")


# -------------------------------------------------
# Main
# -------------------------------------------------

@dataclass
class Config:
    nominees_csv: str
    reviews_csv: str
    train_end_year: int
    val_end_year: int
    test_start_year: int


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nominees_csv", default="moview_after_1970_beset_picture_nominees_with_winner.csv")
    parser.add_argument("--reviews_csv", default="critic_reviews_normalized_textprocessed_sentiment.csv")
    parser.add_argument("--train_end_year", type=int, default=2012)
    parser.add_argument("--val_end_year", type=int, default=2015)
    parser.add_argument("--test_start_year", type=int, default=2016)
    args = parser.parse_args()

    cfg = Config(
        nominees_csv=args.nominees_csv,
        reviews_csv=args.reviews_csv,
        train_end_year=args.train_end_year,
        val_end_year=args.val_end_year,
        test_start_year=args.test_start_year,
    )

    nominees = pd.read_csv(cfg.nominees_csv)
    reviews = pd.read_csv(cfg.reviews_csv)

    if "movieId" not in nominees.columns or "movieId" not in reviews.columns:
        print("ERROR: Both CSVs must contain movieId.", file=sys.stderr)
        sys.exit(1)

    review_agg = aggregate_reviews(reviews)
    df = nominees.merge(review_agg, on="movieId", how="left")

    # Runtime
    df["runtime_min"] = df["runtime"].apply(parse_runtime_to_minutes) if "runtime" in df.columns else np.nan

    # Dates
    if "release_date_theaters" in df.columns:
        df = parse_date_features(df, "release_date_theaters", "theaters")
    if "release_date_streaming" in df.columns:
        df = parse_date_features(df, "release_date_streaming", "streaming")

    # Gap between theater and streaming release
    if "release_date_theaters" in df.columns and "release_date_streaming" in df.columns:
        theaters_dt = pd.to_datetime(df["release_date_theaters"], errors="coerce")
        streaming_dt = pd.to_datetime(df["release_date_streaming"], errors="coerce")
        df["days_theaters_to_streaming"] = (streaming_dt - theaters_dt).dt.days

    # Fill obvious numeric review missings
    review_fill_zero = [
        "review_count", "fresh_count", "rotten_count", "top_critic_count",
        "fresh_ratio", "rotten_ratio", "top_critic_ratio",
        "nltk_pos_count", "nltk_neu_count", "nltk_neg_count",
        "nltk_pos_ratio", "nltk_neu_ratio", "nltk_neg_ratio",
        "sentiment_balance", "score_range_10",
        "std_norm_score_10", "std_nltk_compound"
    ]
    for c in review_fill_zero:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    y = safe_bool_series(df["winner"])

    # Base numeric features
    base_numeric_cols = [
        "critic_score",
        "audience_score",
        "runtime_min",
        "review_count",
        "fresh_ratio",
        "rotten_ratio",
        "top_critic_ratio",
        "avg_norm_score_10",
        "std_norm_score_10",
        "avg_nltk_compound",
        "std_nltk_compound",
        "topcritic_avg_score_10",
        "topcritic_avg_compound",
        "nltk_pos_ratio",
        "nltk_neu_ratio",
        "nltk_neg_ratio",
        "sentiment_balance",
        "score_range_10",
        "theaters_month",
        "theaters_dayofyear",
        "theaters_quarter",
        "streaming_month",
        "streaming_dayofyear",
        "streaming_quarter",
        "days_theaters_to_streaming",
    ]
    base_numeric_cols = [c for c in base_numeric_cols if c in df.columns]

    # Add within-year relative features
    df = add_year_relative_features(df, base_numeric_cols, year_col="movieYear")

    cat_cols = [
        c for c in [
            "rating",
            "original_language",
            "critic_sentiment",
            "audience_sentiment",
        ] if c in df.columns
    ]

    # Final feature set
    feature_cols = []
    for c in df.columns:
        if c in {"movieId", "movieURL", "movieTitle", "winner", "movieYear",
                 "release_date_theaters", "release_date_streaming"}:
            continue
        if c in cat_cols or pd.api.types.is_numeric_dtype(df[c]):
            feature_cols.append(c)

    X = df[feature_cols].copy()

    # Time-based splits
    train_mask = df["movieYear"] <= cfg.train_end_year
    val_mask = (df["movieYear"] > cfg.train_end_year) & (df["movieYear"] <= cfg.val_end_year)
    test_mask = df["movieYear"] >= cfg.test_start_year

    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_val, y_val = X.loc[val_mask], y.loc[val_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]

    df_val = df.loc[val_mask].copy()
    df_test = df.loc[test_mask].copy()

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        print("ERROR: Empty train/val/test split. Adjust the year boundaries.", file=sys.stderr)
        sys.exit(1)

    candidates = []
    for model_name in ["logreg", "svm"]:
        for C in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            preprocessor = build_preprocessor(X_train)
            model = build_model(model_name, C)

            pipe = Pipeline([
                ("prep", preprocessor),
                ("clf", model),
            ])

            pipe.fit(X_train, y_train)
            val_proba = pipe.predict_proba(X_val)[:, 1]
            val_acc = per_year_top_pick_accuracy(df_val, val_proba)

            if len(np.unique(y_val)) == 2:
                val_auc = roc_auc_score(y_val, val_proba)
            else:
                val_auc = np.nan

            candidates.append({
                "model_name": model_name,
                "C": C,
                "val_top_pick_acc": val_acc,
                "val_auc": val_auc,
                "pipe": pipe,
            })

    # prioritize yearly winner-pick accuracy, then AUC
    candidates = sorted(
        candidates,
        key=lambda d: (
            d["val_top_pick_acc"],
            -999 if np.isnan(d["val_auc"]) else d["val_auc"]
        ),
        reverse=True,
    )

    best = candidates[0]
    best_pipe = best["pipe"]

    # retrain on train + val, evaluate on test
    trainval_mask = df["movieYear"] < cfg.test_start_year
    X_trainval, y_trainval = X.loc[trainval_mask], y.loc[trainval_mask]

    final_preprocessor = build_preprocessor(X_trainval)
    final_model = build_model(best["model_name"], best["C"])
    final_pipe = Pipeline([
        ("prep", final_preprocessor),
        ("clf", final_model),
    ])
    final_pipe.fit(X_trainval, y_trainval)

    proba = final_pipe.predict_proba(X_test)[:, 1]

    # optional threshold search on validation
    best_thr = 0.5
    best_f1 = -1
    val_proba_final = final_pipe.predict_proba(X_val)[:, 1]
    for thr in np.arange(0.2, 0.81, 0.05):
        pred = (val_proba_final >= thr).astype(int)
        tp = ((pred == 1) & (y_val.values == 1)).sum()
        fp = ((pred == 1) & (y_val.values == 0)).sum()
        fn = ((pred == 0) & (y_val.values == 1)).sum()

        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)

        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)

    y_pred = (proba >= best_thr).astype(int)

    print("\n=== Model Selection ===")
    print(f"Best model: {best['model_name']} | C={best['C']}")
    print(f"Validation per-year top-pick accuracy: {best['val_top_pick_acc']:.4f}")
    print(f"Validation ROC-AUC: {best['val_auc']:.4f}" if not np.isnan(best["val_auc"]) else "Validation ROC-AUC: N/A")
    print(f"Chosen threshold from validation: {best_thr:.2f}")

    print("\n=== Split ===")
    print(f"Train years: <= {cfg.train_end_year}")
    print(f"Val years  : {cfg.train_end_year + 1} to {cfg.val_end_year}")
    print(f"Test years : >= {cfg.test_start_year}")
    print(f"Train+Val n={len(X_trainval)} | Test n={len(X_test)}")
    print(f"Train+Val positives: {int(y_trainval.sum())} / {len(y_trainval)}")
    print(f"Test positives     : {int(y_test.sum())} / {len(y_test)}")

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4))

    if len(np.unique(y_test)) == 2:
        auc = roc_auc_score(y_test, proba)
        print(f"ROC-AUC: {auc:.4f}")

    per_year_acc = per_year_top_pick_accuracy(df_test, proba)
    print("\n=== Per-year top-pick accuracy ===")
    print(f"Accuracy: {per_year_acc:.4f}")

    print_top_predictions(df_test, proba)

    print("\nDone.")


if __name__ == "__main__":
    main()