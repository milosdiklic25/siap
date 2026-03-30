#!/usr/bin/env python3
from __future__ import annotations

import warnings
from pandas.errors import PerformanceWarning

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=PerformanceWarning)

import argparse
import re
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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


def try_parse_date_series(s: pd.Series) -> pd.Series:
    # safer than strict single-format if your column has mixed formats
    return pd.to_datetime(s, errors="coerce")


def add_date_parts(df: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    out = df.copy()
    dt = try_parse_date_series(out[col])
    out[f"{prefix}_month"] = dt.dt.month
    out[f"{prefix}_quarter"] = dt.dt.quarter
    out[f"{prefix}_dayofyear"] = dt.dt.dayofyear
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

    fill_zero_cols = [
        "std_norm_score_10",
        "std_nltk_compound",
        "topcritic_avg_score_10",
        "topcritic_avg_compound",
        "score_range_10",
    ]
    for c in fill_zero_cols:
        agg[c] = agg[c].fillna(0.0)

    return agg


def add_year_relative_features(df: pd.DataFrame, base_numeric_cols: list[str], year_col: str = "movieYear") -> pd.DataFrame:
    out = df.copy()
    extra = {}

    for col in base_numeric_cols:
        if col not in out.columns:
            continue

        grp = out.groupby(year_col)[col]
        year_mean = grp.transform("mean")
        year_std = grp.transform("std").replace(0, np.nan)
        rank_desc = grp.rank(method="average", ascending=False)

        extra[f"{col}_year_mean"] = year_mean
        extra[f"{col}_year_std"] = year_std
        extra[f"{col}_year_rank_desc"] = rank_desc
        extra[f"{col}_minus_year_mean"] = out[col] - year_mean
        extra[f"{col}_zscore_within_year"] = (out[col] - year_mean) / year_std

    if extra:
        out = pd.concat([out, pd.DataFrame(extra, index=out.index)], axis=1)

    return out


def build_movie_level_dataset(nominees: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    review_agg = aggregate_reviews(reviews)
    df = nominees.merge(review_agg, on="movieId", how="left")

    # runtime
    if "runtime" in df.columns:
        df["runtime_min"] = df["runtime"].apply(parse_runtime_to_minutes)
    else:
        df["runtime_min"] = np.nan

    # dates
    if "release_date_theaters" in df.columns:
        df = add_date_parts(df, "release_date_theaters", "theaters")
    if "release_date_streaming" in df.columns:
        df = add_date_parts(df, "release_date_streaming", "streaming")

    if "release_date_theaters" in df.columns and "release_date_streaming" in df.columns:
        theaters_dt = try_parse_date_series(df["release_date_theaters"])
        streaming_dt = try_parse_date_series(df["release_date_streaming"])
        df["days_theaters_to_streaming"] = (streaming_dt - theaters_dt).dt.days

    # derived simple features
    if "critic_score" in df.columns and "audience_score" in df.columns:
        df["critic_audience_gap"] = df["critic_score"] - df["audience_score"]

    if "fresh_ratio" in df.columns and "review_count" in df.columns:
        df["fresh_reviews_volume"] = df["fresh_ratio"] * df["review_count"]

    if "theaters_month" in df.columns:
        df["is_late_year_release"] = df["theaters_month"].isin([10, 11, 12]).astype(int)

    # fill some review missings
    fill_zero_cols = [
        "review_count", "fresh_count", "rotten_count", "top_critic_count",
        "fresh_ratio", "rotten_ratio", "top_critic_ratio",
        "nltk_pos_count", "nltk_neu_count", "nltk_neg_count",
        "nltk_pos_ratio", "nltk_neu_ratio", "nltk_neg_ratio",
        "sentiment_balance", "score_range_10",
        "std_norm_score_10", "std_nltk_compound",
        "fresh_reviews_volume",
    ]
    for c in fill_zero_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    base_numeric_cols = [
        "critic_score",
        "audience_score",
        "critic_audience_gap",
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
        "fresh_reviews_volume",
        "theaters_month",
        "theaters_quarter",
        "theaters_dayofyear",
        "streaming_month",
        "streaming_quarter",
        "streaming_dayofyear",
        "days_theaters_to_streaming",
        "is_late_year_release",
    ]
    base_numeric_cols = [c for c in base_numeric_cols if c in df.columns]

    df = add_year_relative_features(df, base_numeric_cols, year_col="movieYear")
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {
        "movieId", "movieURL", "movieTitle", "winner", "movieYear",
        "release_date_theaters", "release_date_streaming"
    }

    feature_cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or c in [
            "rating", "original_language", "critic_sentiment", "audience_sentiment"
        ]:
            feature_cols.append(c)
    return feature_cols


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in numeric_cols]

    return ColumnTransformer(
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


# -------------------------------------------------
# Pairwise ranking
# -------------------------------------------------

def build_pairwise_dataset(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Create pairwise training rows within each year.

    If movie A is the winner and movie B is not, create:
      x = features(A) - features(B), y = 1
      x = features(B) - features(A), y = 0

    This teaches the model which nominee should rank above another.
    """
    pair_rows = []
    pair_y = []

    for year, g in df.groupby("movieYear"):
        g = g.reset_index(drop=True)
        if len(g) < 2:
            continue

        winners = g[g["winner"] == 1]
        losers = g[g["winner"] == 0]

        if len(winners) != 1 or len(losers) == 0:
            continue

        winner_row = winners.iloc[0]

        for _, loser_row in losers.iterrows():
            row_pos = {}
            row_neg = {}

            for col in feature_cols:
                a = winner_row[col]
                b = loser_row[col]

                # numeric difference
                if pd.api.types.is_numeric_dtype(df[col]):
                    a = np.nan if pd.isna(a) else a
                    b = np.nan if pd.isna(b) else b
                    row_pos[col] = a - b if pd.notna(a) and pd.notna(b) else np.nan
                    row_neg[col] = b - a if pd.notna(a) and pd.notna(b) else np.nan
                else:
                    # categorical: store both sides for comparison
                    row_pos[f"{col}_left"] = a
                    row_pos[f"{col}_right"] = b
                    row_neg[f"{col}_left"] = b
                    row_neg[f"{col}_right"] = a

            pair_rows.append(row_pos)
            pair_y.append(1)

            pair_rows.append(row_neg)
            pair_y.append(0)

    X_pair = pd.DataFrame(pair_rows)
    y_pair = np.array(pair_y, dtype=int)
    return X_pair, y_pair


def fit_pairwise_ranker(train_df: pd.DataFrame, feature_cols: list[str], C: float) -> Pipeline:
    X_pair, y_pair = build_pairwise_dataset(train_df, feature_cols)
    preprocessor = build_preprocessor(X_pair)

    clf = LogisticRegression(
        C=C,
        max_iter=5000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )

    pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", clf),
    ])
    pipe.fit(X_pair, y_pair)
    return pipe


def pairwise_prob(ranker: Pipeline, left_row: pd.Series, right_row: pd.Series, train_df: pd.DataFrame, feature_cols: list[str]) -> float:
    """
    Probability that left_row should rank above right_row.
    """
    data = {}
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(train_df[col]):
            a = left_row[col]
            b = right_row[col]
            data[col] = a - b if pd.notna(a) and pd.notna(b) else np.nan
        else:
            data[f"{col}_left"] = left_row[col]
            data[f"{col}_right"] = right_row[col]

    X_one = pd.DataFrame([data])
    return float(ranker.predict_proba(X_one)[:, 1][0])


def score_year_nominees(rank_model: Pipeline, year_df: pd.DataFrame, train_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Score each nominee by pairwise wins against all others.
    """
    g = year_df.copy().reset_index(drop=True)
    n = len(g)

    scores = np.zeros(n, dtype=float)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p = pairwise_prob(rank_model, g.iloc[i], g.iloc[j], train_df, feature_cols)
            scores[i] += p

    g["rank_score"] = scores
    g = g.sort_values("rank_score", ascending=False).reset_index(drop=True)
    return g


def top_pick_accuracy(rank_model: Pipeline, eval_df: pd.DataFrame, train_df: pd.DataFrame, feature_cols: list[str]) -> float:
    correct = 0
    total = 0

    for year, g in eval_df.groupby("movieYear"):
        if len(g) < 2:
            continue
        ranked = score_year_nominees(rank_model, g, train_df, feature_cols)
        total += 1
        if int(ranked.iloc[0]["winner"]) == 1:
            correct += 1

    return correct / total if total else 0.0


def top_k_accuracy(rank_model: Pipeline, eval_df: pd.DataFrame, train_df: pd.DataFrame, feature_cols: list[str], k: int = 2) -> float:
    correct = 0
    total = 0

    for year, g in eval_df.groupby("movieYear"):
        if len(g) < 2:
            continue
        ranked = score_year_nominees(rank_model, g, train_df, feature_cols)
        total += 1
        if ranked.head(k)["winner"].astype(int).sum() > 0:
            correct += 1

    return correct / total if total else 0.0


def mean_winner_rank(rank_model: Pipeline, eval_df: pd.DataFrame, train_df: pd.DataFrame, feature_cols: list[str]) -> float:
    ranks = []

    for year, g in eval_df.groupby("movieYear"):
        if len(g) < 2:
            continue
        ranked = score_year_nominees(rank_model, g, train_df, feature_cols)
        winner_positions = ranked.index[ranked["winner"].astype(int) == 1].tolist()
        if winner_positions:
            ranks.append(winner_positions[0] + 1)  # 1-based

    return float(np.mean(ranks)) if ranks else np.nan


def print_ranked_predictions(rank_model: Pipeline, eval_df: pd.DataFrame, train_df: pd.DataFrame, feature_cols: list[str]):
    print("\n=== Ranked nominees by year ===")
    for year, g in eval_df.groupby("movieYear"):
        ranked = score_year_nominees(rank_model, g, train_df, feature_cols)
        top = ranked.iloc[0]
        print(f"{int(year)}: {top['movieTitle']} | rank_score={top['rank_score']:.3f} | winner={bool(top['winner'])}")


def approximate_auc_from_rank_scores(rank_model: Pipeline, eval_df: pd.DataFrame, train_df: pd.DataFrame, feature_cols: list[str]) -> float:
    """
    Build one score per row from yearly ranking scores, then compute AUC.
    This is optional and only approximate because the model is trained for ranking.
    """
    rows = []
    labels = []

    for year, g in eval_df.groupby("movieYear"):
        ranked = score_year_nominees(rank_model, g, train_df, feature_cols)
        for _, row in ranked.iterrows():
            rows.append(row["rank_score"])
            labels.append(int(row["winner"]))

    labels = np.array(labels)
    rows = np.array(rows)

    if len(np.unique(labels)) < 2:
        return np.nan
    return float(roc_auc_score(labels, rows))


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

    if "winner" not in nominees.columns or "movieYear" not in nominees.columns:
        print("ERROR: nominees CSV must contain winner and movieYear.", file=sys.stderr)
        sys.exit(1)

    nominees["winner"] = safe_bool_series(nominees["winner"])

    df = build_movie_level_dataset(nominees, reviews)
    feature_cols = get_feature_columns(df)

    train_mask = df["movieYear"] <= cfg.train_end_year
    val_mask = (df["movieYear"] > cfg.train_end_year) & (df["movieYear"] <= cfg.val_end_year)
    test_mask = df["movieYear"] >= cfg.test_start_year

    train_df = df.loc[train_mask].copy()
    val_df = df.loc[val_mask].copy()
    test_df = df.loc[test_mask].copy()

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        print("ERROR: Empty split. Adjust the year boundaries.", file=sys.stderr)
        sys.exit(1)

    # tune C using validation top-pick accuracy
    candidates = []
    for C in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        ranker = fit_pairwise_ranker(train_df, feature_cols, C=C)
        val_top1 = top_pick_accuracy(ranker, val_df, train_df, feature_cols)
        val_top2 = top_k_accuracy(ranker, val_df, train_df, feature_cols, k=2)
        val_mean_rank = mean_winner_rank(ranker, val_df, train_df, feature_cols)

        candidates.append({
            "C": C,
            "val_top1": val_top1,
            "val_top2": val_top2,
            "val_mean_rank": val_mean_rank,
        })

    candidates = sorted(
        candidates,
        key=lambda x: (x["val_top1"], x["val_top2"], -x["val_mean_rank"]),
        reverse=True,
    )

    best = candidates[0]

    # retrain on train+val
    trainval_df = df.loc[df["movieYear"] < cfg.test_start_year].copy()
    final_ranker = fit_pairwise_ranker(trainval_df, feature_cols, C=best["C"])

    test_top1 = top_pick_accuracy(final_ranker, test_df, trainval_df, feature_cols)
    test_top2 = top_k_accuracy(final_ranker, test_df, trainval_df, feature_cols, k=2)
    test_mean_rank = mean_winner_rank(final_ranker, test_df, trainval_df, feature_cols)
    test_auc = approximate_auc_from_rank_scores(final_ranker, test_df, trainval_df, feature_cols)

    print("\n=== Ranking Model Selection ===")
    print(f"Best C: {best['C']}")
    print(f"Validation Top-1 accuracy: {best['val_top1']:.4f}")
    print(f"Validation Top-2 accuracy: {best['val_top2']:.4f}")
    print(f"Validation mean winner rank: {best['val_mean_rank']:.4f}")

    print("\n=== Split ===")
    print(f"Train years: <= {cfg.train_end_year}")
    print(f"Val years  : {cfg.train_end_year + 1} to {cfg.val_end_year}")
    print(f"Test years : >= {cfg.test_start_year}")
    print(f"Train n={len(train_df)} | Val n={len(val_df)} | Test n={len(test_df)}")

    print("\n=== Ranking Evaluation on Test ===")
    print(f"Top-1 accuracy: {test_top1:.4f}")
    print(f"Top-2 accuracy: {test_top2:.4f}")
    print(f"Mean winner rank: {test_mean_rank:.4f}")
    if not np.isnan(test_auc):
        print(f"Approx row-level AUC from rank scores: {test_auc:.4f}")

    print_ranked_predictions(final_ranker, test_df, trainval_df, feature_cols)

    print("\nDone.")


if __name__ == "__main__":
    main()