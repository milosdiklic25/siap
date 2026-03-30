#!/usr/bin/env python3
"""
Oscar Best Picture prediction using a simple weighted ranking model.

Idea:
- Treat each year as one contest.
- Convert movie features into within-year relative ranks.
- Learn a small set of weights by leave-one-year-out validation on training years.
- Pick the movie with the highest weighted score within each year.

This is intentionally simple and constrained so it does not overfit like the
classification and pairwise models.
"""

from __future__ import annotations

import argparse
import itertools
import re
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ----------------------------
# Helpers
# ----------------------------

def parse_runtime_to_minutes(runtime_val) -> float:
    if pd.isna(runtime_val):
        return np.nan

    s = str(runtime_val).strip().lower()

    hm_match = re.search(r"(\d+)\s*h\s*(\d+)\s*m", s)
    if hm_match:
        return float(int(hm_match.group(1)) * 60 + int(hm_match.group(2)))

    h_match = re.search(r"(\d+)\s*h", s)
    if h_match:
        return float(int(h_match.group(1)) * 60)

    m_match = re.search(r"(\d+)\s*(m|min|minutes)", s)
    if m_match:
        return float(int(m_match.group(1)))

    num_match = re.fullmatch(r"\d+", s)
    if num_match:
        return float(int(num_match.group(0)))

    return np.nan


def parse_release_month(date_val) -> float:
    if pd.isna(date_val):
        return np.nan

    s = str(date_val).strip()
    s = re.sub(r",\s*(Wide|Limited|Original)$", "", s, flags=re.IGNORECASE)

    dt = pd.to_datetime(s, errors="coerce")
    if pd.isna(dt):
        return np.nan

    return float(dt.month)


def safe_bool_series(x: pd.Series) -> pd.Series:
    if x.dtype == bool:
        return x.astype(int)

    lowered = x.astype(str).str.strip().str.lower()
    mapped = lowered.map({
        "true": 1, "false": 0,
        "1": 1, "0": 0,
        "yes": 1, "no": 0
    })

    numeric = pd.to_numeric(x, errors="coerce")
    return mapped.fillna(numeric).fillna(0).astype(int)


def aggregate_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    df = reviews.copy()

    for col in ["isFresh", "isRotten", "isTopCritic"]:
        if col not in df.columns:
            df[col] = False

    df["isFresh_i"] = safe_bool_series(df["isFresh"])
    df["isTopCritic_i"] = safe_bool_series(df["isTopCritic"])

    agg = df.groupby("movieId").agg(
        review_count=("movieId", "size"),
        fresh_count=("isFresh_i", "sum"),
        top_critic_count=("isTopCritic_i", "sum"),
    ).reset_index()

    eps = 1e-9
    agg["fresh_ratio"] = agg["fresh_count"] / (agg["review_count"] + eps)
    agg["top_critic_ratio"] = agg["top_critic_count"] / (agg["review_count"] + eps)

    return agg


def add_rank_pct_within_year(df: pd.DataFrame, col: str, ascending: bool = True) -> pd.Series:
    """
    Percentile rank within each Oscar year.
    Higher values should always mean 'better' for the final score.
    """
    ranked = df.groupby("movieYear")[col].rank(method="average", pct=True, ascending=ascending)
    return ranked.fillna(0.5)


# ----------------------------
# Feature engineering
# ----------------------------

def build_features(nominees: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    df = nominees.copy()
    review_agg = aggregate_reviews(reviews)
    df = df.merge(review_agg, on="movieId", how="left")

    for c in ["review_count", "fresh_count", "top_critic_count", "fresh_ratio", "top_critic_ratio"]:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    if "runtime" in df.columns:
        df["runtime_min"] = df["runtime"].apply(parse_runtime_to_minutes)
    else:
        df["runtime_min"] = np.nan

    df["runtime_min"] = df["runtime_min"].fillna(df["runtime_min"].median())

    if "release_date_theaters" in df.columns:
        df["release_month"] = df["release_date_theaters"].apply(parse_release_month)
    else:
        df["release_month"] = np.nan

    for col in ["critic_score", "audience_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = np.nan

    if "original_language" in df.columns:
        lang = df["original_language"].astype(str).str.strip().str.lower()
        df["is_english"] = lang.str.contains("english", na=False).astype(int)
    else:
        df["is_english"] = 0

    df["is_oscar_season"] = df["release_month"].isin([1, 11, 12]).astype(int)

    # Runtime closeness: smaller distance is better
    df["runtime_diff_from_137"] = (df["runtime_min"] - 137).abs()

    # Winner target
    df["winner_clean"] = safe_bool_series(df["winner"])

    # Within-year relative ranks (higher = better)
    df["critic_rank"] = add_rank_pct_within_year(df, "critic_score", ascending=True)
    df["audience_rank"] = add_rank_pct_within_year(df, "audience_score", ascending=True)
    df["fresh_rank"] = add_rank_pct_within_year(df, "fresh_ratio", ascending=True)
    df["top_critic_rank"] = add_rank_pct_within_year(df, "top_critic_ratio", ascending=True)

    # For runtime distance, smaller is better, so ascending=False
    # Example: smallest distance gets the highest percentile rank
    df["runtime_closeness_rank"] = add_rank_pct_within_year(df, "runtime_diff_from_137", ascending=False)

    return df


# ----------------------------
# Weighted scoring model
# ----------------------------

FEATURES = [
    "critic_rank",
    "audience_rank",
    "fresh_rank",
    "top_critic_rank",
    "runtime_closeness_rank",
    "is_oscar_season",
    "is_english",
]


def apply_weighted_score(df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    score = np.zeros(len(out), dtype=float)

    for feat in FEATURES:
        score += weights.get(feat, 0.0) * out[feat].astype(float).values

    out["score"] = score
    return out


def predict_winners_by_year(df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    scored = apply_weighted_score(df, weights)
    top_each_year = (
        scored.sort_values(["movieYear", "score"], ascending=[True, False])
        .groupby("movieYear", as_index=False)
        .head(1)
        .copy()
    )
    return top_each_year.sort_values("movieYear").reset_index(drop=True)


def per_year_accuracy(df: pd.DataFrame, weights: dict[str, float]) -> float:
    preds = predict_winners_by_year(df, weights)
    if len(preds) == 0:
        return 0.0
    return preds["winner_clean"].mean()


def leave_one_year_out_score(train_df: pd.DataFrame, weights: dict[str, float]) -> float:
    years = sorted(train_df["movieYear"].unique())
    fold_accs = []

    for year in years:
        val_df = train_df[train_df["movieYear"] == year].copy()
        if len(val_df) == 0:
            continue

        # This model doesn't "fit" on fold data in the usual sense;
        # weights are fixed and evaluated on each held-out year.
        acc = per_year_accuracy(val_df, weights)
        fold_accs.append(acc)

    return float(np.mean(fold_accs)) if fold_accs else 0.0


def grid_search_weights(train_df: pd.DataFrame):
    """
    Search over a small, interpretable grid.

    Keep it small enough to run quickly, but broad enough to find a better rule.
    """
    best_weights = None
    best_score = -1.0
    all_results = []

    grid = {
        "critic_rank": [0.0, 0.5, 1.0, 1.5, 2.0],
        "audience_rank": [0.0, 0.25, 0.5, 0.75, 1.0],
        "fresh_rank": [0.0, 0.5, 1.0, 1.5],
        "top_critic_rank": [0.0, 0.5, 1.0, 1.5],
        "runtime_closeness_rank": [0.0, 0.25, 0.5, 0.75, 1.0],
        "is_oscar_season": [0.0, 0.5, 1.0, 1.5],
        "is_english": [0.0, 0.25, 0.5, 0.75],
    }

    keys = list(grid.keys())
    value_product = itertools.product(*(grid[k] for k in keys))

    for values in value_product:
        weights = dict(zip(keys, values))

        # avoid completely empty model
        if sum(weights.values()) == 0:
            continue

        score = leave_one_year_out_score(train_df, weights)
        all_results.append((weights, score))

        if score > best_score:
            best_score = score
            best_weights = weights

    results_df = pd.DataFrame([
        {**w, "cv_per_year_accuracy": s}
        for w, s in all_results
    ]).sort_values("cv_per_year_accuracy", ascending=False)

    return best_weights, best_score, results_df


# ----------------------------
# Main
# ----------------------------

@dataclass
class Config:
    nominees_csv: str
    reviews_csv: str
    split_year: int


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nominees_csv", default="movies_after_1970_best_picture_nominees_with_winner.csv")
    parser.add_argument("--reviews_csv", default="critic_reviews_normalized_textprocessed_sentiment.csv")
    parser.add_argument("--split_year", type=int, default=2015)
    args = parser.parse_args()

    cfg = Config(
        nominees_csv=args.nominees_csv,
        reviews_csv=args.reviews_csv,
        split_year=args.split_year,
    )

    nominees = pd.read_csv(cfg.nominees_csv)
    reviews = pd.read_csv(cfg.reviews_csv)

    required_cols = {"movieId", "movieYear", "winner", "movieTitle"}
    missing = required_cols - set(nominees.columns)
    if missing:
        print(f"ERROR: nominees CSV missing columns: {sorted(missing)}", file=sys.stderr)
        sys.exit(1)

    if "movieId" not in reviews.columns:
        print("ERROR: reviews CSV must include movieId.", file=sys.stderr)
        sys.exit(1)

    df = build_features(nominees, reviews)

    train_df = df[df["movieYear"] <= cfg.split_year].copy()
    test_df = df[df["movieYear"] > cfg.split_year].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        print("ERROR: Empty train or test split.", file=sys.stderr)
        sys.exit(1)

    best_weights, best_cv_score, results_df = grid_search_weights(train_df)

    train_preds = predict_winners_by_year(train_df, best_weights)
    test_preds = predict_winners_by_year(test_df, best_weights)

    train_acc = train_preds["winner_clean"].mean() if len(train_preds) else 0.0
    test_acc = test_preds["winner_clean"].mean() if len(test_preds) else 0.0

    print("\n=== Features used ===")
    print(FEATURES)

    print("\n=== Split ===")
    print(f"Train years: <= {cfg.split_year}  (movies={len(train_df)})")
    print(f"Test years : >  {cfg.split_year}  (movies={len(test_df)})")
    print(f"Train winners: {int(train_df['winner_clean'].sum())}")
    print(f"Test winners : {int(test_df['winner_clean'].sum())}")

    print("\n=== Best weights from leave-one-year-out search ===")
    for k, v in best_weights.items():
        print(f"{k}: {v}")
    print(f"CV per-year accuracy: {best_cv_score:.4f}")

    print("\n=== Top 10 weight combinations ===")
    print(results_df.head(10).to_string(index=False))

    print("\n=== Train per-year top-pick accuracy ===")
    print(f"Accuracy: {train_acc:.4f}")

    print("\n=== Test per-year top-pick accuracy ===")
    print(f"Accuracy: {test_acc:.4f}")

    print("\n=== Top predicted nominee per year (test) ===")
    for _, row in test_preds.iterrows():
        print(
            f"{int(row['movieYear'])}: {row['movieTitle']} | "
            f"score={row['score']:.3f} | winner={bool(row['winner_clean'])}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()