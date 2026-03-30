#!/usr/bin/env python3
"""
Oscar Best Picture prediction using a constrained weighted ranking model.

Why this approach:
- One winner per year => ranking problem, not plain classification
- Small dataset => simple model generalizes better than flexible ML
- Uses only training years to tune weights (no test leakage)

What it does:
1. Build movie-level features
2. Convert core numeric signals into within-year percentile ranks
3. Search over small feature subsets + weight grids
4. Optimize on training years only using:
   - per-year winner accuracy
   - mean winner rank
5. Apply best rule to test years
"""

from __future__ import annotations

import argparse
import itertools
import math
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

    for col in ["isFresh", "isTopCritic"]:
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


def add_rank_pct_within_year(df: pd.DataFrame, col: str, higher_is_better: bool = True) -> pd.Series:
    """
    Returns within-year percentile rank in [0,1].
    Higher output should always mean 'better'.
    """
    ascending = higher_is_better
    ranked = df.groupby("movieYear")[col].rank(method="average", pct=True, ascending=ascending)
    return ranked.fillna(0.5)


# ----------------------------
# Feature building
# ----------------------------

def build_features(nominees: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    df = nominees.copy()
    review_agg = aggregate_reviews(reviews)
    df = df.merge(review_agg, on="movieId", how="left")

    for c in ["review_count", "fresh_count", "top_critic_count", "fresh_ratio", "top_critic_ratio"]:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    for col in ["critic_score", "audience_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    df["critic_score"] = df["critic_score"].fillna(df["critic_score"].median())
    df["audience_score"] = df["audience_score"].fillna(df["audience_score"].median())

    if "runtime" in df.columns:
        df["runtime_min"] = df["runtime"].apply(parse_runtime_to_minutes)
    else:
        df["runtime_min"] = np.nan
    df["runtime_min"] = df["runtime_min"].fillna(df["runtime_min"].median())

    if "release_date_theaters" in df.columns:
        df["release_month"] = df["release_date_theaters"].apply(parse_release_month)
    else:
        df["release_month"] = np.nan

    if "original_language" in df.columns:
        lang = df["original_language"].astype(str).str.strip().str.lower()
        df["is_english"] = lang.str.contains("english", na=False).astype(int)
    else:
        df["is_english"] = 0

    df["winner_clean"] = safe_bool_series(df["winner"])

    # Timing features from your EDA
    df["is_january"] = df["release_month"].eq(1).astype(int)
    df["is_late_year"] = df["release_month"].isin([9, 10, 11, 12]).astype(int)
    df["is_nov_dec"] = df["release_month"].isin([11, 12]).astype(int)

    # Runtime features
    df["runtime_diff_137"] = (df["runtime_min"] - 137).abs()
    df["runtime_good_band"] = df["runtime_min"].between(125, 150).astype(int)
    df["runtime_tight_band"] = df["runtime_min"].between(132, 142).astype(int)

    # Within-year ranks: higher is always better
    df["critic_rank"] = add_rank_pct_within_year(df, "critic_score", higher_is_better=True)
    df["audience_rank"] = add_rank_pct_within_year(df, "audience_score", higher_is_better=True)
    df["fresh_rank"] = add_rank_pct_within_year(df, "fresh_ratio", higher_is_better=True)
    df["top_critic_rank"] = add_rank_pct_within_year(df, "top_critic_ratio", higher_is_better=True)

    # Smaller distance is better => invert via negative
    df["runtime_closeness_rank"] = add_rank_pct_within_year(df, "runtime_diff_137", higher_is_better=False)

    # Score gap / consensus style features
    df["critic_minus_audience"] = df["critic_score"] - df["audience_score"]
    df["consensus_gap_small"] = (df["critic_minus_audience"].abs() <= 10).astype(int)
    df["critic_minus_audience_rank"] = add_rank_pct_within_year(df, "critic_minus_audience", higher_is_better=True)

    return df


# ----------------------------
# Ranking evaluation
# ----------------------------

ALL_FEATURES = [
    "critic_rank",
    "audience_rank",
    "fresh_rank",
    "top_critic_rank",
    "runtime_closeness_rank",
    "runtime_good_band",
    "runtime_tight_band",
    "is_january",
    "is_late_year",
    "is_nov_dec",
    "is_english",
    "consensus_gap_small",
]


def apply_score(df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    score = np.zeros(len(out), dtype=float)

    for feat, w in weights.items():
        score += w * out[feat].astype(float).values

    out["score"] = score
    return out


def winner_rank_metrics(df: pd.DataFrame, weights: dict[str, float]) -> tuple[float, float]:
    """
    Returns:
    - per-year winner accuracy
    - mean winner rank (1 is best)
    """
    scored = apply_score(df, weights)

    accuracies = []
    winner_ranks = []

    for _, g in scored.groupby("movieYear"):
        g = g.sort_values("score", ascending=False).reset_index(drop=True)
        g["pred_rank"] = np.arange(1, len(g) + 1)

        winner_row = g[g["winner_clean"] == 1]
        if len(winner_row) != 1:
            continue

        winner_rank = int(winner_row.iloc[0]["pred_rank"])
        winner_ranks.append(winner_rank)
        accuracies.append(1 if winner_rank == 1 else 0)

    if not winner_ranks:
        return 0.0, float("inf")

    return float(np.mean(accuracies)), float(np.mean(winner_ranks))


def predict_top_each_year(df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    scored = apply_score(df, weights)
    top_each_year = (
        scored.sort_values(["movieYear", "score"], ascending=[True, False])
        .groupby("movieYear", as_index=False)
        .head(1)
        .copy()
    )
    return top_each_year.sort_values("movieYear").reset_index(drop=True)


# ----------------------------
# Search
# ----------------------------

def generate_feature_subsets():
    """
    Keep the search constrained and sensible.
    """
    core_choices = [
        ["critic_rank", "top_critic_rank", "runtime_closeness_rank", "is_late_year"],
        ["critic_rank", "fresh_rank", "runtime_closeness_rank", "is_late_year"],
        ["critic_rank", "top_critic_rank", "runtime_closeness_rank", "is_nov_dec"],
        ["critic_rank", "fresh_rank", "runtime_good_band", "is_late_year"],
        ["critic_rank", "audience_rank", "top_critic_rank", "runtime_closeness_rank", "is_late_year"],
        ["critic_rank", "fresh_rank", "top_critic_rank", "runtime_closeness_rank", "is_late_year"],
        ["critic_rank", "fresh_rank", "top_critic_rank", "runtime_closeness_rank", "is_nov_dec"],
        ["critic_rank", "fresh_rank", "top_critic_rank", "runtime_good_band", "is_late_year"],
    ]

    optional_pool = ["is_english", "runtime_tight_band", "is_january", "consensus_gap_small"]

    subsets = []
    for base in core_choices:
        subsets.append(base)
        for opt in optional_pool:
            subsets.append(base + [opt])
        for opt1, opt2 in itertools.combinations(optional_pool, 2):
            subsets.append(base + [opt1, opt2])

    # deduplicate
    unique = []
    seen = set()
    for s in subsets:
        key = tuple(sorted(s))
        if key not in seen:
            seen.add(key)
            unique.append(list(key))
    return unique


def generate_weight_options(feature_name: str):
    if feature_name in {"critic_rank", "fresh_rank", "top_critic_rank"}:
        return [0.5, 1.0, 1.5, 2.0, 2.5]
    if feature_name in {"audience_rank"}:
        return [0.0, 0.25, 0.5, 0.75, 1.0]
    if feature_name in {"runtime_closeness_rank"}:
        return [0.25, 0.5, 0.75, 1.0, 1.25]
    if feature_name in {"runtime_good_band", "runtime_tight_band"}:
        return [0.0, 0.25, 0.5, 0.75, 1.0]
    if feature_name in {"is_january", "is_late_year", "is_nov_dec"}:
        return [0.0, 0.25, 0.5, 0.75, 1.0]
    if feature_name in {"is_english", "consensus_gap_small"}:
        return [0.0, 0.1, 0.25, 0.5]
    return [0.0, 0.5, 1.0]


def search_best_weights(train_df: pd.DataFrame):
    best = None
    results = []

    for subset in generate_feature_subsets():
        option_lists = [generate_weight_options(f) for f in subset]

        for values in itertools.product(*option_lists):
            weights = dict(zip(subset, values))

            if sum(weights.values()) == 0:
                continue

            acc, mean_rank = winner_rank_metrics(train_df, weights)

            # Primary: per-year accuracy
            # Secondary: lower winner rank
            # Tertiary: slightly prefer simpler models
            complexity_penalty = 0.001 * len(subset)
            objective = acc - 0.02 * (mean_rank - 1.0) - complexity_penalty

            row = {
                "weights": weights,
                "features": subset,
                "train_per_year_acc": acc,
                "train_mean_winner_rank": mean_rank,
                "objective": objective,
            }
            results.append(row)

            if best is None or objective > best["objective"]:
                best = row

    results_df = pd.DataFrame(results).sort_values(
        ["objective", "train_per_year_acc", "train_mean_winner_rank"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    return best, results_df


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

    best, results_df = search_best_weights(train_df)
    best_weights = best["weights"]

    train_acc, train_mean_rank = winner_rank_metrics(train_df, best_weights)
    test_acc, test_mean_rank = winner_rank_metrics(test_df, best_weights)

    test_preds = predict_top_each_year(test_df, best_weights)

    print("\n=== Best feature subset ===")
    print(best["features"])

    print("\n=== Best weights ===")
    for k, v in best_weights.items():
        print(f"{k}: {v}")

    print("\n=== Best training objective ===")
    print(f"Objective: {best['objective']:.4f}")
    print(f"Train per-year accuracy: {train_acc:.4f}")
    print(f"Train mean winner rank: {train_mean_rank:.4f}")

    print("\n=== Test metrics ===")
    print(f"Test per-year accuracy: {test_acc:.4f}")
    print(f"Test mean winner rank: {test_mean_rank:.4f}")

    print("\n=== Top 10 searched models ===")
    print(
        results_df[["features", "train_per_year_acc", "train_mean_winner_rank", "objective"]]
        .head(10)
        .to_string(index=False)
    )

    print("\n=== Top predicted nominee per year (test) ===")
    for _, row in test_preds.iterrows():
        print(
            f"{int(row['movieYear'])}: {row['movieTitle']} | "
            f"score={row['score']:.3f} | winner={bool(row['winner_clean'])}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()