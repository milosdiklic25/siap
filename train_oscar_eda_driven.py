#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


# ----------------------------
# Helpers
# ----------------------------

def parse_runtime(runtime):
    if pd.isna(runtime):
        return np.nan
    s = str(runtime).lower()

    h = re.search(r"(\d+)\s*h", s)
    m = re.search(r"(\d+)\s*m", s)

    hours = int(h.group(1)) if h else 0
    mins = int(m.group(1)) if m else 0

    if not h and not m:
        if s.isdigit():
            return float(int(s))
        return np.nan

    return hours * 60 + mins


def parse_month(date):
    if pd.isna(date):
        return np.nan
    try:
        return pd.to_datetime(date).month
    except:
        return np.nan


def safe_bool(x):
    return (
        x.astype(str)
        .str.lower()
        .map({"true": 1, "false": 0, "1": 1, "0": 0})
        .fillna(0)
        .astype(int)
    )


# ----------------------------
# Feature engineering
# ----------------------------

def build_features(df):

    # runtime
    df["runtime_min"] = df["runtime"].apply(parse_runtime)
    df["runtime_min"] = df["runtime_min"].fillna(df["runtime_min"].median())

    # release month
    df["release_month"] = df["release_date_theaters"].apply(parse_month)

    # award season boost
    df["is_award_season"] = df["release_month"].isin([10,11,12]).astype(int)

    # late fall bias
    df["late_fall"] = df["release_month"].isin([11,12]).astype(int)

    # runtime sweet spot (Oscars LOVE this)
    df["runtime_sweet_spot"] = df["runtime_min"].between(100, 140).astype(int)

    # language bias
    df["is_english"] = (df["original_language"] == "en").astype(int)

    # score interaction (light usage)
    df["score_mean"] = (df["critic_score"] + df["audience_score"]) / 2.0
    df["score_gap"] = (df["critic_score"] - df["audience_score"]).abs()

    # normalize scores
    df["critic_score"] = pd.to_numeric(df["critic_score"], errors="coerce").fillna(0)
    df["audience_score"] = pd.to_numeric(df["audience_score"], errors="coerce").fillna(0)

    return df


# ----------------------------
# Model
# ----------------------------

@dataclass
class Config:
    csv_path: str
    split_year: int


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--split_year", type=int, default=2015)
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)

    df["winner"] = safe_bool(df["winner"])

    df = build_features(df)

    # ----------------------------
    # Feature set (EDA-driven)
    # ----------------------------
    features = [
        "is_award_season",
        "late_fall",
        "runtime_sweet_spot",
        "is_english",
        "release_month",
        "runtime_min",
        "score_mean",
        "score_gap",
    ]

    X = df[features].fillna(0)
    y = df["winner"]

    # ----------------------------
    # Split
    # ----------------------------
    train = df["movieYear"] <= args.split_year
    test = df["movieYear"] > args.split_year

    X_train, y_train = X[train], y[train]
    X_test, y_test = X[test], y[test]

    df_test = df[test].copy()

    # ----------------------------
    # Model
    # ----------------------------
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000))
    ])

    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:,1]

    # ----------------------------
    # Evaluation
    # ----------------------------
    print("\n=== Classification Report ===")
    print(classification_report(y_test, (proba > 0.5).astype(int)))

    # ----------------------------
    # Per-year prediction
    # ----------------------------
    df_test["proba"] = proba

    print("\n=== Top predicted nominee per year ===")

    correct = 0
    total = 0

    for year, group in df_test.groupby("movieYear"):
        total += 1
        top = group.sort_values("proba", ascending=False).iloc[0]

        if top["winner"] == 1:
            correct += 1

        print(
            f"{year}: {top['movieTitle']} | "
            f"proba={top['proba']:.3f} | "
            f"winner={bool(top['winner'])}"
        )

    print("\nAccuracy:", correct / total)


if __name__ == "__main__":
    main()