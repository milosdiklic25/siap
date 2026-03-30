import pandas as pd
import matplotlib.pyplot as plt

def main():
    # ===== 1. Load data =====
    file_path = "movies_after_1970_best_picture_nominees_with_winner.csv"
    df = pd.read_csv(file_path)

    # ===== 2. Clean columns =====
    # Normalize language text
    df["language_clean"] = (
        df["original_language"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Convert winner column to numeric
    df["winner_clean"] = pd.to_numeric(df["winner"], errors="coerce")

    # ===== 3. Count winners per language =====
    winners = df[df["winner_clean"] == 1].copy()

    language_counts = (
        winners["language_clean"]
        .value_counts()
        .sort_values(ascending=False)
    )

    # ===== 4. Plot =====
    plt.figure(figsize=(10, 6))
    bars = plt.bar(language_counts.index, language_counts.values)

    plt.title("Number of Best Picture Winners by Original Language")
    plt.xlabel("Language")
    plt.ylabel("Number of Winners")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Add labels on bars
    for bar, value in zip(bars, language_counts.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.05,
            str(value),
            ha="center"
        )

    plt.show()

    # ===== 5. Print summary =====
    print("Winners per language:")
    print(language_counts)

    # ===== Find specific non-English winners =====

    # Clean language again (just to be safe)
    df["language_clean"] = (
        df["original_language"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["winner_clean"] = pd.to_numeric(df["winner"], errors="coerce")

    winners = df[df["winner_clean"] == 1].copy()

    # Languages of interest
    target_languages = ["english (united kingdom)", "korean", "nan"]

    subset = winners[winners["language_clean"].isin(target_languages)]

    # Print results nicely
    for lang in target_languages:
        print(f"\n=== {lang.upper()} ===")
        movies = subset[subset["language_clean"] == lang]
        
        if movies.empty:
            print("No movies found")
        else:
            for _, row in movies.iterrows():
                print(f"{row['movieYear']} - {row['movieTitle']}")

if __name__ == "__main__":
    main()