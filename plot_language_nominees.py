import pandas as pd
import matplotlib.pyplot as plt

def main():
    # ===== 1. Load data =====
    file_path = "movies_after_1970_best_picture_nominees_with_winner.csv"
    df = pd.read_csv(file_path)

    # ===== 2. Clean columns =====
    df["language_clean"] = (
        df["original_language"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["winner_clean"] = pd.to_numeric(df["winner"], errors="coerce")

    # ===== 3. Count ALL nominees per language =====
    language_counts = (
        df["language_clean"]
        .value_counts()
        .sort_values(ascending=False)
    )

    # ===== 4. Plot =====
    plt.figure(figsize=(10, 6))
    bars = plt.bar(language_counts.index, language_counts.values)

    plt.title("Number of Best Picture Nominees by Original Language")
    plt.xlabel("Language")
    plt.ylabel("Number of Nominees")
    plt.xticks(rotation=45)
    plt.tight_layout()

    for bar, value in zip(bars, language_counts.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.05,
            str(value),
            ha="center"
        )

    plt.show()

    # ===== 5. Print summary =====
    print("Nominees per language:")
    print(language_counts)

    # ===== 6. Print movies for specific languages among ALL nominees =====
    target_languages = ["english (united kingdom)", "korean", "nan"]

    subset = df[df["language_clean"].isin(target_languages)]

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