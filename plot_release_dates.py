import pandas as pd
import matplotlib.pyplot as plt

def main():
    # ===== 1. Load data =====
    file_path = "movies_after_1970_best_picture_nominees_with_winner.csv"
    df = pd.read_csv(file_path)

    # ===== 2. Clean release_date_theaters =====
    # Remove trailing labels like ", Wide", ", Original", ", Limited"
    df["release_date_theaters_clean"] = (
        df["release_date_theaters"]
        .astype(str)
        .str.strip()
        .str.replace(r",\s*(Wide|Original|Limited)$", "", regex=True)
    )

    # Convert cleaned dates to datetime
    df["release_date_theaters_clean"] = pd.to_datetime(
        df["release_date_theaters_clean"],
        format="%b %d, %Y",
        errors="coerce"
    )

    # ===== 3. Keep only winners =====
    # Works whether winner is 1/0 or True/False
    winners = df[df["winner"].astype(int) == 1].copy()

    # Drop rows where cleaned release date is missing
    winners = winners.dropna(subset=["release_date_theaters_clean"])

    # Extract release month
    winners["release_month"] = winners["release_date_theaters_clean"].dt.month

    # ===== 4. Count winners by month =====
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    monthly_counts = winners["release_month"].value_counts().sort_index()
    monthly_counts = monthly_counts.reindex(range(1, 13), fill_value=0)

    # Highlight later-year months
    colors = ["orange"] + ["gray"] * 9 + ["orange", "orange"]

    # ===== 5. Plot =====
    plt.figure(figsize=(12, 6))
    bars = plt.bar(month_names, monthly_counts.values, color=colors)

    plt.title("Best Picture Winners by Theatrical Release Month")
    plt.xlabel("Release Month")
    plt.ylabel("Number of Winning Movies")

    for bar, value in zip(bars, monthly_counts.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.05,
            str(value),
            ha="center"
        )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()