import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Load dataset
    df = pd.read_csv("movies_after_1970_best_picture_nominees_with_winner.csv")

    # Make sure critic_score is numeric
    df["critic_score"] = pd.to_numeric(df["critic_score"], errors="coerce")

    # Group by year and calculate mean critic score
    yearly_trend = (
        df.groupby("movieYear")["critic_score"]
        .mean()
        .reset_index()
    )

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(yearly_trend["movieYear"], yearly_trend["critic_score"], marker="o")
    plt.xlabel("Movie Year")
    plt.ylabel("Average Critic Score")
    plt.title("Average Critic Score of Best Picture Nominees by Year")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()