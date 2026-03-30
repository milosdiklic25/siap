import pandas as pd
import matplotlib.pyplot as plt

def main():
    # ===== 1. Load data =====
    file_path = "movies_after_1970_best_picture_nominees_with_winner.csv"
    df = pd.read_csv(file_path)

    # ===== 2. Clean score columns =====
    df["critic_score_clean"] = pd.to_numeric(df["critic_score"], errors="coerce")
    df["audience_score_clean"] = pd.to_numeric(df["audience_score"], errors="coerce")
    df["winner_clean"] = pd.to_numeric(df["winner"], errors="coerce")

    # Keep only rows with valid scores
    df = df.dropna(subset=["critic_score_clean", "audience_score_clean", "winner_clean"]).copy()
    df["winner_clean"] = df["winner_clean"].astype(int)

    # Split into winners and non-winners
    nominees = df[df["winner_clean"] == 0]
    winners = df[df["winner_clean"] == 1]

    # ===== 3. Plot =====
    plt.figure(figsize=(10, 7))

    plt.scatter(
        nominees["audience_score_clean"],
        nominees["critic_score_clean"],
        label="Nominees",
        alpha=0.7
    )

    plt.scatter(
        winners["audience_score_clean"],
        winners["critic_score_clean"],
        label="Winners",
        alpha=0.9
    )

    plt.xlabel("Audience Score")
    plt.ylabel("Critic Score")
    plt.title("Critic Score vs Audience Score\nBest Picture Nominees and Winners")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ===== 4. Basic summary =====
    print("Average scores:")
    print(f"Nominees  - Audience: {nominees['audience_score_clean'].mean():.2f}, Critic: {nominees['critic_score_clean'].mean():.2f}")
    print(f"Winners   - Audience: {winners['audience_score_clean'].mean():.2f}, Critic: {winners['critic_score_clean'].mean():.2f}")

    # Simple correlations with winning
    print("\nCorrelation with winner:")
    print(f"Audience score vs winner: {df['audience_score_clean'].corr(df['winner_clean']):.3f}")
    print(f"Critic score vs winner:   {df['critic_score_clean'].corr(df['winner_clean']):.3f}")

if __name__ == "__main__":
    main()