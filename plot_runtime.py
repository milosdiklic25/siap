import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

def main():
    file_path = "movies_after_1970_best_picture_nominees_with_winner.csv"
    df = pd.read_csv(file_path)

    df["winner_clean"] = pd.to_numeric(df["winner"], errors="coerce")
    winners = df[df["winner_clean"] == 1].copy()

    def parse_runtime(runtime_value):
        if pd.isna(runtime_value):
            return None

        s = str(runtime_value).strip().lower()

        match_hm = re.search(r'(\d+)\s*h\s*(\d+)\s*m', s)
        if match_hm:
            return int(match_hm.group(1)) * 60 + int(match_hm.group(2))

        match_h = re.search(r'(\d+)\s*h', s)
        if match_h:
            return int(match_h.group(1)) * 60

        match_m = re.search(r'(\d+)\s*(m|min|minutes)', s)
        if match_m:
            return int(match_m.group(1))

        if re.fullmatch(r'\d+', s):
            return int(s)

        return None

    winners["runtime_minutes"] = winners["runtime"].apply(parse_runtime)
    winners = winners.dropna(subset=["runtime_minutes"])

    plt.figure(figsize=(10, 6))
    plt.hist(winners["runtime_minutes"], bins=12)

    plt.title("Distribution of Best Picture Winners' Runtime")
    plt.xlabel("Runtime (minutes)")
    plt.ylabel("Number of Winning Movies")
    plt.tight_layout()
    plt.show()

    print(f"Average runtime of winners: {winners['runtime_minutes'].mean():.2f} minutes")

if __name__ == "__main__":
    main()