import pandas as pd

# 1. Load dataset
data = pd.read_csv("full_data.csv")

# 2. Clean columns (safe practice)
data["CanonicalCategory"] = data["CanonicalCategory"].str.strip().str.upper()
data["Year"] = pd.to_numeric(data["Year"], errors="coerce")

# 3. Apply both filters
filtered_data = data[
    (data["CanonicalCategory"] == "BEST PICTURE") &
    (data["Year"] >= 1970)
]

# 4. Save to new CSV
filtered_data.to_csv("best_picture_after_1970.csv", index=False)

print("Filtering complete. Saved as 'best_picture_after_1970.csv'.")