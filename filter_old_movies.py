import pandas as pd

# 1. Load the dataset
movies = pd.read_csv('movies.csv')

# 2. Remove movies older than 1970
filtered_movies = movies[movies['movieYear'] >= 1970]

# 3. Save the new dataset to a CSV file
filtered_movies.to_csv('movies_after_1970.csv', index=False)

print("Filtering complete. New dataset saved as 'movies_after_1970.csv'.")