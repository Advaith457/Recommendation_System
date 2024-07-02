import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

movies = {
    'movie_id': [1, 2, 3, 4, 5],
    'title': ["The Matrix", "Toy Story", "The Godfather", "Pulp Fiction", "The Dark Knight"],
    'genre': ["Action, Sci-Fi", "Animation, Adventure", "Crime, Drama", "Crime, Drama", "Action, Crime"]
}

ratings = {
    'user_id': [1, 1, 1, 2, 2, 3, 3, 4, 4],
    'movie_id': [1, 2, 3, 1, 4, 2, 3, 5, 4],
    'rating': [5, 3, 4, 4, 5, 3, 4, 5, 3]
}

movies_df = pd.DataFrame(movies)
ratings_df = pd.DataFrame(ratings)

def content_based_recommender(movie_title, movies_df, n_recommendations=3):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['genre'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    movie_idx = movies_df.index[movies_df['title'] == movie_title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[movie_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n_recommendations + 1]
    movie_indices = [i[0] for i in sim_scores]
    return movies_df['title'].iloc[movie_indices]

def collaborative_recommender(user_id, ratings_df, movies_df, n_recommendations=3):
    user_item_matrix = ratings_df.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
    user_ratings = user_item_matrix.loc[user_id]
    sim_scores = user_similarity_df[user_id]
    weighted_sum = np.dot(sim_scores, user_item_matrix)
    sim_sum = np.abs(sim_scores).sum()
    weighted_avg = weighted_sum / sim_sum
    recommendations = pd.DataFrame(weighted_avg, index=user_item_matrix.columns, columns=['predicted_rating'])
    already_rated = user_ratings[user_ratings > 0].index
    recommendations = recommendations[~recommendations.index.isin(already_rated)]
    top_recommendations = recommendations.sort_values(by='predicted_rating', ascending=False).head(n_recommendations)
    return movies_df[movies_df['movie_id'].isin(top_recommendations.index)]['title']

print("Content-Based Recommendations for 'The Matrix':")
print(content_based_recommender("The Matrix", movies_df))
print("\nCollaborative Filtering Recommendations for User 1:")
print(collaborative_recommender(1, ratings_df, movies_df))