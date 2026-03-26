import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

class RecommendationEngine:
    def __init__(self, books_path, ratings_path):
        self.books_path = books_path
        self.ratings_path = ratings_path
        self.user_book_matrix = None
        self.books_df = None
        self.load_data()

    def load_data(self):
        print("Loading data...")
        if not os.path.exists(self.ratings_path) or not os.path.exists(self.books_path):
            print(f"Error: {self.ratings_path} or {self.books_path} not found.")
            return

        ratings = pd.read_csv(self.ratings_path)
        
        # Consistent with the original script: sample 500 users for performance
        sample_users = ratings['user_id'].drop_duplicates().sample(n=500, random_state=42)
        ratings = ratings[ratings['user_id'].isin(sample_users)]

        self.books_df = pd.read_csv(self.books_path)
        self.books_df = self.books_df[['book_id', 'title']]

        df = pd.merge(ratings, self.books_df, on="book_id")

        self.user_book_matrix = df.pivot_table(
            index='user_id',
            columns='book_id', # Use book_id instead of title for reliability
            values='rating'
        ).fillna(0)

        # Filtering to handle sparsity (consistent with notebook)
        # Keep books with > 50 ratings and users with > 20 ratings
        # However, with only 500 users, these thresholds might be too high.
        # Let's adjust slightly for the sample.
        self.user_book_matrix = self.user_book_matrix.loc[:, (self.user_book_matrix > 0).sum(axis=0) > 2]
        self.user_book_matrix = self.user_book_matrix[(self.user_book_matrix > 0).sum(axis=1) > 2]
        
        print(f"Matrix shape: {self.user_book_matrix.shape}")

    def get_user_similarity(self):
        return self.user_book_matrix.T.corr()

    def get_item_similarity(self):
        sim = cosine_similarity(self.user_book_matrix.T)
        return pd.DataFrame(sim, index=self.user_book_matrix.columns, columns=self.user_book_matrix.columns)

    def recommend_user_based(self, user_id, top_n=10):
        if self.user_book_matrix is None or user_id not in self.user_book_matrix.index:
            return []

        sim = self.get_user_similarity()
        if user_id not in sim.index:
            return []

        similar_users = sim[user_id].sort_values(ascending=False).drop(user_id)
        
        recs = []
        for other_user in similar_users.index[:5]:
            # Get books the other user liked (rating >= 4) that the current user hasn't rated
            other_user_ratings = self.user_book_matrix.loc[other_user]
            user_ratings = self.user_book_matrix.loc[user_id]
            
            potential_books = other_user_ratings[(other_user_ratings >= 4) & (user_ratings == 0)].index
            for book_id in potential_books:
                if book_id not in recs:
                    recs.append(int(book_id))
        
        return recs[:top_n]

    def recommend_item_based(self, user_id, top_n=10):
        if self.user_book_matrix is None or user_id not in self.user_book_matrix.index:
            return []

        sim = self.get_item_similarity()
        user_ratings = self.user_book_matrix.loc[user_id]
        liked_books = user_ratings[user_ratings >= 4].index

        recs = []
        for book_id in liked_books:
            if book_id in sim.index:
                similar_items = sim[book_id].sort_values(ascending=False).drop(book_id)
                for item_id in similar_items.index:
                    if self.user_book_matrix.loc[user_id, item_id] == 0 and item_id not in recs:
                        recs.append(int(item_id))
                        break # Just one similar item per liked book for diversity
        
        return recs[:top_n]

    def add_rating(self, user_id, book_id, rating):
        if self.user_book_matrix is None:
            return False
            
        # Add book_id to columns if not present
        if book_id not in self.user_book_matrix.columns:
            self.user_book_matrix[book_id] = 0.0
            
        # Add user_id to index if not present
        if user_id not in self.user_book_matrix.index:
            self.user_book_matrix.loc[user_id] = 0.0
            
        self.user_book_matrix.loc[user_id, book_id] = float(rating)
        return True
