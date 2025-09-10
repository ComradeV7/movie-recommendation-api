import torch
import torch.nn as nn
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class RecommenderNet(nn.Module):
    def __init__(self, n_users, n_movies, n_factors=50, embedding_dropout=0.05, hidden_layers=[512, 256, 128], dropouts=[0.5, 0.25, 0.1]):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, n_factors, sparse=False)
        self.movie_embedding = nn.Embedding(n_movies, n_factors, sparse=False)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        
        self.hidden_layers = nn.ModuleList()
        input_size = n_factors * 2
        for size in hidden_layers:
            self.hidden_layers.append(nn.Linear(input_size, size))
            input_size = size
        
        self.dropouts = nn.ModuleList()
        for p in dropouts:
            self.dropouts.append(nn.Dropout(p))
            
        self.output_layer = nn.Linear(input_size, 1)
    
    def forward(self, users, movies):
        user_emb = self.user_embedding(users)
        movie_emb = self.movie_embedding(movies)
        x = torch.cat([user_emb, movie_emb], dim=1)
        x = self.embedding_dropout(x)
        
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            x = nn.ReLU()(x)
            x = self.dropouts[i](x)
            
        output = self.output_layer(x)
        return output

# --- 2. Function to load all model artifacts ---
def load_artifacts():
    print("Loading all model artifacts from disk...")
    artifacts = {}
    with open('./model_artifacts/model_artifacts.pkl', 'rb') as f:
        artifacts['collab'] = pickle.load(f)
    with open('./model_artifacts/content_artifacts.pkl', 'rb') as f:
        artifacts['content'] = pickle.load(f)
    artifacts['content_sim_matrix'] = np.load('./model_artifacts/tfidf_sim.npy', allow_pickle=True)
    print("All artifacts loaded successfully.")
    return artifacts

# --- 3. Collaborative Recommendation Logic ---
def get_collaborative_recs(favorite_tmdb_ids, model, artifacts, limit=15):
    collab_artifacts = artifacts.get('collab', {})
    tmdb_to_movielens_id = collab_artifacts.get('tmdb_to_movielens_id', {})
    movielens_to_tmdb_id = collab_artifacts.get('movielens_to_tmdb_id', {})
    movie_to_idx = collab_artifacts.get('movie_to_idx', {})
    idx_to_movie = collab_artifacts.get('idx_to_movie', {})
    movies_df = collab_artifacts.get('movies_df', pd.DataFrame())

    fav_movie_ids = [int(tmdb_to_movielens_id.get(str(tid))) for tid in favorite_tmdb_ids if tmdb_to_movielens_id.get(str(tid))]
    fav_indices = [movie_to_idx.get(mid) for mid in fav_movie_ids if movie_to_idx.get(mid) is not None]
    if not fav_indices: return []

    with torch.no_grad():
        movie_vectors = model.movie_embedding(torch.tensor(fav_indices, dtype=torch.long))
    user_taste_vector = torch.mean(movie_vectors, dim=0, keepdim=True)
    
    all_movie_vectors = model.movie_embedding.weight.data
    similarities = cosine_similarity(user_taste_vector.numpy(), all_movie_vectors.numpy()).flatten()
    top_indices = similarities.argsort()[-50:][::-1]

    recommendations = []
    for idx in top_indices:
        movielens_id = idx_to_movie.get(idx)
        if movielens_id and movielens_id not in fav_movie_ids:
            tmdb_id = movielens_to_tmdb_id.get(str(movielens_id))
            if tmdb_id:
                title = movies_df.loc[movielens_id].get('title', 'Title Not Found')
                recommendations.append({
                    # --- FIX: Convert NumPy int to Python int ---
                    'tmdbId': int(tmdb_id), 
                    'title': title,
                    'score': round(float(similarities[idx]), 4), 
                    'type': 'Collaborative'
                })
        if len(recommendations) >= limit: break
    return recommendations

# --- 4. Content-Based Recommendation Logic ---
def get_content_based_recs(favorite_tmdb_ids, artifacts, limit=15):
    content_artifacts = artifacts.get('content', {})
    sim_matrix = artifacts.get('content_sim_matrix')
    tmdb_id_to_index = content_artifacts.get('tmdb_id_to_index', {})
    index_to_tmdb_id = content_artifacts.get('index_to_tmdb_id', {})
    movies_df = content_artifacts.get('movies_df', pd.DataFrame())

    fav_indices = [tmdb_id_to_index.get(tid) for tid in favorite_tmdb_ids if tmdb_id_to_index.get(tid) is not None]
    if not fav_indices: return []

    avg_sim_scores = np.mean(sim_matrix[fav_indices], axis=0)
    top_indices = avg_sim_scores.argsort()[-50:][::-1]

    recommendations = []
    for idx in top_indices:
        tmdb_id = index_to_tmdb_id.get(idx)
        if tmdb_id and tmdb_id not in favorite_tmdb_ids:
            title = movies_df.loc[tmdb_id].get('title', 'Title Not Found')
            recommendations.append({
                # --- FIX: Convert NumPy int to Python int ---
                'tmdbId': int(tmdb_id), 
                'title': title,
                'score': round(float(avg_sim_scores[idx]), 4), 
                'type': 'Content-Based'
            })
        if len(recommendations) >= limit: break
    return recommendations
