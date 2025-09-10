from flask import Flask, jsonify, request
from flask_cors import CORS
from services.recommendation_service import RecommenderNet, load_artifacts, get_collaborative_recs, get_content_based_recs
import torch
import traceback

# --- 1. Load All Artifacts and Models into Memory ---
artifacts = load_artifacts()
collab_artifacts = artifacts['collab']
n_users = len(collab_artifacts['user_to_idx'])
n_movies = len(collab_artifacts['movie_to_idx'])
collab_model = RecommenderNet(n_users, n_movies)
collab_model.load_state_dict(torch.load('model_artifacts/best_model_state_dict.pth', map_location=torch.device('cpu')))
collab_model.eval()

# --- 2. Initialize Flask App ---
app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# --- 3. Define the Main API Endpoint ---
@app.route('/api/recommendations', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        favorite_ids = data.get('favorite_movie_ids', [])
        limit = min(int(request.args.get('limit', 15)), 50)

        if not data or 'favorite_movie_ids' not in data:
            return jsonify({"error": "Request must include 'favorite_movie_ids'"}), 400
            
        
        # --- The Cold Start Logic ---
        COLD_START_THRESHOLD = 5
        
        if len(favorite_ids) < COLD_START_THRESHOLD:
            print("User is in COLD START. Using Content-Based Model.")
            recommendations = get_content_based_recs(favorite_ids, artifacts, limit)
        else:
            print("User is a WARM USER. Using Collaborative Filtering Model.")
            recommendations = get_collaborative_recs(favorite_ids, collab_model, artifacts, limit)
            
        # The recommendation service now adds the titles, so this loop is no longer needed here.

        return jsonify({"recommendations": recommendations})

    except Exception as e:
        # --- Improved Error Logging ---
        print("--- AN ERROR OCCURRED ---")
        print(traceback.format_exc()) # This prints the full error traceback
        print("-------------------------")
        return jsonify({"error": "An internal server error occurred", "details": str(e)}), 500

# --- 4. Run the App ---
if __name__ == '__main__':
    app.run(debug=True, port=5001)

