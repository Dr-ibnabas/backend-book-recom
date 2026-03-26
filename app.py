from flask import Flask, request, jsonify
from flask_cors import CORS
import hashlib
from recommendation_engine import RecommendationEngine
import os

app = Flask(__name__)
CORS(app) # Allow React frontend to connect

# Paths relative to the app.py
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BOOKS_PATH = os.path.join(ROOT_DIR, "books.csv")
RATINGS_PATH = os.path.join(ROOT_DIR, "ratings.csv")

# Initialize the engine
engine = RecommendationEngine(BOOKS_PATH, RATINGS_PATH)

def hash_uid_to_int(uid):
    """Map Firebase UID (string) to a stable integer for the recommendation matrix."""
    # Convert string UID to a large integer hash
    hash_obj = hashlib.md5(uid.encode())
    return int(hash_obj.hexdigest(), 16) % 1000000

@app.route('/recommend/user/<string:uid>', methods=['GET'])
def get_user_recommendations(uid):
    user_id = hash_uid_to_int(uid)
    # Ensure user exists in matrix (at least with zeros)
    if user_id not in engine.user_book_matrix.index:
        engine.add_rating(user_id, 0, 0) # Placeholder to initialize user
        
    recs = engine.recommend_user_based(user_id)
    return jsonify({"recs": recs})

@app.route('/recommend/item/<string:uid>', methods=['GET'])
def get_item_recommendations(uid):
    user_id = hash_uid_to_int(uid)
    recs = engine.recommend_item_based(user_id)
    return jsonify({"recs": recs})

@app.route('/rate', methods=['POST'])
def add_rating():
    data = request.json
    uid = data.get('uid')
    book_id = data.get('book_id')
    rating = data.get('rating')
    
    if not uid or not book_id or not rating:
        return jsonify({"error": "Missing parameters"}), 400
        
    user_id = hash_uid_to_int(uid)
    success = engine.add_rating(user_id, int(book_id), float(rating))
    
    if success:
        return jsonify({"message": "Rating added successfully"})
    return jsonify({"error": "Failed to add rating"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
