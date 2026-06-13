"""
Solace — Mental Health Companion
Flask Backend — Version 1.1 (June 2026)
Run: python backend/app.py
Dashboard: http://localhost:5002
"""

import sys
import os

# ────────────────────────────────────────────────────────────
# SETUP PATHS & ENVIRONMENT
# ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env file if it exists
from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    print("⚠ No .env file found (OK on production with Render env vars)")

import uuid
from flask import Flask, render_template, request, jsonify
from chatbot import Conversation, generate_response

# ────────────────────────────────────────────────────────────
# FLASK APP SETUP
# ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'frontend'),
    static_folder=os.path.join(BASE_DIR, 'frontend'),
)

app.secret_key = os.urandom(24)

# In-memory session store (resets on restart)
conversations: dict = {}

# ────────────────────────────────────────────────────────────
# ROUTES
# ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main UI"""
    try:
        return render_template("index.html")
    except Exception as e:
        return jsonify({"error": f"Failed to load UI: {str(e)}"}), 500

@app.route("/api/new_session", methods=["POST"])
def new_session():
    """Create a new conversation session"""
    try:
        session_id = str(uuid.uuid4())
        conversations[session_id] = Conversation()
        print(f"✓ New session created: {session_id}")
        return jsonify({"session_id": session_id}), 200
    except Exception as e:
        print(f"✗ Error creating session: {e}")
        return jsonify({"error": "Failed to create session"}), 500

@app.route("/api/chat", methods=["POST"])
def chat():
    """Process user message and return AI response"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        session_id = data.get("session_id", "").strip()
        message = (data.get("message") or "").strip()
        
        # Validate session
        if not session_id or session_id not in conversations:
            return jsonify({"error": "Invalid or expired session"}), 400
        
        # Validate message
        if not message:
            return jsonify({"error": "Empty message"}), 400
        
        # Get conversation and generate response
        conv = conversations[session_id]
        result = generate_response(conv, message)
        
        # Return full response data
        return jsonify({
            "response"       : result.get("response", ""),
            "mood"           : result.get("mood", "neutral"),
            "is_crisis"      : result.get("is_crisis", False),
            "sentiment"      : result.get("sentiment", {
                "compound"       : 0.0,
                "intensity"      : "low",
                "intensity_score": 0.0,
            }),
            "emotion_scores" : result.get("emotion_scores", {}),
            "stress_level"   : result.get("stress_level", 50),
            "energy_level"   : result.get("energy_level", 50),
            "weather_type"   : result.get("weather_type", "calm_night"),
        }), 200
        
    except Exception as e:
        print(f"✗ Error in /api/chat: {e}")
        return jsonify({"error": f"Chat processing failed: {str(e)}"}), 500

@app.route("/api/mood_log", methods=["GET"])
def mood_log():
    """Get mood history for a session"""
    try:
        session_id = request.args.get("session_id", "").strip()
        
        if not session_id or session_id not in conversations:
            return jsonify({
                "mood_log"        : [],
                "current_mood"    : "neutral",
                "current_sentiment": {"compound": 0.0, "intensity": "low", "intensity_score": 0.0},
            }), 200
        
        conv = conversations[session_id]
        return jsonify({
            "mood_log"        : conv.mood_log,
            "current_mood"    : conv.current_mood,
            "current_sentiment": conv.current_sentiment,
        }), 200
        
    except Exception as e:
        print(f"✗ Error in /api/mood_log: {e}")
        return jsonify({"mood_log": [], "error": str(e)}), 500

@app.route("/api/session_summary", methods=["GET"])
def session_summary():
    """Get full conversation summary"""
    try:
        session_id = request.args.get("session_id", "").strip()
        
        if not session_id or session_id not in conversations:
            return jsonify({"error": "Invalid or expired session"}), 400
        
        conv = conversations[session_id]
        return jsonify(conv.to_dict()), 200
        
    except Exception as e:
        print(f"✗ Error in /api/session_summary: {e}")
        return jsonify({"error": str(e)}), 500

# ────────────────────────────────────────────────────────────
# ERROR HANDLERS
# ────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    print(f"✗ Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500

# ────────────────────────────────────────────────────────────
# RUN THE APP
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Solace — Mental Health Companion")
    print("=" * 60)
    print("Listening at http://localhost:5002")
    print("=" * 60)
    
    # Run Flask
    app.run(
        debug=False,
        host="0.0.0.0",
        port=5002,
        threaded=True
    )
