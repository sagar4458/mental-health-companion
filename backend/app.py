"""
Solace — Mental Health Companion
Flask Backend — Version 1 (November 2025)
Run: python backend/app.py
Dashboard: http://localhost:5002
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

import uuid
from flask import Flask, render_template, request, jsonify

from chatbot import Conversation, generate_response

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'frontend'),
    static_folder  =os.path.join(BASE_DIR, 'frontend'),
)
app.secret_key = os.urandom(24)

# In-memory session store
conversations: dict = {}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/new_session", methods=["POST"])
def new_session():
    session_id = str(uuid.uuid4())
    conversations[session_id] = Conversation()
    return jsonify({"session_id": session_id})


@app.route("/api/chat", methods=["POST"])
def chat():
    data       = request.get_json()
    session_id = data.get("session_id")
    message    = (data.get("message") or "").strip()

    if not session_id or session_id not in conversations:
        return jsonify({"error": "Invalid session"}), 400
    if not message:
        return jsonify({"error": "Empty message"}), 400

    conv   = conversations[session_id]
    result = generate_response(conv, message)

    # Return ALL fields the frontend needs
    return jsonify({
        "response"      : result["response"],
        "mood"          : result["mood"],
        "is_crisis"     : result["is_crisis"],
        "sentiment"     : result.get("sentiment", {
            "compound"       : 0.0,
            "intensity"      : "low",
            "intensity_score": 0.0,
        }),
        "emotion_scores": result.get("emotion_scores", {}),
        "stress_level"  : result.get("stress_level", 50),
        "energy_level"  : result.get("energy_level", 50),
        "weather_type"  : result.get("weather_type", "calm_night"),
    })


@app.route("/api/mood_log", methods=["GET"])
def mood_log():
    session_id = request.args.get("session_id")
    if not session_id or session_id not in conversations:
        return jsonify({"mood_log": []})
    conv = conversations[session_id]
    return jsonify({
        "mood_log"       : conv.mood_log,
        "current_mood"   : conv.current_mood,
        "current_sentiment": conv.current_sentiment,
    })


@app.route("/api/session_summary", methods=["GET"])
def session_summary():
    session_id = request.args.get("session_id")
    if not session_id or session_id not in conversations:
        return jsonify({"error": "Invalid session"}), 400
    conv = conversations[session_id]
    return jsonify(conv.to_dict())


if __name__ == "__main__":
    print("Solace is listening at http://localhost:5002")
    app.run(debug=False, host="0.0.0.0", port=5002, threaded=True)