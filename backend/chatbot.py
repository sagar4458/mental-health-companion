"""
Solace — Mental Health Companion
Version 1 (November 2025)

Sentiment analysis: VADER (compound score -1 to +1)
Mood detection: scored keyword matching
LLM: Google Gemini 1.5 Flash (1500 req/day free tier)
"""

import os
try:
    from google import genai
    from google.genai import types
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
from datetime import datetime
from typing import List, Dict, Any

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

# Google Gemini client
gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY")) if GOOGLE_AVAILABLE else None
MODEL_NAME    = "gemini-1.5-flash"

# Groq client (free alternative — 14400 req/day)
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY")) if GROQ_AVAILABLE and os.environ.get("GROQ_API_KEY") else None
MAX_HISTORY = 20

CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "self harm", "self-harm",
    "want to die", "cant go on", "can't go on", "no reason to live",
    "hurt myself", "harm myself", "not worth living", "better off dead",
    "ending it all", "take my own life", "end it all",
]

MOOD_PROFILES = {
    "anxious": {
        "keywords": ["anxious","anxiety","worried","panic","nervous","stressed",
                     "overwhelmed","fear","scared","dread","tense","restless",
                     "uneasy","apprehensive","freaking out","cant breathe"],
        "tone": "calm, grounding and steady",
        "intensity_modifier": {
            "high":   "The person is severely anxious. Lead with breathing guidance immediately before anything else.",
            "medium": "The person is moderately anxious. Validate first then gently ground them.",
            "low":    "The person is mildly anxious. Validate and explore what is behind the worry.",
        },
        "strategies": [
            "Try box breathing — breathe in for 4 counts, hold for 4, out for 4, hold for 4.",
            "Ground yourself: name 5 things you can see right now.",
            "Place one hand on your chest and notice your heartbeat slowing as you breathe.",
            "Write your specific worry down — getting it out of your head reduces its power.",
        ],
    },
    "sad": {
        "keywords": ["sad","depressed","hopeless","crying","grief","lonely","empty",
                     "numb","heartbroken","miserable","down","blue","devastated",
                     "worthless","falling apart","broken","lost","exhausted"],
        "tone": "deeply warm, gentle and non-rushing",
        "intensity_modifier": {
            "high":   "The person is deeply sad or depressed. Be very slow, very gentle. No advice yet — just presence.",
            "medium": "The person is sad. Validate fully before gently exploring what happened.",
            "low":    "The person is feeling low. Acknowledge and explore with curiosity.",
        },
        "strategies": [
            "Allow this feeling — sadness needs space, not suppression.",
            "Reach out to one person today, even just a short message.",
            "Do one tiny act of care for yourself — warmth, rest, or something comforting.",
            "Write what you are feeling — no structure needed, just let it out.",
        ],
    },
    "angry": {
        "keywords": ["angry","frustrated","furious","mad","irritated","rage","livid",
                     "annoyed","resentful","bitter","hate","unfair","fed up","sick of"],
        "tone": "steady, non-judgmental and grounding",
        "intensity_modifier": {
            "high":   "The person is very angry. Acknowledge the anger fully first — do not try to calm or reframe yet.",
            "medium": "The person is frustrated. Validate the feeling and explore what triggered it.",
            "low":    "The person is mildly annoyed. Acknowledge and gently explore.",
        },
        "strategies": [
            "Physical movement discharges anger — even shaking your hands helps.",
            "Write an unsent letter expressing everything — no filter.",
            "Anger is often a signal a boundary was crossed. What was it?",
            "Take 20 minutes before responding to the situation.",
        ],
    },
    "positive": {
        "keywords": ["happy","good","great","excited","grateful","better","hopeful",
                     "proud","content","peaceful","joyful","relieved","calm","motivated",
                     "amazing","wonderful","fantastic","thrilled"],
        "tone": "warm, celebratory and genuinely delighted for them",
        "intensity_modifier": {
            "high":   "The person is very happy. Match their energy warmly and help them savour it.",
            "medium": "The person is feeling good. Acknowledge and build on it.",
            "low":    "The person is okay. Gently explore what is going well.",
        },
        "strategies": [
            "Savour this moment — take a breath and really feel it.",
            "Share this feeling with someone you care about.",
            "Note what created this feeling — it is worth remembering.",
        ],
    },
    "neutral": {
        "keywords": ["okay","fine","alright","normal","whatever","not sure",
                     "don't know","unsure","confused","meh","so so","average"],
        "tone": "curious, warm and gently exploratory",
        "intensity_modifier": {
            "high":   "The person seems to be masking something. Gently invite them to go deeper.",
            "medium": "The person seems uncertain. Create space for them to explore.",
            "low":    "The person seems okay. Check in gently.",
        },
        "strategies": [
            "Sometimes fine is the beginning of something deeper. What is really on your mind?",
            "Check in with your body — where are you holding tension right now?",
        ],
    },
}

SYSTEM_PROMPT = """You are Solace — a compassionate AI companion trained in principles of Cognitive Behavioural Therapy (CBT), mindfulness-based stress reduction, and person-centred counselling.

Your name is Solace. You are warm, patient and deeply present. You are not a chatbot — you are a companion who genuinely cares.

## Core Principles

LISTEN BEFORE YOU SPEAK.
Never jump to solutions or advice. Always acknowledge the feeling first. The person needs to feel heard before they can receive anything.

REFLECT WITH PRECISION.
Mirror back what you hear with specificity: not "that sounds hard" but "it sounds like the weight of these exams has been building for a while and it is finally catching up with you." Show you truly listened.

ONE QUESTION ONLY.
Never ask more than one question. Choose the most important one. Multiple questions overwhelm people who are already struggling.

SPEAK LIKE A WARM HUMAN.
No clinical language. No therapy jargon. Speak the way a wise, caring friend would — someone who happens to understand emotions deeply.

REMEMBER EVERYTHING.
Reference what the person shared earlier. If they mentioned their mother, remember their mother. If they said they have not slept, ask about their sleep. Show continuity of care.

MATCH THEIR ENERGY.
If they write short sentences, respond shorter. If they write paragraphs, you can go deeper. Mirror their pace.

## Response Structure

1. ACKNOWLEDGE — Name and validate the emotion with specificity (1-2 sentences)
2. REFLECT — Show you heard the deeper meaning beneath the words (1 sentence)  
3. EXPLORE or SUPPORT — One focused question OR one gentle insight (1-2 sentences)
4. COPING (only if natural) — One strategy, only when the person is ready

## Current Emotional Context

Mood detected: {mood}
Emotional intensity: {intensity} ({intensity_score:.2f} out of 1.0)
Tone guidance: {tone}
Intensity guidance: {intensity_modifier}
Emotional arc this session: {emotional_arc}

## Boundaries

- Never diagnose or prescribe
- Never claim to replace a therapist
- Always encourage professional help for persistent symptoms
- Maximum 4-5 sentences unless they clearly need more
- Write naturally — no bullet points, no headers in responses
- Do not mention you are an AI unless directly asked"""


def get_sentiment(text: str) -> Dict:
    if VADER_AVAILABLE:
        scores = vader.polarity_scores(text)
        compound = scores["compound"]
    else:
        compound = 0.0

    if compound <= -0.6:
        intensity = "high"
        intensity_score = abs(compound)
    elif compound <= -0.2:
        intensity = "medium"
        intensity_score = abs(compound)
    elif compound >= 0.5:
        intensity = "high"
        intensity_score = compound
    elif compound >= 0.2:
        intensity = "medium"
        intensity_score = compound
    else:
        intensity = "low"
        intensity_score = abs(compound)

    return {
        "compound"       : round(compound, 3),
        "intensity"      : intensity,
        "intensity_score": round(intensity_score, 3),
    }


def is_crisis(text: str) -> bool:
    t = text.lower()
    if any(kw in t for kw in CRISIS_KEYWORDS):
        return True
    if VADER_AVAILABLE:
        score = vader.polarity_scores(text)["compound"]
        if score < -0.85:
            return True
    return False


def crisis_response() -> str:
    return (
        "I hear you. What you are feeling right now is real, and you do not have to face it alone.\n\n"
        "Please reach out to someone who can be with you right now:\n\n"
        "📞 iCall (India): 9152987821\n"
        "📞 Vandrevala Foundation: 1860-2662-345 — free, 24/7\n"
        "📞 AASRA: 9820466627 — 24/7\n"
        "🌐 International: befrienders.org\n\n"
        "If you are in immediate danger, please go to your nearest hospital or call emergency services.\n\n"
        "I am here with you right now. Can you tell me — is there one person near you you could reach?"
    )


EMOTION_KEYWORDS = {
    "anxious"  : ["anxious","anxiety","worried","panic","nervous","stressed","overwhelmed","scared","dread","tense","restless"],
    "sad"      : ["sad","depressed","hopeless","crying","grief","lonely","empty","numb","heartbroken","devastated","worthless","down"],
    "angry"    : ["angry","frustrated","furious","mad","irritated","rage","livid","annoyed","resentful","bitter"],
    "positive" : ["happy","good","great","excited","grateful","better","hopeful","proud","content","joyful","amazing","wonderful"],
    "neutral"  : ["okay","fine","alright","normal","whatever","not sure","unsure","confused","meh"],
}

WEATHER_MAP = {
    "crisis"  : "storm",
    "sad"     : "rain",
    "anxious" : "lightning",
    "angry"   : "storm",
    "positive": "sunny",
    "neutral" : "calm_night",
}

def compute_emotion_scores(text: str, primary_mood: str) -> Dict[str, int]:
    t = text.lower()
    raw = {}
    for mood, keywords in EMOTION_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in t)
        if score > 0:
            raw[mood] = score
    if not raw:
        raw[primary_mood] = 1
    total = sum(raw.values())
    return {k: round(v / total * 100) for k, v in sorted(raw.items(), key=lambda x: -x[1])}


def detect_mood(text: str, sentiment: Dict) -> str:
    t = text.lower()
    scores = {}
    for mood, profile in MOOD_PROFILES.items():
        score = sum(1 for kw in profile["keywords"] if kw in t)
        if score > 0:
            scores[mood] = score

    if scores:
        return max(scores, key=scores.get)

    compound = sentiment["compound"]
    if compound <= -0.3:
        return "sad"
    elif compound >= 0.3:
        return "positive"
    return "neutral"


class Conversation:
    def __init__(self):
        self.history       : List[Dict] = []
        self.mood_log      : List[Dict] = []
        self.mood_arc      : List[str]  = []
        self.sentiment_arc : List[float]= []
        self.started_at    : str = datetime.now().isoformat()
        self.current_mood  : str = "neutral"
        self.current_sentiment: Dict = {"compound": 0.0, "intensity": "low", "intensity_score": 0.0}

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if len(self.history) > MAX_HISTORY * 2:
            self.history = self.history[-MAX_HISTORY * 2:]

    def log_mood(self, mood: str, sentiment: Dict):
        self.current_mood      = mood
        self.current_sentiment = sentiment
        self.mood_arc.append(mood)
        self.sentiment_arc.append(sentiment["compound"])
        self.mood_log.append({
            "mood"     : mood,
            "sentiment": sentiment,
            "timestamp": datetime.now().isoformat(),
        })

    def get_emotional_arc(self) -> str:
        if not self.mood_arc:
            return "Session just beginning."
        if len(self.mood_arc) == 1:
            return f"Session just started. Person is feeling {self.mood_arc[0]}."
        unique = list(dict.fromkeys(self.mood_arc))
        if len(unique) == 1:
            return f"Consistent {unique[0]} throughout the session."
        arc = " → ".join(unique)
        if self.sentiment_arc[-1] > self.sentiment_arc[0]:
            return f"Emotional journey: {arc}. The person seems to be moving in a more positive direction — acknowledge this."
        elif self.sentiment_arc[-1] < self.sentiment_arc[0]:
            return f"Emotional journey: {arc}. The person's mood has shifted lower — be especially gentle."
        return f"Emotional journey: {arc}."

    def build_prompt(self, user_message: str, mood: str, sentiment: Dict) -> str:
        profile  = MOOD_PROFILES.get(mood, MOOD_PROFILES["neutral"])
        intensity= sentiment["intensity"]
        modifier = profile["intensity_modifier"].get(intensity, "")

        system = SYSTEM_PROMPT.format(
            mood             = mood,
            intensity        = intensity,
            intensity_score  = sentiment["intensity_score"],
            tone             = profile["tone"],
            intensity_modifier = modifier,
            emotional_arc    = self.get_emotional_arc(),
        )

        parts = [system, "\n## Conversation\n"]
        for msg in self.history[-MAX_HISTORY:]:
            role = "Person" if msg["role"] == "user" else "Solace"
            parts.append(f"{role}: {msg['content']}")
        parts.append(f"\nPerson: {user_message}")
        parts.append("\nSolace:")
        return "\n".join(parts)

    def to_dict(self) -> Dict:
        return {
            "started_at"       : self.started_at,
            "history"          : self.history,
            "mood_log"         : self.mood_log,
            "current_mood"     : self.current_mood,
            "current_sentiment": self.current_sentiment,
        }


def generate_response(conversation: Conversation, user_message: str) -> Dict[str, Any]:
    if is_crisis(user_message):
        response_text = crisis_response()
        conversation.add_message("user",      user_message)
        conversation.add_message("assistant", response_text)
        conversation.log_mood("crisis", {"compound": -1.0, "intensity": "high", "intensity_score": 1.0})
        return {
            "response"        : response_text,
            "mood"            : "crisis",
            "is_crisis"       : True,
            "sentiment"       : {"compound": -1.0, "intensity": "high", "intensity_score": 1.0},
        }

    sentiment = get_sentiment(user_message)
    mood      = detect_mood(user_message, sentiment)
    conversation.log_mood(mood, sentiment)
    conversation.add_message("user", user_message)

    prompt = conversation.build_prompt(user_message, mood, sentiment)

    import time
    response_text = None
    last_error    = ""

    # ── 1. Try Groq first — 14,400 req/day free, completely separate from Google ──
    if groq_client:
        try:
            groq_resp = groq_client.chat.completions.create(
                model       = "llama-3.1-8b-instant",
                messages    = [{"role": "user", "content": prompt}],
                temperature = 0.78,
                max_tokens  = 450,
            )
            response_text = groq_resp.choices[0].message.content.strip()
        except Exception as ge:
            last_error = str(ge)

    # ── 2. Fallback: try all Google Gemini models ──────────────────────────────
    if not response_text and gemini_client:
        for model in ["gemini-2.0-flash-lite","gemini-2.0-flash","gemini-1.5-flash-8b","gemini-1.5-flash","gemini-1.0-pro"]:
            try:
                resp = gemini_client.models.generate_content(
                    model   = model,
                    contents= prompt,
                    config  = types.GenerateContentConfig(
                        temperature=0.78, max_output_tokens=450, top_p=0.92),
                )
                response_text = resp.text.strip()
                break
            except Exception as me:
                last_error = str(me)
                if "429" in last_error or "quota" in last_error.lower():
                    time.sleep(1); continue
                break

    if not response_text:
        response_text = "I need a brief moment. Please try again in 30 seconds — I'm still here with you."

    conversation.add_message("assistant", response_text)

    emotion_scores = compute_emotion_scores(user_message, mood)
    weather_type   = WEATHER_MAP.get(mood, "calm_night")

    return {
        "response"      : response_text,
        "mood"          : mood,
        "is_crisis"     : False,
        "sentiment"     : sentiment,
        "emotion_scores": emotion_scores,
        "weather_type"  : weather_type,
        "stress_level"  : max(0, min(100, round(((-sentiment.get("compound", 0) + 1) / 2) * 100))),
        "energy_level"  : max(0, min(100, round(((sentiment.get("compound", 0) + 1) / 2) * 100))),
    }