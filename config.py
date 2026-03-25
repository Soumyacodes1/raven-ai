import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_MODEL_FAST = "llama-3.1-8b-instant"
EMOTIONS = ["happy", "sad", "anxious", "angry", "confused", "neutral"]

EMOTION_PERSONAS = {
    "happy":    "upbeat, enthusiastic, and engaging",
    "sad":      "warm, gentle, and compassionate",
    "anxious":  "calm, slow-paced, and reassuring",
    "angry":    "patient, non-confrontational, and understanding",
    "confused": "clear, simple, and step-by-step",
    "neutral":  "friendly, balanced, and helpful",
}

CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "want to die",
    "self harm", "hurt myself", "no reason to live"
]