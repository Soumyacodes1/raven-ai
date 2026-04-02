import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_API_KEY   = os.getenv("HF_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ── Models ────────────────────────────────────────────────────────────────────
GROQ_MODEL        = "llama-3.3-70b-versatile"    # response generation (primary)
GROQ_MODEL_FAST   = "llama-3.1-8b-instant"       # intent detection + emotion fallback
GROQ_MODEL_EVAL   = "llama-3.3-70b-versatile"    # evaluation experiments only
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  # image understanding (FREE, replaces decommissioned 90b-vision)
GROQ_WHISPER_MODEL = "whisper-large-v3-turbo"     # voice transcription (FREE)

HF_EMOTION_MODEL = "SoumyaCodes/raven-emotion-distilbert"  # Fine-tuned DistilBERT — 97.62%

# ── Emotions & personas ───────────────────────────────────────────────────────
EMOTIONS = ["happy", "sad", "anxious", "angry", "confused", "neutral"]

EMOTION_PERSONAS = {
    "happy":    "upbeat, enthusiastic, and engaging",
    "sad":      "warm, gentle, and compassionate",
    "anxious":  "calm, slow-paced, and reassuring",
    "angry":    "patient, non-confrontational, and understanding",
    "confused": "clear, simple, and step-by-step",
    "neutral":  "friendly, balanced, and helpful",
}

# ── Crisis keywords ───────────────────────────────────────────────────────────
CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "want to die",
    "wish i was dead", "better off dead", "end it all",
    "want it all to stop", "no point in anything",
    "no reason to live", "no point living",
    "don't want to be here", "dont want to be here",
    "want to disappear forever", "give up on life",
    "can't go on", "cannot go on", "cant go on",
    "self harm", "self-harm", "hurt myself", "cut myself",
    "harm myself", "injure myself", "punish myself",
    "no hope", "completely hopeless", "nothing to live for",
    "life is pointless", "life is meaningless",
    "nobody cares about me", "everyone would be better without me",
    "i am a burden", "i'm a burden", "im a burden",
    "world is better without me", "tired of living",
    "don't want to exist", "dont want to exist",
    "wish i never existed", "wish i wasnt born",
    "final goodbye", "last goodbye", "saying goodbye forever",
    "won't be here anymore", "wont be here anymore",
    "ending things", "ending everything",
    "can't take it anymore", "cant take it anymore",
    "reached my limit", "too much pain",
    "dont feel like living", "don't feel like living",
]
