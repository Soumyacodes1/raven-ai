import re
import streamlit as st
from groq import Groq
from config import (
    GROQ_API_KEY, GROQ_MODEL_FAST,
    HF_EMOTION_MODEL, EMOTIONS, EMOTION_PERSONAS, CRISIS_KEYWORDS
)

client = Groq(api_key=GROQ_API_KEY)

# ── Intent config ─────────────────────────────────────────────────────────────
INTENTS = ["venting", "question", "advice", "casual", "crisis", "coding", "learning", "creative", "deep_dive"]

INTENT_PROMPTS = {
    "venting":   "The user is venting and needs to feel heard. Acknowledge their feelings first, then gently offer help if appropriate.",
    "question":  "The user wants a clear, accurate, informative answer. Be thorough and direct.",
    "advice":    "The user wants your opinion or recommendation. Be honest, thoughtful, and give a clear suggestion.",
    "casual":    "The user is just having a casual conversation. Be relaxed, friendly, and natural.",
    "crisis":    "The user may be in distress. Be gentle, supportive, and prioritise their wellbeing.",
    "coding":    "The user wants code or technical help. Write clean, working, well-commented code. Explain your approach. If debugging, identify the root cause first.",
    "learning":  "The user wants to learn something. Explain clearly with examples, analogies, and structure. Build from basics to depth. Be thorough but not boring.",
    "creative":  "The user wants creative output. Be imaginative, original, and expressive. Match the creative energy they bring.",
    "deep_dive": "The user wants a comprehensive, detailed answer. Go deep — cover all angles, provide examples, structure with sections. Do NOT cut short.",
}

# Contradictory emotion+intent combinations — auto corrected
INTENT_CORRECTIONS = {
    "angry":    {"casual": "venting", "coding": "question"},
    "anxious":  {"casual": "venting"},
    "sad":      {"casual": "venting"},
    "happy":    {"crisis": "casual", "venting": "casual"},
    "neutral":  {},
    "confused": {},
}

# ── Crisis detection ──────────────────────────────────────────────────────────
def detect_crisis(text):
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in CRISIS_KEYWORDS)


# ── Emotion/intent reconciliation ─────────────────────────────────────────────
def reconcile_emotion_intent(emotion, intent):
    corrections = INTENT_CORRECTIONS.get(emotion, {})
    return corrections.get(intent, intent)


# ── Emotion smoothing ─────────────────────────────────────────────────────────
def smooth_emotion(new_emotion, emotion_history):
    """Prevent jarring single-message emotion spikes."""
    if len(emotion_history) < 2:
        return new_emotion
    last_two = [e for _, e in emotion_history[-2:]]
    if last_two[0] == last_two[1]:
        prev = last_two[-1]
        jumps = {
            ("happy", "angry"): "confused",
            ("happy", "sad"):   "neutral",
            ("angry", "happy"): "neutral",
            ("sad",   "angry"): "sad",
        }
        smoothed = jumps.get((prev, new_emotion))
        if smoothed:
            return smoothed
    return new_emotion


# ── PRIMARY: HuggingFace emotion classifier ───────────────────────────────────
@st.cache_resource
def load_hf_model():
    """Load HF emotion model once and cache it."""
    from transformers import pipeline
    return pipeline(
        "text-classification",
        model=HF_EMOTION_MODEL,
        top_k=None
    )


def detect_emotion_hf(text):
    """
    PRIMARY emotion classifier.
    Uses our fine-tuned DistilBERT model (77.33% GoEmotions accuracy).
    The model outputs Raven labels directly: happy, sad, anxious, angry, confused, neutral.
    Returns (emotion_string, raw_scores_list).
    """
    try:
        classifier = load_hf_model()
        results = classifier(text)[0]
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        top_label = results[0]["label"].lower()
        emotion = top_label if top_label in EMOTIONS else "neutral"
        return emotion, results
    except Exception:
        return None, []


# ── FALLBACK: Groq emotion classifier ────────────────────────────────────────
def detect_emotion_groq_fallback(text):
    """
    FALLBACK only — used when HF model fails completely.
    Uses GROQ_MODEL_FAST (8B).
    """
    prompt = f"""You are an emotion classifier. Classify the emotional tone of this text into exactly one of: happy, sad, anxious, angry, confused, neutral.

Rules:
- Greetings (hello, hi, hey) are always neutral
- Jokes and "I am kidding" are always happy or neutral
- Questions about topics are confused or neutral, NOT angry
- Only classify as angry if there is clear rage or hostility
- Mild frustration is confused, not angry
- When in doubt, choose neutral

Reply with ONLY the emotion word, nothing else.

Text: "{text}"
Emotion:"""
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL_FAST,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.1,
        )
        result = response.choices[0].message.content.strip().lower()
        result = re.sub(r'[^a-z]', '', result)
        return result if result in EMOTIONS else "neutral"
    except Exception:
        return "neutral"


# ── Intent detection (Groq 8B) ────────────────────────────────────────────────
def detect_intent(text, emotion="neutral", recent_context=""):
    """
    Intent classifier using GROQ_MODEL_FAST (8B).
    Runs on every message alongside emotion detection.
    """
    context_line = f"\nRecent conversation context: {recent_context}" if recent_context else ""
    prompt = f"""You are an intent classifier. Classify the intent of the user message into exactly one of:
- venting: user is expressing strong negative emotion and wants to be heard
- question: user wants information, explanation, or facts
- advice: user wants a recommendation, suggestion, or opinion
- casual: user is making small talk, joking, teasing, or having general conversation
- crisis: user seems to be in serious emotional distress or danger
- coding: user wants code, debugging help, programming concepts, or technical implementation
- learning: user wants to understand a topic deeply — GK, science, history, math, concepts
- creative: user wants stories, poems, writing help, brainstorming, or creative output
- deep_dive: user explicitly asks for a detailed, extensive, or comprehensive explanation

Rules:
- Jokes, teasing, and playful messages are ALWAYS casual
- Short messages like "hello", "ok", "thanks", "lol", "haha" are ALWAYS casual
- Only classify as venting if the emotion is clearly negative AND the user wants to express it
- If the user asks "write code", "debug this", "how to code X" → coding
- If the user asks about facts, concepts, or "explain X" → learning
- If the user asks for a story, poem, or creative piece → creative
- If the user says "explain in detail", "tell me everything about" → deep_dive
{context_line}
Current detected emotion: {emotion}

Reply with ONLY the intent word, nothing else.

Message: "{text}"
Intent:"""
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL_FAST,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )
        result = response.choices[0].message.content.strip().lower()
        result = ''.join(c for c in result if c.isalpha() or c == '_')
        intent = result if result in INTENTS else "casual"
        return reconcile_emotion_intent(emotion, intent)
    except Exception:
        return "casual"


# ── Main detection pipeline ───────────────────────────────────────────────────
def detect_emotion(text, recent_context="", emotion_history=None):
    """
    Full emotion detection pipeline:
    1. Try fine-tuned DistilBERT (primary — 77.33% accuracy)
    2. Fall back to Groq 8B if model fails
    3. Apply smoothing to prevent jarring jumps
    """
    hf_emotion, hf_scores = detect_emotion_hf(text)

    if hf_emotion:
        final_emotion = hf_emotion
    else:
        final_emotion = detect_emotion_groq_fallback(text)

    if emotion_history:
        final_emotion = smooth_emotion(final_emotion, emotion_history)

    return final_emotion, hf_emotion or final_emotion


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_persona(emotion):
    return EMOTION_PERSONAS.get(emotion, EMOTION_PERSONAS["neutral"])


def get_intent_prompt(intent):
    return INTENT_PROMPTS.get(intent, INTENT_PROMPTS["casual"])
