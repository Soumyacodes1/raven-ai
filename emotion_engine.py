import re
import streamlit as st
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL_FAST, EMOTIONS, EMOTION_PERSONAS, CRISIS_KEYWORDS

client = Groq(api_key=GROQ_API_KEY)


def detect_crisis(text):
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in CRISIS_KEYWORDS)


def detect_emotion_groq(text):
    prompt = f"""You are an emotion classifier. Analyze the emotional tone of the text below and classify it into exactly one of these emotions: {', '.join(EMOTIONS)}.

Rules:
- Reply with ONLY the emotion word, nothing else
- Choose the most dominant emotion
- If unsure, choose neutral

Text: "{text}"

Emotion:"""

    response = client.chat.completions.create(
        model=GROQ_MODEL_FAST,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0.1,
    )

    result = response.choices[0].message.content.strip().lower()
    result = re.sub(r'[^a-z]', '', result)
    return result if result in EMOTIONS else "neutral"


@st.cache_resource
def load_hf_model():
    from transformers import pipeline
    return pipeline(
        "text-classification",
        model="bhadresh-savani/distilbert-base-uncased-emotion",
        top_k=1
    )


def detect_emotion_hf(text):
    try:
        classifier = load_hf_model()
        hf_to_raven = {
            "joy": "happy", "sadness": "sad", "fear": "anxious",
            "anger": "angry", "surprise": "confused", "love": "happy"
        }
        result = classifier(text)[0][0]
        label = result["label"].lower()
        return hf_to_raven.get(label, "neutral")
    except Exception:
        return None


def detect_emotion(text):
    groq_emotion = detect_emotion_groq(text)
    hf_emotion = detect_emotion_hf(text)

    if hf_emotion and hf_emotion != groq_emotion:
        priority = {"anxious": 2, "sad": 2, "angry": 2,
                    "confused": 1, "happy": 1, "neutral": 0}
        if priority.get(hf_emotion, 0) >= priority.get(groq_emotion, 0):
            return hf_emotion, groq_emotion

    return groq_emotion, hf_emotion or groq_emotion


def get_persona(emotion):
    return EMOTION_PERSONAS.get(emotion, EMOTION_PERSONAS["neutral"])
