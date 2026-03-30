from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL, GROQ_MODEL_FAST
from emotion_engine import get_persona, detect_crisis, detect_intent, get_intent_prompt

client = Groq(api_key=GROQ_API_KEY)


def clean_history(conversation_history):
    return [
        {"role": m["role"], "content": m["content"]}
        for m in conversation_history
        if m.get("role") in ["user", "assistant"] and m.get("content")
    ]


INTENT_TEMPERATURES = {
    "coding":    0.5,
    "learning":  0.5,
    "deep_dive": 0.6,
    "question":  0.65,
    "creative":  0.85,
    "venting":   0.75,
    "advice":    0.7,
    "casual":    0.75,
    "crisis":    0.6,
}


def get_temperature(intent):
    return INTENT_TEMPERATURES.get(intent, 0.75)


def build_system_prompt(emotion, persona, intent):
    intent_instruction = get_intent_prompt(intent)
    return f"""You are Raven — a brilliant, knowledgeable, and genuinely warm AI assistant. You care about the people you talk to, and it shows naturally in how you respond.

Your primary job is to be deeply useful — give accurate answers, explain things clearly, help solve problems, write code, do calculations, and provide real value. You do this with warmth and personality, never in a cold or robotic way.

The user appears to be feeling: {emotion}.
Subtly shape your delivery to be: {persona}.

The user's intent is: {intent}.
How to handle this: {intent_instruction}

Core principles:
- Be knowledgeable and thorough — but never dry or textbook-like. Add a human touch.
- Be warm and friendly — natural warmth, not performative cheerfulness.
- Match your energy to the user's emotion and intent.
- Never water down an answer because of someone's emotional state.
- Be honest and direct — say what you mean.
- Use natural conversational language — like a brilliant friend who happens to know a lot.
- For calculations, always show your working step by step.
- For code, always write clean, working, well-commented code. Use proper language-tagged code blocks (```python, ```javascript, etc.). If the problem is complex, break it into steps.
- For GK, science, history, or conceptual questions, be comprehensive and accurate. Use structured format (headings, bullet points) for long answers. Include interesting context and connections.
- For creative writing, match the user's creative vision. Be vivid, original, and expressive.
- For deep dives, be exhaustive. Use sections, examples, and analogies. Never truncate — complete the full answer.
- If the user asks for something long or detailed, ALWAYS deliver the full content. Never say "I'll keep it brief" unless the user asks for brevity.
- For long detailed requests, always complete the full answer — never stop halfway.
- The emotional awareness shapes how you say things, never what you say."""


def get_response(user_message, emotion, conversation_history):
    if detect_crisis(user_message):
        return """I can hear that you're going through something really difficult right now.
You don't have to face this alone. Please reach out to someone who can help:

\U0001f198 iCall (India): 9152987821
\U0001f198 Vandrevala Foundation: 1860-2662-345 (24/7)
\U0001f198 International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/

I'm here to talk if you need me."""

    intent = detect_intent(user_message, emotion)
    persona = get_persona(emotion)
    system_prompt = build_system_prompt(emotion, persona, intent)
    temp = get_temperature(intent)

    messages = [{"role": "system", "content": system_prompt}]
    messages += clean_history(conversation_history[-20:])
    messages.append({"role": "user", "content": user_message})

    # Try 70B first, fall back to 8B automatically if limit hit
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=4096,
            temperature=temp,
        )
    except Exception:
        response = client.chat.completions.create(
            model=GROQ_MODEL_FAST,
            messages=messages,
            max_tokens=4096,
            temperature=temp,
        )

    return response.choices[0].message.content.strip()


def search_and_respond(user_message, emotion, conversation_history):
    response = get_response(user_message, emotion, conversation_history)
    return response, False
