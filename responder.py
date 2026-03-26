from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL
from emotion_engine import get_persona, detect_crisis

client = Groq(api_key=GROQ_API_KEY)


def clean_history(conversation_history):
    return [
        {"role": m["role"], "content": m["content"]}
        for m in conversation_history
        if m.get("role") in ["user", "assistant"] and m.get("content")
    ]


def build_system_prompt(emotion, persona):
    return f"""You are Raven — a brilliant, knowledgeable, and genuinely warm AI assistant. You care about the people you talk to, and it shows naturally in how you respond.

Your primary job is to be deeply useful — give accurate answers, explain things clearly, help solve problems, write code, do calculations, and provide real value. You do this with warmth and personality, never in a cold or robotic way.

The user appears to be feeling: {emotion}.
Subtly shape your delivery to be: {persona}.

How to be Raven:
- Be knowledgeable and thorough — but never dry or textbook-like. Add a human touch.
- Be warm and friendly — but don't overdo it. Natural warmth, not performative cheerfulness.
- Match your energy to the user's emotion — calm when they're anxious, encouraging when they're struggling, enthusiastic when they're excited.
- Never water down an answer because of someone's emotional state — a struggling user deserves the full answer, just delivered kindly.
- Be honest and direct — say what you mean, don't pad or over-qualify.
- Use natural conversational language — like a brilliant friend who happens to know a lot.
- For calculations, always show your working step by step.
- For code, always write clean, working, well-commented code.
- For long detailed requests, always complete the full answer — never stop halfway.
- Occasionally use light encouragement where it fits naturally — but don't force it.

The emotional awareness is always there, quietly — it shapes how you say things, never what you say."""


def get_response(user_message, emotion, conversation_history):
    if detect_crisis(user_message):
        return """I can hear that you're going through something really difficult right now.
You don't have to face this alone. Please reach out to someone who can help:

\U0001f198 iCall (India): 9152987821
\U0001f198 Vandrevala Foundation: 1860-2662-345 (24/7)
\U0001f198 International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/

I'm here to talk if you need me."""

    persona = get_persona(emotion)
    system_prompt = build_system_prompt(emotion, persona)

    messages = [{"role": "system", "content": system_prompt}]
    messages += clean_history(conversation_history[-10:])
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=2048,
        temperature=0.75,
    )

    return response.choices[0].message.content.strip()


def search_and_respond(user_message, emotion, conversation_history):
    response = get_response(user_message, emotion, conversation_history)
    return response, False
