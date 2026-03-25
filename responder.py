from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL
from emotion_engine import get_persona, detect_crisis

client = Groq(api_key=GROQ_API_KEY)

def build_system_prompt(emotion, persona):
    return f"""You are Raven, an emotionally aware AI assistant.

The user is currently feeling: {emotion}
Your response style should be: {persona}

Core rules:
- Always adapt your tone to match the user's emotional state
- Be genuinely helpful, not just sympathetic
- Keep responses concise but warm
- Never mention that you are detecting emotions
- Just naturally respond in the right tone"""

def get_response(user_message, emotion, conversation_history):
    if detect_crisis(user_message):
        return """I can hear that you're going through something really difficult right now.
You don't have to face this alone. Please reach out to someone who can help:

🆘 iCall (India): 9152987821
🆘 Vandrevala Foundation: 1860-2662-345 (24/7)
🆘 International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/

I'm here to talk if you need me."""

    persona = get_persona(emotion)
    system_prompt = build_system_prompt(emotion, persona)

    messages = [{"role": "system", "content": system_prompt}]
    messages += conversation_history[-10:]
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=500,
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()

def search_and_respond(user_message, emotion, conversation_history):
    try:
        from tavily import TavilyClient
        from config import TAVILY_API_KEY

        study_keywords = [
            "what is", "explain", "how does", "define", "tell me about",
            "who is", "when did", "why does", "difference between"
        ]
        needs_search = any(kw in user_message.lower() for kw in study_keywords)

        if needs_search:
            tavily = TavilyClient(api_key=TAVILY_API_KEY)
            results = tavily.search(query=user_message, max_results=3)
            context = "\n".join([r["content"] for r in results.get("results", [])])

            persona = get_persona(emotion)
            system_prompt = build_system_prompt(emotion, persona)
            system_prompt += f"\n\nWeb search context (use this to answer accurately):\n{context}"

            messages = [{"role": "system", "content": system_prompt}]
            messages += conversation_history[-10:]
            messages.append({"role": "user", "content": user_message})

            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                max_tokens=600,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip(), True

    except Exception:
        pass

    return get_response(user_message, emotion, conversation_history), False