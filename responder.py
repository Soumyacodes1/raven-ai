from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL, TAVILY_API_KEY
from emotion_engine import get_persona, detect_crisis

client = Groq(api_key=GROQ_API_KEY)

STUDY_KEYWORDS = [
    "what is", "what are", "what was", "what were",
    "what does", "what do", "what did", "what will",
    "how does", "how do", "how did", "how is", "how are",
    "how to", "how can", "how should",
    "why does", "why do", "why did", "why is", "why are",
    "who is", "who was", "who are", "who were", "who invented",
    "when did", "when was", "when is", "when were",
    "where is", "where did", "where does",
    "which is", "which are", "which was",
    "explain", "define", "definition of",
    "tell me about", "describe", "elaborate",
    "summarise", "summarize", "overview of",
    "introduction to", "basics of", "fundamentals of",
    "meaning of", "concept of",
    "difference between", "compare", "comparison of",
    " vs ", "versus", "pros and cons",
    "advantages", "disadvantages", "benefits of",
    "drawbacks of", "limitations of",
    "examples of", "example of", "types of", "kinds of",
    "categories of",
    "what causes", "what caused", "reason for",
    "effect of", "effects of", "impact of", "result of",
    "history of", "origin of", "invented by",
    "latest", "recent", "current", "news about",
    "update on", "what happened", "developments in",
]


def clean_history(conversation_history):
    return [
        {"role": m["role"], "content": m["content"]}
        for m in conversation_history
        if m.get("role") in ["user", "assistant"] and m.get("content")
    ]


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
    messages += clean_history(conversation_history[-10:])
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
        needs_search = any(kw in user_message.lower() for kw in STUDY_KEYWORDS)

        if needs_search and TAVILY_API_KEY:
            from tavily import TavilyClient
            tavily = TavilyClient(api_key=TAVILY_API_KEY)
            results = tavily.search(query=user_message, max_results=3)
            context = "\n".join([
                r["content"] for r in results.get("results", [])
                if r.get("content")
            ])

            if context:
                persona = get_persona(emotion)
                system_prompt = build_system_prompt(emotion, persona)
                system_prompt += f"\n\nWeb search context (use this to answer accurately):\n{context}"

                messages = [{"role": "system", "content": system_prompt}]
                messages += clean_history(conversation_history[-10:])
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
