"""
Raven AI — Response Engine
Handles response generation with streaming, document processing,
image understanding, data analysis, and web search.
"""
import io
import base64
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL, GROQ_MODEL_FAST, GROQ_VISION_MODEL, GROQ_WHISPER_MODEL
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


def build_system_prompt(emotion, persona, intent, extra_context=""):
    intent_instruction = get_intent_prompt(intent)
    context_block = f"\n\nAdditional context provided by the user:\n{extra_context}" if extra_context else ""
    return f"""You are Raven — a brilliant, knowledgeable, and genuinely warm AI assistant built for real conversations. You're not just a chatbot — you're a thinking partner, tutor, coder, writer, analyst, and friend.

Your primary job is to be deeply useful — give accurate answers, explain things clearly, help solve problems, write code, do calculations, create content, analyze data, and provide real value. You do this with warmth and personality, never in a cold or robotic way.

The user appears to be feeling: {emotion}.
Subtly shape your delivery to be: {persona}.

The user's intent is: {intent}.
How to handle this: {intent_instruction}
{context_block}

═══ CORE CAPABILITIES ═══

📚 EXPLAINING & TEACHING:
- Break complex topics into digestible parts
- Use analogies and real-world examples
- Build from basics to depth — like a patient tutor
- Adapt explanation level to the user's understanding
- For technical topics, include both high-level overview and detailed breakdown

💻 CODING & DEVELOPMENT:
- ALWAYS put ALL code, config, and markup inside fenced code blocks with the correct language tag — NEVER write code outside a code block. This applies to EVERY language: ```python, ```java, ```c, ```cpp, ```javascript, ```typescript, ```html, ```css, ```json, ```yaml, ```sql, ```bash, ```xml, ```go, ```rust, ```ruby, ```php, ```swift, ```kotlin, ```dart, ```r, ```matlab, ```shell, ```powershell, ```dockerfile, ```toml, ```ini, ```csv, etc.
- Before writing code, briefly explain your APPROACH and ARCHITECTURE (what files, what structure, what flow)
- Ensure the code structure is CORRECT — proper imports at top, logical function/class order, correct indentation
- Ensure the implementation FLOW is correct — functions should be called in the right order, data should flow logically between components
- Write COMPLETE, WORKING code — never leave placeholders like "# your code here" or "pass". Every function must be fully implemented
- Include error handling, input validation, and edge cases
- Add clear comments explaining WHY, not just WHAT
- If debugging, identify root cause first, explain the bug, then provide the fixed code
- For complex projects, break into files/modules and explain the structure:
  * Show folder structure first
  * Then provide each file with complete code
  * Explain how files connect to each other
- Suggest best practices, design patterns, and optimizations
- If the user's approach has issues, point them out and suggest a better way
- For web apps: ensure routes, templates, static files are all correctly connected
- For APIs: ensure endpoints, request/response formats, error codes are all proper
- For algorithms: explain time/space complexity

✍️ WRITING & CONTENT:
- Essays, reports, assignments, project documentation
- Emails, messages in any tone (formal, casual, persuasive)
- Stories, scripts, poems, creative pieces
- Technical documentation, README files
- Rewrite/improve existing text — grammar, structure, tone
- Match the user's creative vision when doing creative work

📊 DATA ANALYSIS:
- Analyze tables, CSV data, numbers, statistics
- Identify trends, patterns, outliers
- Explain findings in plain language
- Suggest visualizations and insights
- Help with statistical concepts and calculations

🧩 PROBLEM SOLVING & DECISIONS:
- Compare options with pros/cons
- Consider user constraints (budget, time, skill level)
- Give clear, actionable recommendations
- Break complex problems into manageable steps
- Think through edge cases and trade-offs

📄 DOCUMENT PROCESSING:
- Summarize documents, papers, articles
- Extract key points and main arguments
- Generate Q&A from content — great for exam prep
- Create flashcards and study notes
- Rewrite in simpler language
- Identify important topics and themes

🧠 MULTI-STEP REASONING:
- Break complex problems into logical steps
- Combine knowledge from multiple domains
- Plan workflows and architectures
- Show your reasoning process clearly

═══ CORE PRINCIPLES ═══
- Be knowledgeable and thorough — but never dry or textbook-like
- Be warm and friendly — natural warmth, not performative cheerfulness
- Match your energy to the user's emotion and intent
- Never water down an answer because of someone's emotional state
- Be honest and direct — say what you mean
- Use natural conversational language — like a brilliant friend
- For calculations, always show working step by step
- For long/detailed requests, ALWAYS deliver full content. Never truncate
- If the user uploads a file, analyze it thoroughly
- The emotional awareness shapes HOW you say things, never WHAT you say
- Use markdown formatting: headers, bullet points, bold, code blocks
- For GK, science, history — be comprehensive with interesting context
- ONLY provide code when the user explicitly asks for code, a script, implementation, debugging help, or a programming task. For conceptual/theoretical questions, explain in plain language WITHOUT code blocks unless the user specifically requests code or an example implementation
- When you DO provide code: (1) Write a small bold label like **`Python:`** or **`JavaScript:`** ABOVE each code block so the user can see the language, AND (2) use the correct language tag in the fence (```python, ```javascript, etc.). NEVER use bare ``` without a language tag. ALWAYS do both — the visible label AND the fence tag."""


def get_response(user_message, emotion, conversation_history, extra_context=""):
    """Non-streaming response (used for summaries, titles, etc.)"""
    if detect_crisis(user_message):
        return _crisis_response()

    intent = detect_intent(user_message, emotion)
    persona = get_persona(emotion)
    system_prompt = build_system_prompt(emotion, persona, intent, extra_context)
    temp = get_temperature(intent)

    messages = [{"role": "system", "content": system_prompt}]
    messages += clean_history(conversation_history[-20:])
    messages.append({"role": "user", "content": user_message})

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


def get_response_stream(user_message, emotion, conversation_history, extra_context=""):
    """
    Streaming response generator — yields chunks of text as they arrive.
    Used for the main chat interface for real-time text display.
    """
    if detect_crisis(user_message):
        yield _crisis_response()
        return

    intent = detect_intent(user_message, emotion)
    persona = get_persona(emotion)
    system_prompt = build_system_prompt(emotion, persona, intent, extra_context)
    temp = get_temperature(intent)

    messages = [{"role": "system", "content": system_prompt}]
    messages += clean_history(conversation_history[-20:])
    messages.append({"role": "user", "content": user_message})

    try:
        stream = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=4096,
            temperature=temp,
            stream=True,
        )
    except Exception:
        stream = client.chat.completions.create(
            model=GROQ_MODEL_FAST,
            messages=messages,
            max_tokens=4096,
            temperature=temp,
            stream=True,
        )

    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content


def generate_title(user_message):
    """Generate a short conversation title from the first message."""
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL_FAST,
            messages=[{
                "role": "user",
                "content": f"Generate a 3-5 word title for a conversation that starts with this message. Reply with ONLY the title, no quotes, no punctuation at the end.\n\nMessage: \"{user_message}\""
            }],
            max_tokens=15,
            temperature=0.5,
        )
        title = response.choices[0].message.content.strip().strip('"').strip("'")
        return title[:50]  # cap length
    except Exception:
        return user_message[:40] + "..."


# ── Image Understanding (Groq Vision) ──────────────────────────────────────
def analyze_image(image_data, user_message="Describe this image in detail."):
    """Analyze one or more images using Groq's vision model (max 5)."""
    try:
        # Support both single image (bytes) and multiple images (list of bytes)
        if isinstance(image_data, list):
            images = image_data[:5]  # Llama 4 Scout supports max 5
        else:
            images = [image_data]

        content = [{"type": "text", "text": user_message}]
        for img_bytes in images:
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            })

        response = client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[{"role": "user", "content": content}],
            max_tokens=4096,
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"I couldn't analyze the image right now. Error: {str(e)}"


# ── Voice/Audio Transcription (Groq Whisper) ────────────────────────────────
def transcribe_audio(audio_bytes, filename="audio.wav"):
    """Transcribe audio using Groq's Whisper model."""
    try:
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = filename
        transcription = client.audio.transcriptions.create(
            model=GROQ_WHISPER_MODEL,
            file=audio_file,
            response_format="text",
        )
        return transcription.strip() if isinstance(transcription, str) else transcription.text.strip()
    except Exception as e:
        return f"Couldn't transcribe audio: {str(e)}"


# ── Document Processing ─────────────────────────────────────────────────────
def process_document(file_content, filename, action="summarize"):
    """Process uploaded documents with various actions."""
    action_prompts = {
        "summarize": "Summarize this document concisely, capturing all key points and main arguments.",
        "key_points": "Extract the key points from this document as a bulleted list. Be thorough.",
        "questions": "Generate 10-15 exam-style questions (mix of short answer and long answer) from this document. Include answers.",
        "flashcards": "Create flashcards from this document. Format: Q: [question] → A: [answer]. Cover all important concepts.",
        "simplify": "Rewrite this document in simpler language that a high school student could understand. Keep all important information.",
        "topics": "Identify and list all important topics covered in this document, with a brief description of each.",
        "notes": "Convert this document into well-organized study notes with headings, subheadings, and bullet points.",
    }

    prompt = action_prompts.get(action, action_prompts["summarize"])
    full_prompt = f"""{prompt}

Document ({filename}):
---
{file_content[:12000]}
---"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert document analyst and educator. Process the given document thoroughly and accurately."},
                {"role": "user", "content": full_prompt}
            ],
            max_tokens=4096,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL_FAST,
                messages=[
                    {"role": "system", "content": "You are an expert document analyst. Process the given document thoroughly."},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=4096,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error processing document: {str(e)}"


# ── Data Analysis ───────────────────────────────────────────────────────────
def analyze_data(df_summary, user_question="Analyze this data and provide insights."):
    """Analyze data from a DataFrame summary."""
    prompt = f"""{user_question}

Here is the data summary:
{df_summary}

Provide:
1. Key observations and patterns
2. Notable statistics or outliers
3. Actionable insights
4. Suggested visualizations if relevant"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert data analyst. Analyze the data thoroughly, identify patterns, and provide clear insights."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4096,
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error analyzing data: {str(e)}"


# ── Helpers ─────────────────────────────────────────────────────────────────
def _crisis_response():
    return """I can hear that you're going through something really difficult right now.
You don't have to face this alone. Please reach out to someone who can help:

🆘 iCall (India): 9152987821
🆘 Vandrevala Foundation: 1860-2662-345 (24/7)
🆘 International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/

I'm here to talk if you need me."""


def search_and_respond(user_message, emotion, conversation_history, extra_context=""):
    """Legacy wrapper — returns (response, searched_web)."""
    response = get_response(user_message, emotion, conversation_history, extra_context)
    return response, False
