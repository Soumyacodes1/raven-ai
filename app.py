"""
Raven AI — Main Application
Full-featured emotionally-aware AI assistant with streaming, multi-chat,
document processing, image understanding, voice input, data analysis, and web search.
"""
import json
import os
import uuid
import time
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from emotion_engine import detect_emotion, detect_intent, detect_crisis
from responder import (
    get_response, get_response_stream, generate_title,
    analyze_image, transcribe_audio, process_document, analyze_data,
    search_and_respond,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Raven AI",
    page_icon="🐦‍⬛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Conversation persistence file ────────────────────────────────────────────
CONV_FILE = os.path.join(os.path.dirname(__file__), ".raven_conversations.json")


def save_conversations():
    """Save all conversations to disk for persistence."""
    try:
        data = {}
        for cid, conv in st.session_state.conversations.items():
            data[cid] = {
                "title": conv.get("title", "New Chat"),
                "messages": conv.get("messages", []),
                "emotion_log": conv.get("emotion_log", []),
                "current_emotion": conv.get("current_emotion", "neutral"),
                "current_intent": conv.get("current_intent", "casual"),
                "created_at": conv.get("created_at", time.time()),
            }
        with open(CONV_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def load_conversations():
    """Load conversations from disk."""
    try:
        if os.path.exists(CONV_FILE):
            with open(CONV_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    [data-testid="stSidebar"] {
        background: #0a0a0f;
        border-right: 1px solid #1e1e2e;
    }

    .stChatMessage {
        border-radius: 12px !important;
        margin-bottom: 8px !important;
    }

    /* Small action buttons under responses */
    .action-row {
        display: flex;
        gap: 4px;
        margin-top: 4px;
        align-items: center;
    }
    .action-btn {
        background: transparent;
        border: 1px solid #2d2d3a;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 13px;
        cursor: pointer;
        color: #6B7280;
        transition: all 0.15s;
        line-height: 1.4;
    }
    .action-btn:hover {
        background: #1e1e2e;
        border-color: #6366F1;
        color: #E5E7EB;
    }
    .action-btn.active {
        background: #6366F122;
        border-color: #6366F1;
        color: #6366F1;
    }

    /* Word count */
    .word-count {
        font-size: 11px;
        color: #4B5563;
        margin-top: 2px;
    }

    /* Badge styling */
    .raven-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        margin-right: 6px;
        font-weight: 500;
    }

    /* Welcome card */
    .welcome-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d1b4e 100%);
        border: 1px solid #3d2d5e;
        border-radius: 16px;
        padding: 24px;
        margin: 8px 4px;
        min-height: 100px;
    }

    /* Position 📎 and 🎙️ buttons overlaid on the chat input bar */
    .chat-tools-wrapper {
        position: relative;
    }
    .chat-tools-wrapper .tool-left {
        position: absolute;
        bottom: 18px;
        left: 8px;
        z-index: 100;
    }
    .chat-tools-wrapper .tool-right {
        position: absolute;
        bottom: 18px;
        right: 8px;
        z-index: 100;
    }

    /* Make the popover trigger buttons look clean */
    .tool-btn [data-testid="stPopoverButton"] > button {
        background: transparent !important;
        border: none !important;
        padding: 4px 8px !important;
        font-size: 18px !important;
        color: #6B7280 !important;
        min-height: 0 !important;
        line-height: 1 !important;
    }
    .tool-btn [data-testid="stPopoverButton"] > button:hover {
        color: #E5E7EB !important;
        background: #1e1e2e !important;
        border-radius: 6px !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0a0a0f; }
    ::-webkit-scrollbar-thumb { background: #2d2d3a; border-radius: 3px; }

    .stButton > button {
        border-radius: 8px !important;
        font-weight: 500 !important;
    }

    hr { border-color: #1e1e2e !important; }

    /* Code block language labels */
    .stCodeBlock [data-testid="stCodeBlockLanguage"],
    code[class*="language-"]::before {
        font-size: 11px !important;
        color: #9CA3AF !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

</style>
""", unsafe_allow_html=True)


# ── Style maps ───────────────────────────────────────────────────────────────
EMOTION_STYLE = {
    "happy":    {"emoji": "😊", "color": "#F59E0B"},
    "sad":      {"emoji": "😔", "color": "#60A5FA"},
    "anxious":  {"emoji": "😰", "color": "#A78BFA"},
    "angry":    {"emoji": "😤", "color": "#F87171"},
    "confused": {"emoji": "😕", "color": "#34D399"},
    "neutral":  {"emoji": "😐", "color": "#9CA3AF"},
}

INTENT_STYLE = {
    "venting":   {"emoji": "💬", "label": "Venting"},
    "question":  {"emoji": "🔍", "label": "Question"},
    "advice":    {"emoji": "💡", "label": "Advice"},
    "casual":    {"emoji": "😌", "label": "Casual"},
    "crisis":    {"emoji": "🆘", "label": "Crisis"},
    "coding":    {"emoji": "💻", "label": "Coding"},
    "learning":  {"emoji": "📚", "label": "Learning"},
    "creative":  {"emoji": "🎨", "label": "Creative"},
    "deep_dive": {"emoji": "🔬", "label": "Deep Dive"},
}

EMOTION_ORDER = ["happy", "neutral", "confused", "anxious", "sad", "angry"]
EMOTION_COLORS = {
    "happy": "#F59E0B", "sad": "#60A5FA", "anxious": "#A78BFA",
    "angry": "#F87171", "confused": "#34D399", "neutral": "#9CA3AF",
}

WELCOME_PROMPTS = [
    {"emoji": "💻", "title": "Write Code", "prompt": "Write a Python script that scrapes the top 10 trending GitHub repos"},
    {"emoji": "📚", "title": "Learn Something", "prompt": "Explain how neural networks work — from scratch, step by step"},
    {"emoji": "🎨", "title": "Creative Writing", "prompt": "Write a short sci-fi story about an AI that develops emotions"},
    {"emoji": "🧩", "title": "Solve a Problem", "prompt": "I need to build a portfolio website. Help me plan the architecture and tech stack"},
    {"emoji": "📊", "title": "Analyze Data", "prompt": "Explain the key concepts of statistics I need for data science"},
    {"emoji": "🔬", "title": "Deep Dive", "prompt": "Give me an exhaustive breakdown of how transformers work in AI"},
]


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

def create_conversation():
    """Create a new conversation and return its ID."""
    conv_id = str(uuid.uuid4())
    st.session_state.conversations[conv_id] = {
        "title": "New Chat",
        "messages": [],
        "emotion_log": [],
        "current_emotion": "neutral",
        "current_intent": "casual",
        "created_at": time.time(),
    }
    return conv_id


def get_conv():
    """Get the active conversation dict."""
    return st.session_state.conversations.get(st.session_state.active_conv_id, {})


def delete_conversation(conv_id):
    """Delete a conversation."""
    if conv_id in st.session_state.conversations:
        del st.session_state.conversations[conv_id]
    # If we deleted the active one, switch to another or create new
    if conv_id == st.session_state.active_conv_id:
        remaining = list(st.session_state.conversations.keys())
        if remaining:
            st.session_state.active_conv_id = remaining[0]
        else:
            new_id = create_conversation()
            st.session_state.active_conv_id = new_id
    save_conversations()


def init_session_state():
    """Initialize all session state variables."""
    if "conversations" not in st.session_state:
        loaded = load_conversations()
        if loaded:
            st.session_state.conversations = loaded
        else:
            st.session_state.conversations = {}
    if "active_conv_id" not in st.session_state:
        # Always start with a fresh new chat — old chats stay in sidebar
        conv_id = create_conversation()
        st.session_state.active_conv_id = conv_id
    # Ensure active conv exists
    if st.session_state.active_conv_id not in st.session_state.conversations:
        conv_id = create_conversation()
        st.session_state.active_conv_id = conv_id
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None
    if "regenerate_idx" not in st.session_state:
        st.session_state.regenerate_idx = None
    if "uploaded_context" not in st.session_state:
        st.session_state.uploaded_context = None
    if "preferences" not in st.session_state:
        st.session_state.preferences = {"tone": "balanced", "name": ""}
    if "voice_transcript" not in st.session_state:
        st.session_state.voice_transcript = None
    if "edit_msg_idx" not in st.session_state:
        st.session_state.edit_msg_idx = None
    if "upload_key" not in st.session_state:
        st.session_state.upload_key = 0


init_session_state()


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def build_emotion_chart(emotion_log):
    """Build the emotion trend chart."""
    if len(emotion_log) < 2:
        return None
    emotions = [e for _, e in emotion_log]
    y_vals = [EMOTION_ORDER.index(e) if e in EMOTION_ORDER else 2 for e in emotions]
    x_vals = list(range(1, len(emotions) + 1))
    colors = [EMOTION_COLORS.get(e, "#9CA3AF") for e in emotions]

    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.plot(x_vals, y_vals, color="#6366F1", linewidth=1.5, alpha=0.6, zorder=1)
    for x, y, c in zip(x_vals, y_vals, colors):
        ax.scatter(x, y, color=c, s=50, zorder=2)
    ax.set_yticks(range(len(EMOTION_ORDER)))
    ax.set_yticklabels(EMOTION_ORDER, fontsize=7, color="#9CA3AF")
    ax.set_xticks(x_vals[-10:])
    ax.set_xticklabels([str(i) for i in x_vals[-10:]], fontsize=7, color="#9CA3AF")
    ax.set_xlabel("Message", fontsize=7, color="#9CA3AF")
    ax.tick_params(colors="#9CA3AF")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d2d3a")
    ax.grid(axis="y", color="#2d2d3a", linewidth=0.5, linestyle="--")
    plt.tight_layout(pad=0.5)
    return fig


def get_recent_context():
    """Get recent user messages for context."""
    conv = get_conv()
    user_msgs = [m["content"] for m in conv.get("messages", []) if m.get("role") == "user"][-2:]
    return " | ".join(user_msgs) if user_msgs else ""


def render_badges(emotion, intent):
    """Render emotion + intent badges."""
    s = EMOTION_STYLE.get(emotion, EMOTION_STYLE["neutral"])
    i = INTENT_STYLE.get(intent, INTENT_STYLE["casual"])
    st.markdown(
        f"<span class='raven-badge' style='background:{s['color']}22;color:{s['color']}'>"
        f"{s['emoji']} {emotion}</span>"
        f"<span class='raven-badge' style='background:#1e1e2e;color:#9CA3AF'>"
        f"{i['emoji']} {i['label']}</span>",
        unsafe_allow_html=True,
    )


def render_word_count(text):
    """Show word count below a response."""
    words = len(text.split())
    st.markdown(f"<div class='word-count'>{words} words</div>", unsafe_allow_html=True)


def process_uploaded_file(uploaded_file):
    """Process an uploaded file and return (content_text, file_type)."""
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()

    if name.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
        return raw, "image"
    if name.endswith((".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm")):
        return raw, "audio"
    if name.endswith((".csv", ".tsv")):
        try:
            sep = "\t" if name.endswith(".tsv") else ","
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=sep)
            summary = f"Shape: {df.shape}\n\nColumns: {list(df.columns)}\n\nFirst 10 rows:\n{df.head(10).to_string()}\n\nDescribe:\n{df.describe().to_string()}"
            return summary, "data"
        except Exception:
            return raw.decode("utf-8", errors="ignore"), "text"
    if name.endswith(".pdf"):
        try:
            import PyPDF2
            uploaded_file.seek(0)
            reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in reader.pages[:50]:
                text += page.extract_text() or ""
            return text, "document"
        except Exception:
            return "Could not read PDF.", "error"
    try:
        text = raw.decode("utf-8", errors="ignore")
        return text, "document"
    except Exception:
        return "Could not read file.", "error"


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Clean layout: New Chat, Conversations, Mood, Preferences, Graph
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    # ── i) Header + New Chat ─────────────────────────────────────────────────
    st.markdown(
        "<h1 style='margin:0;padding:0'>🐦‍⬛ Raven AI</h1>"
        "<p style='color:#9CA3AF;margin:0 0 8px 0;font-size:13px'>"
        "Thought. Memory. Empathy.</p>",
        unsafe_allow_html=True,
    )

    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        conv_id = create_conversation()
        st.session_state.active_conv_id = conv_id
        st.session_state.uploaded_context = None
        st.session_state.upload_key += 1
        save_conversations()
        st.rerun()

    st.divider()

    # ── ii) Past Conversations (with 3-dot menu each) ────────────────────────
    convs = sorted(
        st.session_state.conversations.items(),
        key=lambda x: x[1].get("created_at", 0),
        reverse=True,
    )

    # Filter: show chats that have messages, plus the active one
    visible_convs = [
        (cid, c) for cid, c in convs
        if len([m for m in c.get("messages", []) if m.get("role") == "user"]) > 0
        or cid == st.session_state.active_conv_id
    ]

    if visible_convs:
        st.markdown(
            "<p style='color:#6B7280;font-size:11px;text-transform:uppercase;"
            "letter-spacing:1px;margin-bottom:4px'>Conversations</p>",
            unsafe_allow_html=True,
        )

        for conv_id, conv in visible_convs[:30]:
            is_active = conv_id == st.session_state.active_conv_id
            title = conv.get("title", "New Chat")
            msg_count = len([m for m in conv.get("messages", []) if m.get("role") == "user"])

            col_title, col_menu = st.columns([5, 1])

            with col_title:
                display = f"**{title}**" if is_active else title
                if st.button(
                    display,
                    key=f"conv_{conv_id}",
                    use_container_width=True,
                    disabled=is_active,
                ):
                    st.session_state.active_conv_id = conv_id
                    st.session_state.uploaded_context = None
                    st.session_state.upload_key += 1
                    st.rerun()

            with col_menu:
                with st.popover("⋮", use_container_width=True):
                    if st.button("🗑️ Delete", key=f"del_{conv_id}", use_container_width=True):
                        delete_conversation(conv_id)
                        st.rerun()

                    if msg_count > 0:
                        if st.button("📥 Export", key=f"exp_{conv_id}", use_container_width=True):
                            export_data = [
                                {"role": m.get("role", ""), "content": m.get("content", ""),
                                 "emotion": m.get("emotion", ""), "intent": m.get("intent", "")}
                                for m in conv.get("messages", [])
                            ]
                            csv_data = pd.DataFrame(export_data).to_csv(index=False)
                            st.download_button(
                                "Download CSV", csv_data,
                                f"raven_{title[:20]}.csv", "text/csv",
                                key=f"dl_{conv_id}",
                            )

                        if st.button("📋 Summarize", key=f"sum_{conv_id}", use_container_width=True):
                            emotion_trend = [e for _, e in conv.get("emotion_log", [])]
                            summary_prompt = (
                                f"Summarise this conversation in 3-4 sentences. "
                                f"Detected emotions: {', '.join(emotion_trend) if emotion_trend else 'none'}. "
                                f"Base your summary strictly on these emotions."
                            )
                            clean_msgs = [
                                {"role": m["role"], "content": m["content"]}
                                for m in conv.get("messages", [])
                                if m.get("role") in ["user", "assistant"] and m.get("content")
                            ]
                            summary = get_response(
                                summary_prompt,
                                conv.get("current_emotion", "neutral"),
                                clean_msgs,
                            )
                            st.info(summary)

    st.divider()

    # ── iii) Current Mood & Intent ───────────────────────────────────────────
    conv = get_conv()
    emotion = conv.get("current_emotion", "neutral")
    style = EMOTION_STYLE.get(emotion, EMOTION_STYLE["neutral"])
    color, emoji = style["color"], style["emoji"]

    st.markdown(
        "<p style='color:#6B7280;font-size:11px;text-transform:uppercase;"
        "letter-spacing:1px;margin-bottom:4px'>Mood & Intent</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='background:#1e1e2e;padding:10px;border-radius:10px;"
        f"border-left:4px solid {color};font-size:15px;margin-bottom:6px'>"
        f"{emoji} <b style='color:{color}'>{emotion.capitalize()}</b></div>",
        unsafe_allow_html=True,
    )

    intent = conv.get("current_intent", "casual")
    intent_info = INTENT_STYLE.get(intent, INTENT_STYLE["casual"])
    st.markdown(
        f"<div style='background:#1e1e2e;padding:8px;border-radius:10px;"
        f"border-left:4px solid #6B7280;font-size:13px;margin-bottom:8px'>"
        f"{intent_info['emoji']} <span style='color:#E5E7EB'>{intent_info['label']}</span></div>",
        unsafe_allow_html=True,
    )

    # ── iv) Preferences ──────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        "<p style='color:#6B7280;font-size:11px;text-transform:uppercase;"
        "letter-spacing:1px;margin-bottom:4px'>Preferences</p>",
        unsafe_allow_html=True,
    )

    prefs = st.session_state.preferences
    new_name = st.text_input(
        "Your name",
        value=prefs.get("name", ""),
        placeholder="So Raven can address you",
        key="pref_name",
        label_visibility="collapsed",
    )
    prefs["name"] = new_name

    new_tone = st.selectbox(
        "Tone",
        ["Balanced", "Formal", "Casual", "Concise", "Detailed"],
        index=["balanced", "formal", "casual", "concise", "detailed"].index(
            prefs.get("tone", "balanced")
        ) if prefs.get("tone", "balanced") in ["balanced", "formal", "casual", "concise", "detailed"] else 0,
        key="pref_tone",
        label_visibility="collapsed",
    )
    prefs["tone"] = new_tone.lower()

    # ── v) Emotion Trend Graph ───────────────────────────────────────────────
    emotion_log = conv.get("emotion_log", [])
    if len(emotion_log) >= 2:
        st.divider()
        st.markdown(
            "<p style='color:#6B7280;font-size:11px;text-transform:uppercase;"
            "letter-spacing:1px;margin-bottom:4px'>Emotion Trend</p>",
            unsafe_allow_html=True,
        )
        fig = build_emotion_chart(emotion_log)
        if fig:
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CHAT AREA
# ═══════════════════════════════════════════════════════════════════════════════

conv = get_conv()
messages = conv.get("messages", [])

# ── Welcome screen (when no messages) ────────────────────────────────────────
if not messages:
    st.markdown("")
    st.markdown(
        "<div style='text-align:center;padding:30px 0 10px 0'>"
        "<h1 style='font-size:48px;margin:0'>🐦‍⬛</h1>"
        "<h2 style='margin:8px 0 4px 0;color:#E5E7EB'>Hello, I'm Raven</h2>"
        "<p style='color:#9CA3AF;font-size:16px;margin:0 0 24px 0'>"
        "Your emotionally aware AI assistant — I can code, explain, create, analyze, and more.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    for idx, prompt_item in enumerate(WELCOME_PROMPTS):
        with cols[idx % 3]:
            if st.button(
                f"{prompt_item['emoji']} {prompt_item['title']}\n{prompt_item['prompt'][:50]}...",
                key=f"welcome_{idx}",
                use_container_width=True,
            ):
                st.session_state.pending_prompt = prompt_item["prompt"]
                st.rerun()

    st.markdown(
        "<div style='text-align:center;padding:16px 0'>"
        "<p style='color:#6B7280;font-size:13px'>"
        "📄 Upload docs &nbsp;•&nbsp; "
        "📷 Analyze images &nbsp;•&nbsp; "
        "🎙️ Voice input &nbsp;•&nbsp; "
        "📊 Data analysis &nbsp;•&nbsp; "
        "🌐 Web search</p>"
        "</div>",
        unsafe_allow_html=True,
    )

# ── Render existing messages ─────────────────────────────────────────────────
else:
    for idx, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content", "")
        with st.chat_message(role):
            if role == "user":
                render_badges(msg.get("emotion", "neutral"), msg.get("intent", "casual"))

                # ── Edit mode for this user message ──
                if st.session_state.edit_msg_idx == idx:
                    edited_text = st.text_area(
                        "Edit your message:",
                        value=content,
                        key=f"edit_area_{idx}",
                        height=100,
                        label_visibility="collapsed",
                    )
                    ec1, ec2, ec3 = st.columns([1, 1, 6])
                    with ec1:
                        if st.button("Save & Resend", key=f"edit_save_{idx}", type="primary"):
                            edited_text = edited_text.strip()
                            if edited_text:
                                # Count how many user messages come after this index
                                # so we can trim emotion_log accordingly
                                user_msgs_after = sum(
                                    1 for m in messages[idx:]
                                    if m.get("role") == "user"
                                )
                                emotion_log = conv.get("emotion_log", [])
                                if user_msgs_after > 0 and len(emotion_log) >= user_msgs_after:
                                    del emotion_log[-user_msgs_after:]
                                # Remove all messages from this index onward
                                del messages[idx:]
                                # Trigger re-generation by setting pending_prompt
                                st.session_state.pending_prompt = edited_text
                                st.session_state.edit_msg_idx = None
                                save_conversations()
                                st.rerun()
                    with ec2:
                        if st.button("Cancel", key=f"edit_cancel_{idx}"):
                            st.session_state.edit_msg_idx = None
                            st.rerun()
                else:
                    st.markdown(content)
                    # Show edit and copy buttons on user messages
                    user_btn_cols = st.columns([0.5, 0.5, 8])
                    with user_btn_cols[0]:
                        if st.button("✏️", key=f"edit_{idx}", help="Edit message"):
                            st.session_state.edit_msg_idx = idx
                    with user_btn_cols[1]:
                        with st.popover("📋", help="Copy message"):
                            st.code(content, language=None)

            else:
                st.markdown(content)

            # Assistant message: small action buttons
            if role == "assistant":
                render_word_count(content)
                feedback = msg.get("feedback")
                is_last = idx == len(messages) - 1

                # Small inline buttons
                btn_cols = st.columns([0.5, 0.5, 0.5, 0.5, 7] if is_last else [0.5, 0.5, 0.5, 7.5])

                with btn_cols[0]:
                    up_label = "👍" if feedback != "up" else "✅"
                    if st.button(up_label, key=f"up_{idx}", help="Good response"):
                        messages[idx]["feedback"] = "up"
                        save_conversations()
                        st.rerun()

                with btn_cols[1]:
                    dn_label = "👎" if feedback != "down" else "❌"
                    if st.button(dn_label, key=f"dn_{idx}", help="Bad response"):
                        messages[idx]["feedback"] = "down"
                        save_conversations()
                        st.rerun()

                with btn_cols[2]:
                    with st.popover("📋", help="Copy"):
                        st.code(content, language=None)

                if is_last:
                    with btn_cols[3]:
                        if st.button("🔄", key=f"rg_{idx}", help="Regenerate"):
                            st.session_state.regenerate_idx = idx
                            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# HANDLE REGENERATION
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.regenerate_idx is not None:
    regen_idx = st.session_state.regenerate_idx
    st.session_state.regenerate_idx = None
    conv = get_conv()
    messages = conv.get("messages", [])

    if regen_idx < len(messages) and messages[regen_idx].get("role") == "assistant":
        user_msg = None
        user_emotion = "neutral"
        for i in range(regen_idx - 1, -1, -1):
            if messages[i].get("role") == "user":
                user_msg = messages[i]["content"]
                user_emotion = messages[i].get("emotion", "neutral")
                break

        if user_msg:
            messages.pop(regen_idx)
            with st.chat_message("assistant"):
                with st.status("🔄 Regenerating...", expanded=True) as status:
                    st.write("Generating new response...")
                    status.update(label="Streaming...", state="running")

                clean_msgs = [
                    {"role": m["role"], "content": m["content"]}
                    for m in messages
                    if m.get("role") in ["user", "assistant"] and m.get("content")
                ]
                stream = get_response_stream(user_msg, user_emotion, clean_msgs)
                response = st.write_stream(stream)

            messages.append({"role": "assistant", "content": response})
            save_conversations()
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# CHAT INPUT AREA — 📎 (inside left) + Text input + 🎙️ (inside right)
# ═══════════════════════════════════════════════════════════════════════════════

# Show attached files above chat input (thumbnails for images, tags for others)
if st.session_state.uploaded_context:
    ctx = st.session_state.uploaded_context
    attach_display_col, clear_col = st.columns([9, 1])
    with attach_display_col:
        if ctx["type"] == "image" and isinstance(ctx["content"], list):
            # Show image thumbnails in a row
            img_cols = st.columns(min(len(ctx["content"]), 5) + 1)
            for i, img_bytes in enumerate(ctx["content"][:5]):
                with img_cols[i]:
                    st.image(img_bytes, width=80, caption=ctx["names"][i] if i < len(ctx.get("names", [])) else f"Image {i+1}")
        elif ctx["type"] == "image":
            st.image(ctx["content"], width=80, caption=ctx.get("name", "Image"))
        else:
            type_emoji = {"audio": "🎙️", "data": "📊", "document": "📄"}.get(ctx["type"], "📎")
            st.markdown(
                f"<div style='font-size:12px;color:#9CA3AF;padding:4px 10px;background:#1e1e2e;"
                f"border-radius:6px;display:inline-block'>"
                f"{type_emoji} {ctx.get('name', 'File')} attached</div>",
                unsafe_allow_html=True,
            )
    with clear_col:
        if st.button("✕", key="clear_upload", help="Remove attachment"):
            st.session_state.uploaded_context = None
            st.session_state.upload_key += 1
            st.rerun()

# File upload popover (appears above chat, left side)
attach_col, spacer_col, voice_col = st.columns([1, 8, 1])

with attach_col:
    with st.popover("📎", use_container_width=True, help="Attach file"):
        uploaded_files = st.file_uploader(
            "Upload files",
            type=["pdf", "txt", "py", "js", "java", "c", "cpp", "csv", "tsv",
                  "json", "md", "html", "css", "png", "jpg", "jpeg", "gif",
                  "webp", "wav", "mp3", "m4a", "ogg", "flac", "webm",
                  "xlsx", "docx"],
            label_visibility="collapsed",
            accept_multiple_files=True,
            key=f"file_upload_{st.session_state.upload_key}",
        )

        if uploaded_files:
            # Check if all files are images
            image_files = [f for f in uploaded_files if f.name.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp"))]

            if len(image_files) == len(uploaded_files) and len(image_files) > 1:
                # Multiple images — batch them together (max 5)
                images_bytes = []
                names = []
                for img_file in image_files[:5]:
                    images_bytes.append(img_file.read())
                    names.append(img_file.name)
                for img_file in image_files[:5]:
                    img_file.seek(0)
                    st.image(img_file, caption=img_file.name, width=100)
                st.session_state.uploaded_context = {"type": "image", "content": images_bytes, "names": names}
                st.success(f"📷 {len(images_bytes)} images ready — ask about them!")
                if len(image_files) > 5:
                    st.warning("Max 5 images per request — first 5 selected.")
            else:
                # Single file or mixed — process the first file
                uploaded_file = uploaded_files[0]
                content, file_type = process_uploaded_file(uploaded_file)

                if file_type == "image":
                    st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)
                    st.session_state.uploaded_context = {"type": "image", "content": content, "name": uploaded_file.name}
                    st.success("📷 Image ready — ask about it!")

                elif file_type == "audio":
                    st.audio(uploaded_file)
                    with st.spinner("Transcribing..."):
                        transcript = transcribe_audio(content, uploaded_file.name)
                    st.session_state.uploaded_context = {"type": "audio", "content": transcript, "name": uploaded_file.name}
                    st.success("🎙️ Transcribed!")

                elif file_type == "data":
                    st.session_state.uploaded_context = {"type": "data", "content": content, "name": uploaded_file.name}
                    st.success("📊 Data loaded!")

                elif file_type == "document":
                    st.session_state.uploaded_context = {"type": "document", "content": content, "name": uploaded_file.name}
                    st.success("📄 Document ready!")
                    doc_action = st.selectbox(
                        "Quick action:",
                        ["Chat about it", "Summarize", "Key Points", "Generate Q&A",
                         "Flashcards", "Simplify", "Study Notes", "Topics"],
                        key="doc_action_chat",
                    )
                    action_map = {
                        "Summarize": "summarize", "Key Points": "key_points",
                        "Generate Q&A": "questions", "Flashcards": "flashcards",
                        "Simplify": "simplify", "Study Notes": "notes", "Topics": "topics",
                    }
                    if doc_action != "Chat about it" and st.button("🚀 Process"):
                        with st.spinner(f"Processing: {doc_action}..."):
                            result = process_document(content, uploaded_file.name, action_map[doc_action])
                        st.markdown(result)

                elif file_type == "error":
                    st.error(content)

with voice_col:
    with st.popover("🎙️", use_container_width=True, help="Voice input"):
        st.markdown("**🎙️ Record your message:**")
        audio_input = st.audio_input("Record", key="voice_chat", label_visibility="collapsed")
        if audio_input:
            # Only transcribe if we haven't already processed this audio
            audio_bytes = audio_input.read()
            audio_hash = hash(audio_bytes[:100])  # quick fingerprint
            if st.session_state.get("last_audio_hash") != audio_hash:
                st.session_state.last_audio_hash = audio_hash
                with st.spinner("Transcribing..."):
                    transcript = transcribe_audio(audio_bytes, "recording.wav")
                if transcript and not transcript.startswith("Couldn't"):
                    st.session_state.voice_transcript = transcript
                else:
                    st.error(transcript)

        # Show editable transcript if available
        if st.session_state.get("voice_transcript"):
            st.markdown("---")
            st.markdown("**✏️ Edit before sending:**")
            edited = st.text_area(
                "Edit transcript",
                value=st.session_state.voice_transcript,
                key="voice_edit",
                height=150,
                label_visibility="collapsed",
            )
            if st.button("📨 Send Message", key="voice_send", type="primary", use_container_width=True):
                st.session_state.pending_prompt = edited
                st.session_state.voice_transcript = None
                st.rerun()

# Process pending voice/welcome prompt BEFORE chat_input renders
voice_pending = st.session_state.pending_prompt

# The actual chat input
user_input = st.chat_input("Talk to Raven — ask anything...")

# Voice/welcome prompt takes priority if no typed input
if not user_input and voice_pending:
    user_input = voice_pending
    st.session_state.pending_prompt = None
elif user_input:
    # User typed something — clear any pending prompt
    st.session_state.pending_prompt = None

if user_input:
    user_input = user_input.strip()
    if not user_input:
        st.stop()
    if len(user_input) > 5000:
        st.warning("Message too long — keep under 5000 characters.")
        st.stop()

    conv = get_conv()
    messages = conv.get("messages", [])
    recent_context = get_recent_context()

    # ── Build extra context ──────────────────────────────────────────────────
    extra_context = ""
    prefs = st.session_state.preferences
    pref_parts = []
    if prefs.get("name"):
        pref_parts.append(f"The user's name is {prefs['name']}. Address them by name occasionally.")
    if prefs.get("tone") and prefs["tone"] != "balanced":
        pref_parts.append(f"The user prefers a {prefs['tone']} response style.")
    if pref_parts:
        extra_context = "User preferences: " + " ".join(pref_parts)

    upload_ctx = st.session_state.uploaded_context

    if upload_ctx:
        if upload_ctx["type"] == "image":
            # Image(s) — use vision model directly
            with st.chat_message("user"):
                st.markdown(user_input)
                # Show thumbnail previews of attached images
                if isinstance(upload_ctx["content"], list):
                    img_cols = st.columns(min(len(upload_ctx["content"]), 5))
                    for i, img_b in enumerate(upload_ctx["content"][:5]):
                        with img_cols[i]:
                            st.image(img_b, width=80)
            conv["messages"].append({"role": "user", "content": user_input, "emotion": "neutral", "intent": "question"})

            with st.chat_message("assistant"):
                img_count = len(upload_ctx["content"]) if isinstance(upload_ctx["content"], list) else 1
                with st.status("🐦‍⬛ Raven is thinking...", expanded=True) as status:
                    st.write(f"🎨 Analyzing {img_count} image{'s' if img_count > 1 else ''}...")
                    response = analyze_image(upload_ctx["content"], user_input)
                    status.update(label="Done!", state="complete")
                st.markdown(response)
                render_word_count(response)

            conv["messages"].append({"role": "assistant", "content": response})
            st.session_state.uploaded_context = None
            st.session_state.upload_key += 1
            if conv.get("title") == "New Chat":
                conv["title"] = generate_title(user_input)
            save_conversations()
            st.rerun()

        elif upload_ctx["type"] == "audio":
            extra_context += f"\n\n[Transcribed audio from {upload_ctx['name']}]:\n{upload_ctx['content'][:8000]}"
        elif upload_ctx["type"] == "data":
            extra_context += f"\n\n[Data from {upload_ctx['name']}]:\n{upload_ctx['content'][:8000]}"
        elif upload_ctx["type"] == "document":
            extra_context += f"\n\n[Document: {upload_ctx['name']}]:\n{upload_ctx['content'][:8000]}"

    # ── Web search ───────────────────────────────────────────────────────────
    searched_web = False
    try:
        from web_search import should_search, search_web
        preliminary_intent = detect_intent(user_input, "neutral", recent_context)
        if should_search(user_input, preliminary_intent):
            web_context, _ = search_web(user_input)
            if web_context:
                extra_context += f"\n\n{web_context}"
                searched_web = True
    except ImportError:
        pass

    # ── Emotion & intent ─────────────────────────────────────────────────────
    primary_emotion, _ = detect_emotion(user_input, recent_context, conv.get("emotion_log", []))
    intent = detect_intent(user_input, primary_emotion, recent_context)

    # Crisis continuation: only if the previous user message had actual crisis keywords
    direct_crisis = detect_crisis(user_input)
    prev_user_msgs = [m for m in messages if m.get("role") == "user"]
    recent_crisis = (
        len(prev_user_msgs) > 0
        and prev_user_msgs[-1].get("intent") == "crisis"
        and detect_crisis(prev_user_msgs[-1].get("content", ""))
    )

    if direct_crisis or recent_crisis:
        primary_emotion = "sad"
        intent = "crisis"

    conv["current_emotion"] = primary_emotion
    conv["current_intent"] = intent
    conv["messages"].append({
        "role": "user", "content": user_input,
        "emotion": primary_emotion, "intent": intent,
    })
    conv["emotion_log"].append((user_input, primary_emotion))

    # ── Display user message ─────────────────────────────────────────────────
    with st.chat_message("user"):
        render_badges(primary_emotion, intent)
        st.markdown(user_input)

    # ── Stream response ──────────────────────────────────────────────────────
    with st.chat_message("assistant"):
        if intent == "crisis" and direct_crisis:
            # First crisis detection (keywords found) — show helplines directly
            response = ("I can hear that you're going through something really difficult right now.\n"
                        "You don't have to face this alone. Please reach out to someone who can help:\n\n"
                        "🆘 **iCall (India):** 9152987821\n"
                        "🆘 **Vandrevala Foundation:** 1860-2662-345 (24/7)\n"
                        "🆘 **AASRA:** 9820466726\n"
                        "🆘 **International:** https://www.iasp.info/resources/Crisis_Centres/\n\n"
                        "You matter. I'm here to talk if you need me. 💙")
            st.markdown(response)
        elif intent == "crisis" and recent_crisis:
            # Crisis continuation — send to Groq with gentle crisis-aware prompt
            extra_context += ("\n\n[IMPORTANT CONTEXT: The user recently expressed crisis-level distress. "
                              "Helpline numbers have ALREADY been provided. Do NOT repeat helpline numbers. "
                              "Instead, be deeply empathetic, warm, and supportive. Acknowledge their feelings, "
                              "gently encourage them, and remind them they are not alone. Do NOT say 'I cannot assist with self-harm' "
                              "or any cold safety refusal. Be a caring friend.]")
            with st.status("🐦‍⬛ Raven is thinking...", expanded=True) as status:
                st.write(f"🧠 Emotion: **{primary_emotion}**")
                st.write(f"🎯 Intent: **{intent}**")
                status.update(label="Streaming response...", state="running")
            try:
                clean_msgs = [
                    {"role": m["role"], "content": m["content"]}
                    for m in conv["messages"][:-1]
                    if m.get("role") in ["user", "assistant"] and m.get("content")
                ]
                stream = get_response_stream(user_input, primary_emotion, clean_msgs, extra_context)
                response = st.write_stream(stream)
            except Exception:
                response = ("I hear you, and I want you to know that your feelings are valid. "
                            "You don't have to go through this alone. I'm right here with you. 💙")
                st.markdown(response)
        else:
            with st.status("🐦‍⬛ Raven is thinking...", expanded=True) as status:
                st.write(f"🧠 Emotion: **{primary_emotion}**")
                st.write(f"🎯 Intent: **{intent}**")
                if searched_web:
                    st.write("🌐 Searched the web...")
                if upload_ctx:
                    st.write(f"📎 Using: {upload_ctx.get('name', 'file')}")
                status.update(label="Streaming response...", state="running")

            try:
                clean_msgs = [
                    {"role": m["role"], "content": m["content"]}
                    for m in conv["messages"][:-1]
                    if m.get("role") in ["user", "assistant"] and m.get("content")
                ]
                stream = get_response_stream(user_input, primary_emotion, clean_msgs, extra_context)
                response = st.write_stream(stream)
            except Exception:
                response = "I ran into a small issue. Could you try again?"
                st.markdown(response)

        render_word_count(response)

    conv["messages"].append({"role": "assistant", "content": response})

    # Clear attachment after it's been used in a message
    if st.session_state.uploaded_context:
        st.session_state.uploaded_context = None
        st.session_state.upload_key += 1

    # Auto-title
    if conv.get("title") == "New Chat" and len([m for m in conv["messages"] if m["role"] == "user"]) == 1:
        conv["title"] = generate_title(user_input)

    save_conversations()
    st.rerun()
