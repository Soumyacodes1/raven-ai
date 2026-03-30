import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from emotion_engine import detect_emotion, detect_intent, detect_crisis
from responder import search_and_respond, get_response

st.set_page_config(
    page_title="Raven AI",
    page_icon="🐦‍⬛",
    layout="wide"
)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "emotion_log" not in st.session_state:
    st.session_state.emotion_log = []
if "current_emotion" not in st.session_state:
    st.session_state.current_emotion = "neutral"
if "current_intent" not in st.session_state:
    st.session_state.current_intent = "casual"

# ── Style maps ────────────────────────────────────────────────────────────────
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
    "happy":    "#F59E0B",
    "sad":      "#60A5FA",
    "anxious":  "#A78BFA",
    "angry":    "#F87171",
    "confused": "#34D399",
    "neutral":  "#9CA3AF",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def build_emotion_chart(emotion_log):
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
    ax.set_xticks(x_vals)
    ax.set_xticklabels([str(i) for i in x_vals], fontsize=7, color="#9CA3AF")
    ax.set_xlabel("Message", fontsize=7, color="#9CA3AF")
    ax.tick_params(colors="#9CA3AF")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d2d3a")
    ax.grid(axis="y", color="#2d2d3a", linewidth=0.5, linestyle="--")
    plt.tight_layout(pad=0.5)
    return fig


def get_recent_context():
    user_msgs = [m["content"] for m in st.session_state.messages
                 if m.get("role") == "user"][-2:]
    return " | ".join(user_msgs) if user_msgs else ""


def render_badges(emotion, intent):
    s = EMOTION_STYLE.get(emotion, EMOTION_STYLE["neutral"])
    i = INTENT_STYLE.get(intent, INTENT_STYLE["casual"])
    st.markdown(
        f"<span style='background:{s['color']}22;padding:2px 8px;"
        f"border-radius:10px;font-size:12px;color:{s['color']};margin-right:6px'>"
        f"{s['emoji']} {emotion}</span>"
        f"<span style='background:#1e1e2e;padding:2px 8px;"
        f"border-radius:10px;font-size:12px;color:#9CA3AF'>"
        f"{i['emoji']} {i['label']}</span>",
        unsafe_allow_html=True
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🐦‍⬛ Raven AI")
    st.markdown("*Thought. Memory. Empathy.*")
    st.divider()

    emotion = st.session_state.current_emotion
    style   = EMOTION_STYLE.get(emotion, EMOTION_STYLE["neutral"])
    color, emoji, label = style["color"], style["emoji"], emotion.capitalize()

    st.markdown("### Current mood detected")
    st.markdown(
        f"<div style='background:#1e1e2e;padding:12px;border-radius:10px;"
        f"border-left:4px solid {color};font-size:18px;'>"
        f"{emoji} <b style='color:{color}'>{label}</b></div>",
        unsafe_allow_html=True
    )
    st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

    intent      = st.session_state.current_intent
    intent_info = INTENT_STYLE.get(intent, INTENT_STYLE["casual"])
    st.markdown(
        f"<div style='background:#1e1e2e;padding:10px;border-radius:10px;"
        f"border-left:4px solid #6B7280;font-size:14px;'>"
        f"{intent_info['emoji']} <b style='color:#9CA3AF'>Intent:</b> "
        f"<span style='color:#E5E7EB'>{intent_info['label']}</span></div>",
        unsafe_allow_html=True
    )

    st.divider()

    if len(st.session_state.emotion_log) >= 2:
        st.markdown("### Emotion trend")
        fig = build_emotion_chart(st.session_state.emotion_log)
        if fig:
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
    elif st.session_state.emotion_log:
        st.markdown("### Emotion journey")
        for msg_preview, emo in st.session_state.emotion_log[-8:]:
            s = EMOTION_STYLE.get(emo, EMOTION_STYLE["neutral"])
            st.markdown(
                f"<div style='font-size:12px;padding:4px 8px;margin:2px 0;"
                f"border-radius:6px;background:#1e1e2e;"
                f"border-left:3px solid {s['color']}'>"
                f"{s['emoji']} {msg_preview[:30]}...</div>",
                unsafe_allow_html=True
            )

    st.divider()

    if st.button("🗑️ Clear conversation"):
        st.session_state.messages      = []
        st.session_state.emotion_log   = []
        st.session_state.current_emotion = "neutral"
        st.session_state.current_intent  = "casual"
        st.rerun()

    if st.session_state.messages:
        if st.button("📋 Session summary"):
            emotion_trend = [e for _, e in st.session_state.emotion_log]
            summary_prompt = (
                f"Summarise this conversation in 3 sentences. "
                f"The actual detected emotions in order were: {', '.join(emotion_trend)}. "
                f"Base your summary strictly on these emotions — do not invent a positive ending if not supported."
            )
            clean_msgs = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
                if m.get("role") in ["user", "assistant"] and m.get("content")
            ]
            summary = get_response(summary_prompt, st.session_state.current_emotion, clean_msgs)
            st.info(summary)

        if st.button("📥 Export chat"):
            export_data = [
                {
                    "role":    m.get("role", ""),
                    "content": m.get("content", ""),
                    "emotion": m.get("emotion", ""),
                    "intent":  m.get("intent", ""),
                }
                for m in st.session_state.messages
            ]
            csv = pd.DataFrame(export_data).to_csv(index=False)
            st.download_button("Download CSV", csv, "raven_conversation.csv", "text/csv")

# ── Main chat ─────────────────────────────────────────────────────────────────
st.markdown("## 🐦‍⬛ Raven")
st.markdown("*An emotionally aware AI — senses how you feel, responds accordingly*")
st.divider()

for msg in st.session_state.messages:
    role    = msg.get("role")
    content = msg.get("content", "")
    with st.chat_message(role):
        if role == "user":
            render_badges(msg.get("emotion", "neutral"), msg.get("intent", "casual"))
        st.markdown(content)

if user_input := st.chat_input("Talk to Raven..."):
    user_input = user_input.strip()
    if not user_input:
        st.warning("Please type a message first.")
        st.stop()
    if len(user_input) > 2000:
        st.warning("Message too long — please keep it under 2000 characters.")
        st.stop()

    recent_context  = get_recent_context()

    # HF is PRIMARY for emotion — Groq 8B is FALLBACK
    primary_emotion, _ = detect_emotion(
        user_input, recent_context, st.session_state.emotion_log
    )
    # Groq 8B for intent
    intent = detect_intent(user_input, primary_emotion, recent_context)

    # Crisis override — if crisis keywords detected, force emotion & intent
    if detect_crisis(user_input):
        primary_emotion = "sad"
        intent = "crisis"

    st.session_state.current_emotion = primary_emotion
    st.session_state.current_intent  = intent
    st.session_state.messages.append({
        "role": "user", "content": user_input,
        "emotion": primary_emotion, "intent": intent
    })
    st.session_state.emotion_log.append((user_input, primary_emotion))

    with st.chat_message("user"):
        render_badges(primary_emotion, intent)
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Raven is thinking..."):
            try:
                clean_msgs = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                    if m.get("role") in ["user", "assistant"] and m.get("content")
                ]
                # Groq 70B is PRIMARY for responses — 8B is FALLBACK
                response, _ = search_and_respond(user_input, primary_emotion, clean_msgs)
            except Exception:
                response = "I ran into a small issue there. Could you try again? I'm here and ready to help."
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
