import streamlit as st
from emotion_engine import detect_emotion
from responder import search_and_respond, get_response

st.set_page_config(
    page_title="Raven AI",
    page_icon="🐦‍⬛",
    layout="wide"
)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "emotion_log" not in st.session_state:
    st.session_state.emotion_log = []
if "current_emotion" not in st.session_state:
    st.session_state.current_emotion = "neutral"

EMOTION_STYLE = {
    "happy":    {"emoji": "😊", "color": "#F59E0B"},
    "sad":      {"emoji": "😔", "color": "#60A5FA"},
    "anxious":  {"emoji": "😰", "color": "#A78BFA"},
    "angry":    {"emoji": "😤", "color": "#F87171"},
    "confused": {"emoji": "😕", "color": "#34D399"},
    "neutral":  {"emoji": "😐", "color": "#9CA3AF"},
}

# Sidebar
with st.sidebar:
    st.markdown("# 🐦‍⬛ Raven AI")
    st.markdown("*Thought. Memory. Empathy.*")
    st.divider()

    emotion = st.session_state.current_emotion
    style = EMOTION_STYLE.get(emotion, EMOTION_STYLE["neutral"])
    color = style["color"]
    emoji = style["emoji"]
    label = emotion.capitalize()

    st.markdown("### Current mood detected")
    st.markdown(
        f"<div style='background:#1e1e2e;padding:12px;border-radius:10px;"
        f"border-left:4px solid {color};font-size:18px;'>"
        f"{emoji} <b style='color:{color}'>{label}</b></div>",
        unsafe_allow_html=True
    )

    st.divider()

    if st.session_state.emotion_log:
        st.markdown("### Emotion journey")
        for msg_preview, emo in st.session_state.emotion_log[-8:]:
            s = EMOTION_STYLE.get(emo, EMOTION_STYLE["neutral"])
            s_color = s["color"]
            s_emoji = s["emoji"]
            preview = msg_preview[:30]
            st.markdown(
                f"<div style='font-size:12px;padding:4px 8px;margin:2px 0;"
                f"border-radius:6px;background:#1e1e2e;"
                f"border-left:3px solid {s_color}'>"
                f"{s_emoji} {preview}...</div>",
                unsafe_allow_html=True
            )

    st.divider()

    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.session_state.emotion_log = []
        st.session_state.current_emotion = "neutral"
        st.rerun()

    if st.session_state.messages:
        if st.button("📋 Session summary"):
            emotions_seen = list(set([e for _, e in st.session_state.emotion_log]))
            summary_prompt = (
                f"Summarise this conversation in 3 sentences. "
                f"Emotions detected: {', '.join(emotions_seen)}. "
                f"Be warm and insightful."
            )
            clean_msgs = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
                if m.get("role") in ["user", "assistant"] and m.get("content")
            ]
            summary = get_response(
                summary_prompt,
                st.session_state.current_emotion,
                clean_msgs
            )
            st.info(summary)

# Main chat
st.markdown("## 🐦‍⬛ Raven")
st.markdown("*An emotionally aware AI — senses how you feel, responds accordingly*")
st.divider()

# Display chat history
for msg in st.session_state.messages:
    role = msg.get("role")
    content = msg.get("content", "")

    with st.chat_message(role):
        if role == "user":
            emo = msg.get("emotion", "neutral")
            s = EMOTION_STYLE.get(emo, EMOTION_STYLE["neutral"])
            s_color = s["color"]
            s_emoji = s["emoji"]
            st.markdown(
                f"<span style='background:{s_color}22;padding:2px 8px;"
                f"border-radius:10px;font-size:12px;color:{s_color}'>"
                f"{s_emoji} {emo}</span>",
                unsafe_allow_html=True
            )
        st.markdown(content)

# Chat input
if user_input := st.chat_input("Talk to Raven..."):
    primary_emotion, secondary_emotion = detect_emotion(user_input)
    st.session_state.current_emotion = primary_emotion

    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "emotion": primary_emotion
    })
    st.session_state.emotion_log.append((user_input, primary_emotion))

    with st.chat_message("user"):
        s = EMOTION_STYLE.get(primary_emotion, EMOTION_STYLE["neutral"])
        s_color = s["color"]
        s_emoji = s["emoji"]
        st.markdown(
            f"<span style='background:{s_color}22;padding:2px 8px;"
            f"border-radius:10px;font-size:12px;color:{s_color}'>"
            f"{s_emoji} {primary_emotion}</span>",
            unsafe_allow_html=True
        )
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Raven is thinking..."):
            clean_msgs = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
                if m.get("role") in ["user", "assistant"] and m.get("content")
            ]
            response, _ = search_and_respond(
                user_input, primary_emotion, clean_msgs
            )
            st.markdown(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })
    st.rerun()
