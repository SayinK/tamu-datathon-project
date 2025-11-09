# Gemini_Ai.py — safe Gemini chat component
import streamlit as st
from google import genai

client = genai.Client()  # Make sure GEMINI_API_KEY is set

def render_gemini_chat():
    # ---------- SESSION STATE ----------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_open" not in st.session_state:
        st.session_state.chat_open = False
    if "chat_input" not in st.session_state:
        st.session_state.chat_input = ""

    # ---------- CHAT TOGGLE BUTTON ----------
    if st.button("💬 Chat"):
        st.session_state.chat_open = not st.session_state.chat_open

    # ---------- SEND MESSAGE FUNCTION ----------
    def send_message():
        msg = st.session_state.chat_input.strip()
        if not msg:
            return
        st.session_state.chat_input = ""  # clear input
        st.session_state.chat_history.append(("user", msg))
        st.session_state.chat_history.append(("Gemini", "*typing...*"))

        # Call Gemini API synchronously
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=msg
            )
            # Replace the "*typing...*" placeholder with actual response
            st.session_state.chat_history[-1] = ("Gemini", response.text)
        except Exception as e:
            st.session_state.chat_history[-1] = ("Gemini", f"Error: {e}")

    # ---------- CHAT BOX ----------
    if st.session_state.chat_open:
        # Display chat in the right sidebar
        st.sidebar.markdown("### 💬 Gemini Chat")

        # Display chat history
        for role, text in st.session_state.chat_history:
            st.sidebar.markdown(f"**{role}:** {text}")

        # Input box
        st.sidebar.text_input("Type your message:", key="chat_input", on_change=send_message)
