
import streamlit as st

from chatbot import generate_response


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="My AI Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 My AI Chatbot")
st.caption("Powered by Ollama + LangChain")


# -----------------------------
# Initialize Session State
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    st.header("Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.divider()

    if st.button("New Chat"):

        st.session_state.messages = []
        st.rerun()


# -----------------------------
# Display Chat History
# -----------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -----------------------------
# User Input
# -----------------------------

if prompt := st.chat_input("Ask something..."):

    # Store user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                response = generate_response(
                    st.session_state.messages
                )

                st.markdown(response)

                # Store AI response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

            except Exception as e:

                st.error(f"Something went wrong: {e}")
