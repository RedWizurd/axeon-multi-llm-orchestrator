import streamlit as st
from pathlib import Path
from axeon_orchestrator import AxeonOrchestrator

st.title("Axeon Local Chat")

# Load orchestrator
config_path = Path("config.json")
orchestrator = AxeonOrchestrator(config_path)

# Session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = orchestrator.handle_turn(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
