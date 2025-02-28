import streamlit as st
import os
import time
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import json
from datetime import datetime
# Page config for better appearance
st.set_page_config(
    page_title="Perplexity AI Research Assistant",
    page_icon=":magnifying_glass:",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Function to initialize the chat model
def initialize_chat_model(api_key):
    return ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base="https://api.perplexity.ai",
        model="sonar-deep-research",
        max_tokens=2000
    )
# Function to save chat history
def save_chat_history(filename="chat_history.json"):
    with open(filename, "w") as f:
        json.dump(st.session_state.messages, f)
    return filename
# Function to load chat history
def load_chat_history(filename="chat_history.json"):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []
# Function to handle chat
def process_chat(prompt):
    try:
        messages = [HumanMessage(content=prompt)]
        # Get response from Perplexity
        with st.spinner(""):
            start_time = time.time()
            response = chat(messages)
            end_time = time.time()
        return {
            "content": response.content,
            "time_taken": round(end_time - start_time, 2)
        }
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return {"content": f"I encountered an error: {str(e)}", "time_taken": 0}
# Streamlit app
def main():
    # Custom CSS for better UI
    st.markdown("""
        <style>
        .main .block-container {padding-top: 2rem;}
        .stChatMessage {
            background-color: #1A1A1A;
            border-radius: 10px;
            padding: 0.5rem;
            margin-bottom: 1rem;
        }
        .user-message {
            background-color: #2D3748;
            color: white;
            padding: 1rem;
            border-radius: 8px;
        }
        .assistant-message {
            background-color: #1E1E1E;
            color: #F0F0F0;
            padding: 1rem;
            border-radius: 8px;
        }
        .thinking-box {
            background-color: #1A1A1A;
            color: #A0A0A0;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .searching-text {
            color: #4DA6FF;
            font-weight: bold;
        }
        .meta-info {
            font-size: 0.8rem;
            color: #888;
            text-align: right;
        }
        .sidebar .block-container {
            background-color: #1A1A1A;
        }
        </style>
    """, unsafe_allow_html=True)
    # App title and description
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title(":magnifying_glass: Perplexity AI Research Assistant")
    with col2:
        current_time = datetime.now().strftime("%b %d, %Y")
        st.markdown(f"<div style='text-align: right; padding-top: 1rem;'>{current_time}</div>", unsafe_allow_html=True)
    st.markdown("Powered by Sonar Deep Research model - Ask any research question to get comprehensive answers with citations.")
    # Sidebar configuration
    with st.sidebar:
        st.header(":cog: Configuration")
        # API Key handling
        api_key = st.text_input("Enter your Perplexity API Key", type="password")
        if api_key:
            st.session_state.api_key = api_key
            st.success("API Key saved!")
        st.divider()
        st.divider()
        # Session management
        st.subheader("Session Management")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("Clear Chat"):
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("Save Chat"):
                filename = save_chat_history()
                st.success(f"Saved to {filename}")
        if st.button("Load Previous Chat"):
            st.session_state.messages = load_chat_history()
            st.rerun()
    # Check if API key is provided
    if "api_key" not in st.session_state or not st.session_state.api_key:
        st.warning("Please enter your Perplexity API key in the sidebar to continue.")
        st.markdown("""
        ### How to get a Perplexity API key
        1. Create an account on [Perplexity AI](https://www.perplexity.ai/)
        2. Navigate to your account settings
        3. Generate a new API key
        4. Copy and paste the key in the sidebar
        """)
        return
    # Initialize chat model with provided API key
    global chat
    chat = initialize_chat_model(st.session_state.api_key)
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Display chat history
    for idx, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            with st.chat_message("user", avatar=":silhouette:"):
                st.markdown(f"<div class='user-message'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            with st.chat_message("assistant", avatar=":magnifying_glass:"):
                st.markdown(f"<div class='assistant-message'>{message['content']}</div>", unsafe_allow_html=True)
                if "metadata" in message:
                    st.markdown(f"<div class='meta-info'>Response time: {message['metadata']['time_taken']}s</div>",
                                unsafe_allow_html=True)
    # Chat input
    if prompt := st.chat_input("What would you like to research today?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message
        with st.chat_message("user", avatar=":silhouette:"):
            st.markdown(f"<div class='user-message'>{prompt}</div>", unsafe_allow_html=True)
        # Display assistant response with thinking animation
        with st.chat_message("assistant", avatar=":magnifying_glass:"):
            response_container = st.empty()
            # Show thinking/searching status
            response_container.markdown(
                f'<div class="thinking-box">'
                f'<span class="searching-text">Searching</span><br>'
                f'<span>{prompt[:40]}{"..." if len(prompt) > 40 else ""}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
            # Process the chat
            response_data = process_chat(prompt)
            # Display the response
            response_container.markdown(
                f"<div class='assistant-message'>{response_data['content']}</div>",
                unsafe_allow_html=True
            )
            # Show metadata about the response
            st.markdown(
                f"<div class='meta-info'>Response time: {response_data['time_taken']}s</div>",
                unsafe_allow_html=True
            )
            # Add to chat history with metadata
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_data['content'],
                "metadata": {
                    "time_taken": response_data['time_taken'],
                    "timestamp": datetime.now().isoformat(),
                    "model": "sonar-deep-research"
                }
            })
if __name__ == "__main__":
    main()
