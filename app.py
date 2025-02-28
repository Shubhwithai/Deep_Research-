import streamlit as st
import time
from langchain_community.chat_models import ChatOpenAI  # Updated import
from langchain.schema import HumanMessage
import json
from datetime import datetime
from typing import Dict, List, Optional

# Page config for better appearance
st.set_page_config(
    page_title="Perplexity AI Research Assistant",
    page_icon=":mag:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
CHAT_HISTORY_FILE = "chat_history.json"
MODEL_NAME = "sonar-deep-research"
API_BASE_URL = "https://api.perplexity.ai"

# Custom CSS for better UI
def apply_custom_css():
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

# Function to initialize the chat model
def initialize_chat_model(api_key: str) -> ChatOpenAI:
    """Initialize the ChatOpenAI model with the provided API key."""
    try:
        return ChatOpenAI(
            openai_api_key=api_key,
            openai_api_base=API_BASE_URL,
            model=MODEL_NAME,
            max_tokens=2000
        )
    except Exception as e:
        st.error(f"Failed to initialize chat model: {str(e)}")
        return None

# Function to save chat history
def save_chat_history(messages: List[Dict], filename: str = CHAT_HISTORY_FILE) -> bool:
    """Save chat history to a JSON file."""
    try:
        with open(filename, "w") as f:
            json.dump(messages, f, indent=4)
        return True
    except Exception as e:
        st.error(f"Failed to save chat history: {str(e)}")
        return False

# Function to load chat history
def load_chat_history(filename: str = CHAT_HISTORY_FILE) -> List[Dict]:
    """Load chat history from a JSON file."""
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except Exception as e:
        st.error(f"Failed to load chat history: {str(e)}")
        return []

# Main Streamlit app
def main():
    apply_custom_css()

    # App title and description
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title(":mag: Perplexity AI Research Assistant")
    with col2:
        current_time = datetime.now().strftime("%b %d, %Y")
        st.markdown(f"<div style='text-align: right; padding-top: 1rem;'>{current_time}</div>", unsafe_allow_html=True)

    st.markdown("Powered by Sonar Deep Research model - Ask any research question to get comprehensive answers with citations.")

    # Sidebar configuration
    api_key = st.sidebar.text_input("Enter your Perplexity API Key", type="password")
    if api_key:
        st.session_state.api_key = api_key
        st.sidebar.success("API Key saved!")

    # Check if API key is provided
    if not api_key:
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
    chat_model = initialize_chat_model(api_key)
    if chat_model is None:
        st.error("Failed to initialize chat model. Please check your API key and try again.")
        return

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = load_chat_history()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("What would you like to research today?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            thinking_placeholder = st.empty()  # Placeholder for thinking process
            message_placeholder = st.empty()   # Placeholder for the actual message
            full_response = ""

            # Simulate streaming response (replace `chat.stream` with your actual streaming logic)
            for chunk in simulate_streaming_response(chat_model, prompt):
                if chunk.content:
                    full_response += chunk.content

                    # Only update message placeholder during streaming if no thinking tags detected yet
                    if "<think>" not in full_response:
                        message_placeholder.write(full_response)

            # After streaming is complete, check for thinking tags
            if "<think>" in full_response and "</think>" in full_response:
                # Extract thinking content
                think_start = full_response.index("<think>")
                think_end = full_response.index("</think>") + len("</think>")
                thinking = full_response[think_start:think_end]

                # Extract actual response
                actual_response = full_response[think_end:].strip()

                # First display thinking in expandable section
                with thinking_placeholder:
                    with st.expander("Show AI thinking process"):
                        st.write(thinking)

                # Then display actual response
                message_placeholder.write(actual_response)
            else:
                # If no thinking tags, clear thinking placeholder and show full response
                thinking_placeholder.empty()
                message_placeholder.write(full_response)

        # Save the assistant's response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        save_chat_history(st.session_state.messages)

# Simulate streaming response (replace with actual streaming logic)
def simulate_streaming_response(chat_model, prompt):
    """Simulate a streaming response for demonstration purposes."""
    full_response = (
        "This is the AI's response. "
        "<think>This is the AI's thinking process. It can include reasoning, steps, or intermediate thoughts.</think> "
        "Here is the final part of the response."
    )
    for word in full_response.split(" "):
        time.sleep(0.1)  # Simulate delay
        yield HumanMessage(content=word + " ")

if __name__ == "__main__":
    main()
