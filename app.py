import streamlit as st
import time
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import json
from datetime import datetime
from typing import Dict, List, Optional

# Page configuration
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

# Custom CSS for premium UI
def apply_custom_css():
    st.markdown("""
        <style>
        /* Main container adjustments */
        .main .block-container {
            padding-top: 1rem;
            max-width: 90vw;
        }
        
        /* Chat message styling */
        .stChatMessage {
            background: transparent !important;
            box-shadow: none !important;
            padding: 0.5rem 1rem;
        }
        
        /* User message styling */
        .user-message {
            background: linear-gradient(135deg, #6366f1 0%, #4338ca 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 20px 20px 5px 20px;
            margin: 0.5rem 0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* Assistant message styling */
        .assistant-message {
            background: #1f2937;
            color: #f3f4f6;
            padding: 1.5rem;
            border-radius: 20px 20px 20px 5px;
            margin: 0.5rem 0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        /* Thinking animation */
        .thinking-box {
            background: #1f2937;
            color: #9ca3af;
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            border: 1px solid rgba(255, 255, 255, 0.05);
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 0.9; }
            50% { opacity: 0.7; }
            100% { opacity: 0.9; }
        }
        
        /* Metadata styling */
        .meta-info {
            font-size: 0.75rem;
            color: #6b7280;
            text-align: right;
            margin-top: 0.5rem;
            font-family: 'Courier New', monospace;
        }
        
        /* Sidebar enhancements */
        .sidebar .block-container {
            background: #111827;
            padding: 1rem;
            border-radius: 15px;
        }
        
        /* Input field styling */
        .stTextInput input {
            background: #1f2937 !important;
            color: white !important;
            border: 1px solid #374151 !important;
        }
        
        /* Button styling */
        .stButton button {
            background: linear-gradient(135deg, #6366f1 0%, #4338ca 100%);
            color: white !important;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            width: 100%;
            transition: all 0.3s ease;
        }
        
        .stButton button:hover {
            opacity: 0.9;
            transform: translateY(-1px);
        }
        
        /* Gradient text for title */
        .gradient-text {
            background: linear-gradient(45deg, #6366f1, #ec4899);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            font-weight: 800;
        }
        </style>
    """, unsafe_allow_html=True)

def initialize_chat_model(api_key: str) -> ChatOpenAI:
    """Initialize the ChatOpenAI model with the provided API key."""
    return ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base=API_BASE_URL,
        model=MODEL_NAME,
        max_tokens=2000
    )

def save_chat_history(messages: List[Dict], filename: str = CHAT_HISTORY_FILE) -> str:
    """Save chat history to a JSON file."""
    try:
        with open(filename, "w") as f:
            json.dump(messages, f, indent=4)
        return filename
    except Exception as e:
        st.error(f"Failed to save chat history: {str(e)}")
        return ""

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

def process_chat(prompt: str, chat_model: ChatOpenAI) -> Dict:
    """Process the user's prompt and return the assistant's response."""
    try:
        messages = [HumanMessage(content=prompt)]

        # Get response from Perplexity
        with st.spinner(""):
            start_time = time.time()
            response = chat_model(messages)
            end_time = time.time()

        return {
            "content": response.content,
            "time_taken": round(end_time - start_time, 2)
        }
    except Exception as e:
        st.error(f"Error processing chat: {str(e)}")
        return {"content": f"I encountered an error: {str(e)}", "time_taken": 0}

def display_chat_history(messages: List[Dict]):
    """Display the chat history in the Streamlit app."""
    for message in messages:
        if message["role"] == "user":
            with st.chat_message("user", avatar="🧑💻"):
                st.markdown(f"<div class='user-message'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(f"<div class='assistant-message'>{message['content']}</div>", unsafe_allow_html=True)
                if "metadata" in message:
                    st.markdown(f"<div class='meta-info'>⏱️ {message['metadata']['time_taken']}s | 🤖 {MODEL_NAME}</div>",
                                unsafe_allow_html=True)

def sidebar_configuration() -> Optional[str]:
    """Configure the sidebar and return the API key if provided."""
    st.sidebar.header("⚙️ Configuration")
    api_key = st.sidebar.text_input("Enter your Perplexity API Key", type="password")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠️ Actions")
    if st.sidebar.button("🚀 Start New Chat", key="start_new_chat"):
        st.session_state.messages = []
        st.rerun()
        
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
        <div style="color: #6b7280; font-size: 0.9rem;">
        <strong>Built with ❤️ by</strong><br>
        <a href="https://buildfastwithai.com/genai-course" style="color: #818cf8; text-decoration: none;">Build Fast with AI</a>
        </div>
    """, unsafe_allow_html=True)

    return api_key if api_key else None

def main():
    apply_custom_css()

    # App header
    col1, col2 = st.columns([6, 1])
    with col1:
        st.markdown("<h1 class='gradient-text'>🔍 Perplexity AI Research Assistant</h1>", unsafe_allow_html=True)
    with col2:
        current_time = datetime.now().strftime("%b %d, %Y")
        st.markdown(f"<div style='text-align: right; color: #6b7280; padding-top: 1rem;'>{current_time}</div>", 
                    unsafe_allow_html=True)

    st.markdown("""
        <div style="color: #9ca3af; margin-bottom: 2rem;">
        Powered by <span style="color: #818cf8;">Sonar Deep Research</span> model - 
        Get comprehensive, citation-backed answers to complex research questions.
        </div>
    """, unsafe_allow_html=True)

    # Sidebar configuration
    api_key = sidebar_configuration()

    if not api_key:
        st.warning("🔑 Please enter your Perplexity API key in the sidebar to continue.")
        st.markdown("""
            <div style="background: #1f2937; padding: 1.5rem; border-radius: 10px; color: #9ca3af;">
            <h3 style="color: #f3f4f6;">How to get started:</h3>
            <ol>
                <li>Create an account on <a href="https://www.perplexity.ai/" style="color: #818cf8;">Perplexity AI</a></li>
                <li>Navigate to your account settings</li>
                <li>Generate a new API key</li>
                <li>Copy and paste the key in the sidebar</li>
            </ol>
            </div>
        """, unsafe_allow_html=True)
        return

    # Initialize chat model
    chat_model = initialize_chat_model(api_key)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = load_chat_history()

    # Display chat history
    display_chat_history(st.session_state.messages)

    # Chat input
    if prompt := st.chat_input("What would you like to research today?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user", avatar="🧑💻"):
            st.markdown(f"<div class='user-message'>{prompt}</div>", unsafe_allow_html=True)

        # Process and display assistant response
        with st.chat_message("assistant", avatar="🤖"):
            response_container = st.empty()
            
            # Show thinking animation
            response_container.markdown(
                f'<div class="thinking-box">'
                f'<span style="color: #818cf8;">🔍 Researching</span><br>'
                f'<span>{prompt[:50]}{"..." if len(prompt) > 50 else ""}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

            # Get response
            response_data = process_chat(prompt, chat_model)

            # Display final response
            response_container.markdown(
                f"<div class='assistant-message'>{response_data['content']}</div>",
                unsafe_allow_html=True
            )

            # Show metadata
            st.markdown(
                f"<div class='meta-info'>⏱️ {response_data['time_taken']}s | 🤖 {MODEL_NAME}</div>",
                unsafe_allow_html=True
            )

        # Save to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_data['content'],
            "metadata": {
                "time_taken": response_data['time_taken'],
                "timestamp": datetime.now().isoformat(),
                "model": MODEL_NAME
            }
        })
        save_chat_history(st.session_state.messages)

if __name__ == "__main__":
    main()
