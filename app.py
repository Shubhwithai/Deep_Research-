import streamlit as st
import time
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import json
from datetime import datetime
import os
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd

# ==========================================================
# Constants and Configuration
# ==========================================================
CHAT_HISTORY_FOLDER = "chat_histories"
MODEL_NAMES = {
    "sonar-deep-research": "Deep Research",
    "sonar-medium-research": "Medium Research",
    "sonar-small-research": "Small Research",
    "mixtral-8x7b-instruct": "Mixtral 8x7B"
}
DEFAULT_MODEL = "sonar-deep-research"
API_BASE_URL = "https://api.perplexity.ai"
APP_VERSION = "1.2.0"

# ==========================================================
# Setup and Initialization
# ==========================================================
def setup_folders():
    """Create necessary folders if they don't exist."""
    os.makedirs(CHAT_HISTORY_FOLDER, exist_ok=True)

def setup_page_config():
    """Configure the Streamlit page settings."""
    st.set_page_config(
        page_title="Advanced Perplexity Research Assistant",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
def apply_custom_css():
    """Apply custom CSS for an enhanced UI experience."""
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
        .stTextInput input, .stSelectbox > div > div {
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
        
        /* Stats box */
        .stats-box {
            background: #1f2937;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        /* Citation styling */
        .citation {
            background: rgba(99, 102, 241, 0.1);
            border-left: 3px solid #6366f1;
            padding: 0.75rem;
            margin: 0.5rem 0;
            border-radius: 0 8px 8px 0;
            font-size: 0.9rem;
        }
        
        /* Chat history list */
        .history-item {
            background: #1f2937;
            padding: 0.75rem;
            border-radius: 8px;
            margin-bottom: 0.5rem;
            cursor: pointer;
            transition: all 0.2s ease;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .history-item:hover {
            background: #2d3748;
            transform: translateY(-1px);
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
            background-color: #111827;
            border-radius: 8px;
            padding: 0.2rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 40px;
            background-color: #1f2937;
            border-radius: 8px;
            color: #9ca3af;
            padding: 0 16px;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #6366f1 !important;
            color: white !important;
        }
        </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize or update session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = get_chat_sessions()
    
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = datetime.now().strftime("%Y%m%d%H%M%S")
    
    if "model_name" not in st.session_state:
        st.session_state.model_name = DEFAULT_MODEL
    
    if "total_tokens_used" not in st.session_state:
        st.session_state.total_tokens_used = 0
    
    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0

# ==========================================================
# Chat History and Session Management
# ==========================================================
def get_chat_history_path(session_id: str) -> str:
    """Get the file path for a specific chat history."""
    return os.path.join(CHAT_HISTORY_FOLDER, f"{session_id}.json")

def save_chat_history(messages: List[Dict], session_id: str) -> str:
    """Save chat history to a JSON file."""
    file_path = get_chat_history_path(session_id)
    try:
        with open(file_path, "w") as f:
            json.dump(messages, f, indent=4)
        return file_path
    except Exception as e:
        st.error(f"Failed to save chat history: {str(e)}")
        return ""

def load_chat_history(session_id: str) -> List[Dict]:
    """Load chat history from a JSON file."""
    file_path = get_chat_history_path(session_id)
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except Exception as e:
        st.error(f"Failed to load chat history: {str(e)}")
        return []

def get_chat_sessions() -> Dict[str, Dict]:
    """Get all available chat sessions with metadata."""
    sessions = {}
    try:
        for filename in os.listdir(CHAT_HISTORY_FOLDER):
            if filename.endswith(".json"):
                session_id = filename.replace(".json", "")
                try:
                    with open(os.path.join(CHAT_HISTORY_FOLDER, filename), "r") as f:
                        messages = json.load(f)
                        if messages:
                            first_message = next((m for m in messages if m["role"] == "user"), None)
                            sessions[session_id] = {
                                "id": session_id,
                                "title": first_message["content"][:50] if first_message else "Untitled Chat",
                                "timestamp": datetime.fromtimestamp(os.path.getmtime(os.path.join(CHAT_HISTORY_FOLDER, filename))),
                                "message_count": len(messages)
                            }
                except Exception:
                    # Skip corrupted files
                    continue
    except Exception as e:
        st.error(f"Error loading chat sessions: {str(e)}")
    
    # Sort sessions by timestamp (newest first)
    return dict(sorted(sessions.items(), key=lambda x: x[1]["timestamp"], reverse=True))

def create_new_session():
    """Create a new chat session."""
    st.session_state.current_session_id = datetime.now().strftime("%Y%m%d%H%M%S")
    st.session_state.messages = []
    st.session_state.chat_sessions = get_chat_sessions()
    st.rerun()

def load_session(session_id: str):
    """Load a specific chat session."""
    st.session_state.current_session_id = session_id
    st.session_state.messages = load_chat_history(session_id)
    st.rerun()

def delete_session(session_id: str):
    """Delete a chat session."""
    try:
        os.remove(get_chat_history_path(session_id))
        if st.session_state.current_session_id == session_id:
            create_new_session()
        else:
            st.session_state.chat_sessions = get_chat_sessions()
            st.rerun()
    except Exception as e:
        st.error(f"Failed to delete chat session: {str(e)}")

# ==========================================================
# Model and API Interaction
# ==========================================================
def initialize_chat_model(api_key: str, model_name: str) -> ChatOpenAI:
    """Initialize the ChatOpenAI model with the provided API key and model name."""
    return ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base=API_BASE_URL,
        model=model_name,
        max_tokens=4000,
        streaming=True,
        temperature=0.7
    )

def process_chat(prompt: str, history: List[Dict], chat_model: ChatOpenAI) -> Dict:
    """Process the user's prompt with conversation history and return the assistant's response."""
    try:
        # Convert history to langchain message format
        history_messages = []
        if history:
            for msg in history[-5:]:  # Use last 5 messages as context
                if msg["role"] == "user":
                    history_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    history_messages.append(AIMessage(content=msg["content"]))
        
        # Add current prompt
        messages = history_messages + [HumanMessage(content=prompt)]
        
        # System message to give context
        system_prompt = """You are an advanced research assistant powered by Perplexity AI. Provide thorough, accurate, and well-cited answers.
        Use markdown formatting for clarity: headers, lists, and bold text for key points.
        Always cite your sources clearly. Format citations as [1], [2], etc."""
        messages.insert(0, SystemMessage(content=system_prompt))
        
        # Get response from Perplexity
        start_time = time.time()
        
        response_container = st.empty()
        full_response = ""
        
        for chunk in chat_model.stream(messages):
            content_chunk = chunk.content
            full_response += content_chunk
            response_container.markdown(
                f"<div class='assistant-message'>{full_response}</div>",
                unsafe_allow_html=True
            )
        
        end_time = time.time()
        
        # Estimate token usage (rough estimate)
        input_tokens = sum(len(msg.content.split()) * 1.3 for msg in messages)
        output_tokens = len(full_response.split()) * 1.3
        total_tokens = int(input_tokens + output_tokens)
        
        # Update stats
        st.session_state.total_tokens_used += total_tokens
        st.session_state.total_queries += 1

        return {
            "content": full_response,
            "time_taken": round(end_time - start_time, 2),
            "tokens": total_tokens
        }
    except Exception as e:
        error_message = f"Error processing chat: {str(e)}"
        st.error(error_message)
        return {
            "content": f"I encountered an error: {str(e)}",
            "time_taken": 0,
            "tokens": 0
        }

# ==========================================================
# UI Components
# ==========================================================
def display_header():
    """Display the app header with title and current date."""
    col1, col2 = st.columns([6, 1])
    with col1:
        st.markdown("<h1 class='gradient-text'>🔍 Advanced Perplexity Research Assistant</h1>", unsafe_allow_html=True)
    with col2:
        current_time = datetime.now().strftime("%b %d, %Y")
        st.markdown(f"<div style='text-align: right; color: #6b7280; padding-top: 1rem;'>{current_time}</div>", 
                    unsafe_allow_html=True)

    st.markdown(f"""
        <div style="color: #9ca3af; margin-bottom: 2rem;">
        Powered by <span style="color: #818cf8;">{MODEL_NAMES.get(st.session_state.model_name, st.session_state.model_name)}</span> model - 
        Get comprehensive, citation-backed answers to complex research questions.
        <span style="float: right; font-size: 0.75rem;">v{APP_VERSION}</span>
        </div>
    """, unsafe_allow_html=True)

def display_chat_history(messages: List[Dict]):
    """Display the chat history in the Streamlit app."""
    for message in messages:
        if message["role"] == "user":
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(f"<div class='user-message'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(f"<div class='assistant-message'>{message['content']}</div>", unsafe_allow_html=True)
                if "metadata" in message:
                    model_name = message["metadata"].get("model", st.session_state.model_name)
                    display_name = MODEL_NAMES.get(model_name, model_name)
                    tokens = message["metadata"].get("tokens", "N/A")
                    time_taken = message["metadata"].get("time_taken", "N/A")
                    
                    st.markdown(
                        f"<div class='meta-info'>⏱️ {time_taken}s | 📝 ~{tokens} tokens | 🤖 {display_name}</div>",
                        unsafe_allow_html=True
                    )

def sidebar_configuration() -> Tuple[Optional[str], str]:
    """Configure the sidebar and return the API key and selected model."""
    st.sidebar.header("⚙️ Configuration")
    
    # API Key input
    api_key = st.sidebar.text_input("Enter your Perplexity API Key", type="password")
    
    # Save API key to session state if provided
    if api_key and "api_key" not in st.session_state:
        st.session_state.api_key = api_key
    elif "api_key" in st.session_state and not api_key:
        api_key = st.session_state.api_key
    
    # Model selection
    model_name = st.sidebar.selectbox(
        "Select Model",
        options=list(MODEL_NAMES.keys()),
        index=list(MODEL_NAMES.keys()).index(st.session_state.model_name),
        format_func=lambda x: MODEL_NAMES.get(x, x)
    )
    
    if model_name != st.session_state.model_name:
        st.session_state.model_name = model_name
        
    # Usage statistics
    if st.session_state.total_queries > 0:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Usage Statistics")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.markdown(f"<div class='stats-box'><b>Queries</b><br>{st.session_state.total_queries}</div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='stats-box'><b>Tokens</b><br>{st.session_state.total_tokens_used}</div>", unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Chat sessions
    st.sidebar.subheader("💬 Chat Sessions")
    
    # New chat button
    if st.sidebar.button("🚀 Start New Chat", key="start_new_chat"):
        create_new_session()
    
    # Show existing chats
    sessions = st.session_state.chat_sessions
    for session_id, session in sessions.items():
        col1, col2 = st.sidebar.columns([5, 1])
        with col1:
            if st.button(
                f"📝 {session['title']}...",
                key=f"load_{session_id}",
                help=f"Created: {session['timestamp'].strftime('%Y-%m-%d %H:%M')}\nMessages: {session['message_count']}"
            ):
                load_session(session_id)
        with col2:
            if st.button("🗑️", key=f"delete_{session_id}", help="Delete this chat"):
                delete_session(session_id)
                
    # Credits
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
        <div style="color: #6b7280; font-size: 0.9rem;">
        <strong>Built with ❤️ by</strong><br>
        <a href="https://buildfastwithai.com/genai-course" style="color: #818cf8; text-decoration: none;">Build Fast with AI</a>
        </div>
    """, unsafe_allow_html=True)

    return api_key, model_name

def welcome_screen():
    """Display welcome information when no API key is provided."""
    st.markdown("""
        <div style="background: #1f2937; padding: 1.5rem; border-radius: 10px; color: #9ca3af;">
        <h3 style="color: #f3f4f6;">How to get started:</h3>
        <ol>
            <li>Create an account on <a href="https://www.perplexity.ai/" style="color: #818cf8;">Perplexity AI</a></li>
            <li>Navigate to your account settings</li>
            <li>Generate a new API key</li>
            <li>Copy and paste the key in the sidebar</li>
        </ol>
        <p style="margin-top: 1rem;">
        This advanced research assistant helps you get comprehensive answers with proper citations.
        Choose from different models to balance between depth and speed.
        </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style="background: #1f2937; padding: 1.5rem; border-radius: 10px; color: #9ca3af; margin-top: 1rem;">
        <h3 style="color: #f3f4f6;">Features:</h3>
        <ul>
            <li>💬 Multiple chat sessions</li>
            <li>🔄 Model switching</li>
            <li>📊 Usage tracking</li>
            <li>📑 Citation support</li>
            <li>⚡ Streaming responses</li>
        </ul>
        </div>
    """, unsafe_allow_html=True)

def display_settings_and_help():
    """Display settings and help tabs."""
    tab1, tab2 = st.tabs(["⚙️ Settings", "❓ Help"])
    
    with tab1:
        st.subheader("Response Settings")
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1,
                              help="Lower values make responses more deterministic, higher values more creative")
        
        max_tokens = st.number_input("Maximum Response Length", min_value=500, max_value=4000, value=2000, step=500,
                                   help="Maximum number of tokens in the response")
        
        st.button("Apply Settings", help="Apply these settings to future responses")
    
    with tab2:
        st.subheader("Tips for Better Results")
        st.markdown("""
        - Be specific in your questions
        - Include key terms you want researched
        - Ask for citations when you need sources
        - Use follow-up questions to dive deeper
        """)
        
        st.subheader("Model Information")
        model_info = {
            "sonar-deep-research": "Most comprehensive model with best citation ability",
            "sonar-medium-research": "Good balance of speed and depth",
            "sonar-small-research": "Fastest model for simpler questions",
            "mixtral-8x7b-instruct": "Alternative model with different strengths"
        }
        
        for model, description in model_info.items():
            st.markdown(f"**{MODEL_NAMES.get(model, model)}**: {description}")

# ==========================================================
# Main Application
# ==========================================================
def main():
    """Main application function."""
    # Initial setup
    setup_folders()
    setup_page_config()
    apply_custom_css()
    initialize_session_state()
    
    # Display header
    display_header()
    
    # Sidebar and configuration
    api_key, model_name = sidebar_configuration()
    
    # Check if API key is provided
    if not api_key:
        welcome_screen()
        return
    
    # Initialize chat model
    chat_model = initialize_chat_model(api_key, model_name)
    
    # Load current session history
    if st.session_state.messages == []:
        st.session_state.messages = load_chat_history(st.session_state.current_session_id)
    
    # Settings and help expandable section
    with st.expander("Settings & Help", expanded=False):
        display_settings_and_help()
    
    # Display chat history
    display_chat_history(st.session_state.messages)
    
    # Chat input
    if prompt := st.chat_input("What would you like to research today?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(f"<div class='user-message'>{prompt}</div>", unsafe_allow_html=True)
        
        # Process and display assistant response
        with st.chat_message("assistant", avatar="🤖"):
            response_data = process_chat(prompt, st.session_state.messages, chat_model)
            
            # Show metadata
            model_display_name = MODEL_NAMES.get(model_name, model_name)
            st.markdown(
                f"<div class='meta-info'>⏱️ {response_data['time_taken']}s | 📝 ~{response_data['tokens']} tokens | 🤖 {model_display_name}</div>",
                unsafe_allow_html=True
            )
        
        # Save to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_data['content'],
            "metadata": {
                "time_taken": response_data['time_taken'],
                "tokens": response_data['tokens'],
                "timestamp": datetime.now().isoformat(),
                "model": model_name
            }
        })
        save_chat_history(st.session_state.messages, st.session_state.current_session_id)
        
        # Update chat sessions
        st.session_state.chat_sessions = get_chat_sessions()

if __name__ == "__main__":
    main()
