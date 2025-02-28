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
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to initialize the chat model
def initialize_chat_model(api_key, model="sonar-deep-research", temperature=0.7):
    return ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base="https://api.perplexity.ai",
        model=model,
        max_tokens=2000,
        temperature=temperature
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
def process_chat(prompt, system_prompt=""):
    try:
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
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
            background-color: #1a1a1a;
            border-radius: 10px;
            padding: 0.5rem;
            margin-bottom: 1rem;
        }
        .user-message {
            background-color: #2d3748;
            color: white;
            padding: 1rem;
            border-radius: 8px;
        }
        .assistant-message {
            background-color: #1e1e1e;
            color: #f0f0f0;
            padding: 1rem;
            border-radius: 8px;
        }
        .thinking-box {
            background-color: #1a1a1a;
            color: #a0a0a0;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .searching-text {
            color: #4da6ff;
            font-weight: bold;
        }
        .meta-info {
            font-size: 0.8rem;
            color: #888;
            text-align: right;
        }
        .sidebar .block-container {
            background-color: #1a1a1a;
        }
        </style>
    """, unsafe_allow_html=True)

    # App title and description
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title("🔍 Perplexity AI Research Assistant")
    with col2:
        current_time = datetime.now().strftime("%b %d, %Y")
        st.markdown(f"<div style='text-align: right; padding-top: 1rem;'>{current_time}</div>", unsafe_allow_html=True)
    
    st.markdown("Powered by Sonar Deep Research model - Ask any research question to get comprehensive answers with citations.")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key handling
        api_key = st.text_input("Enter your Perplexity API Key", type="password")
        if api_key:
            st.session_state.api_key = api_key
            st.success("API Key saved!")
        
        st.divider()
        
        # Model selection and parameters
        st.subheader("Model Settings")
        model_option = st.selectbox(
            "Select Model",
            options=["sonar-deep-research", "sonar-medium-online", "sonar-small-online"],
            index=0
        )
        
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1,
                               help="Higher values make output more creative, lower values more deterministic")
        
        st.divider()
        
        # System prompt for model behavior
        st.subheader("Assistant Behavior")
        system_prompt = st.text_area(
            "System Prompt (Optional)",
            value="You are a helpful research assistant. Provide comprehensive, accurate, and well-cited answers.",
            help="Instructions that guide how the assistant responds"
        )
        
        st.divider()
        
        # Session management
        st.subheader("Session Management")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("Clear Chat"):
                st.session_state.messages = []
                st.experimental_rerun()
        with col2:
            if st.button("Save Chat"):
                filename = save_chat_history()
                st.success(f"Saved to {filename}")
        
        if st.button("Load Previous Chat"):
            st.session_state.messages = load_chat_history()
            st.experimental_rerun()
    
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
    
    # Initialize chat model with provided API key and settings
    global chat
    chat = initialize_chat_model(
        st.session_state.api_key,
        model=model_option,
        temperature=temperature
    )
    
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for idx, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(f"<div class='user-message'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            with st.chat_message("assistant", avatar="🔍"):
                st.markdown(f"<div class='assistant-message'>{message['content']}</div>", unsafe_allow_html=True)
                if "metadata" in message:
                    st.markdown(f"<div class='meta-info'>Response time: {message['metadata']['time_taken']}s</div>", 
                                unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("What would you like to research today?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.markdown(f"<div class='user-message'>{prompt}</div>", unsafe_allow_html=True)
        
        # Display assistant response with thinking animation
        with st.chat_message("assistant", avatar="🔍"):
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
            response_data = process_chat(prompt, system_prompt)
            
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
                    "model": model_option
                }
            })

if __name__ == "__main__":
    main()
