import streamlit as st
import os
import time
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import json
from datetime import datetime, timedelta
import uuid

# Page config for better appearance
st.set_page_config(
    page_title="Perplexity AI Research Assistant",
    page_icon="🔍",
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

# Function to save all chats
def save_all_chats():
    if not os.path.exists("chats"):
        os.makedirs("chats")
    
    for chat_id, chat_data in st.session_state.chats.items():
        with open(f"chats/{chat_id}.json", "w") as f:
            json.dump(chat_data, f)

# Function to load all saved chats
def load_all_chats():
    chats = {}
    if not os.path.exists("chats"):
        os.makedirs("chats")
        return chats
    
    for filename in os.listdir("chats"):
        if filename.endswith(".json"):
            chat_id = filename.split(".")[0]
            try:
                with open(f"chats/{chat_id}.json", "r") as f:
                    chats[chat_id] = json.load(f)
            except:
                pass
    return chats

# Function to create a new chat
def create_new_chat():
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {
        "title": f"New Chat {len(st.session_state.chats) + 1}",
        "messages": [],
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    st.session_state.current_chat_id = chat_id
    return chat_id

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
    
    # Initialize chat system in session state
    if "chats" not in st.session_state:
        st.session_state.chats = load_all_chats()
        
    if "current_chat_id" not in st.session_state or st.session_state.current_chat_id not in st.session_state.chats:
        # If no chats exist, create a new one
        if not st.session_state.chats:
            create_new_chat()
        else:
            # Use the most recent chat
            recent_chat_id = max(st.session_state.chats.keys(), 
                               key=lambda k: st.session_state.chats[k].get("updated_at", ""))
            st.session_state.current_chat_id = recent_chat_id
            
    # For convenience, get current chat messages
    current_chat = st.session_state.chats[st.session_state.current_chat_id]
    current_messages = current_chat["messages"]
    
    # Show current chat title in main area
    st.subheader(f"💬 {current_chat['title']}")
    
    # Display current chat history
    for idx, message in enumerate(current_messages):
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
        # Update chat title based on first message if it's a new chat
        if len(current_messages) == 0:
            current_chat["title"] = prompt[:30] + ("..." if len(prompt) > 30 else "")
        
        # Add user message to chat history
        current_messages.append({"role": "user", "content": prompt})
        current_chat["updated_at"] = datetime.now().isoformat()
        
        # Save chats after every message
        save_all_chats()
        
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
            current_messages.append({
                "role": "assistant",
                "content": response_data['content'],
                "metadata": {
                    "time_taken": response_data['time_taken'],
                    "timestamp": datetime.now().isoformat(),
                    "model": "sonar-deep-research"
                }
            })
            
            # Update chat and save
            current_chat["updated_at"] = datetime.now().isoformat()
            save_all_chats()

if __name__ == "__main__":
    main()
