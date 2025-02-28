import streamlit as st
import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# Function to initialize the chat model
def initialize_chat_model(api_key):
    return ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base="https://api.perplexity.ai",
        model="sonar-deep-research",
        max_tokens=2000
    )

# Streamlit app
def main():
    st.title("Perplexity AI Chat with Sonar Deep Research")
    
    # API Key input in sidebar
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Enter your Perplexity API Key", type="password")
        
        # Store API key in session state
        if api_key:
            st.session_state.api_key = api_key
            st.success("API Key saved!")
    
    # Check if API key is provided
    if "api_key" not in st.session_state or not st.session_state.api_key:
        st.warning("Please enter your Perplexity API key in the sidebar to continue.")
        return
    
    # Initialize chat model with provided API key
    chat = initialize_chat_model(st.session_state.api_key)
    
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Create message for the model
                    messages = [HumanMessage(content=prompt)]
                    # Get response from Perplexity
                    response = chat(messages)
                    # Display response
                    st.markdown(response.content)
                    # Add to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.content
                    })
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
