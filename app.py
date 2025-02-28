import streamlit as st
import os
from google.colab import userdata
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# Set up the Perplexity API key
os.environ["PERPLEXITY_API_KEY"] = userdata.get('PERPLEXITY_API_KEY')

# Initialize ChatOpenAI with Perplexity's endpoint and sonar-deep-research model
chat = ChatOpenAI(
    openai_api_key=os.environ["PERPLEXITY_API_KEY"],
    openai_api_base="https://api.perplexity.ai",
    model="sonar-deep-research",
    max_tokens=2000
)

# Streamlit app
def main():
    st.title("Perplexity AI Chat with Sonar Deep Research")
    
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
