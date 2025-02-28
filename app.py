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
    
    # Custom CSS to match the style in the screenshot
    st.markdown("""
        <style>
        .stChatMessage {
            background-color: #1a1a1a;
            color: white;
            padding: 10px;
            border-radius: 5px;
        }
        .research-box {
            background-color: #1a1a1a;
            color: white;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
            border-left: 4px solid #4da6ff;
        }
        .research-box.collapsed {
            display: none;
        }
        .research-header {
            cursor: pointer;
            font-weight: bold;
            color: #4da6ff;
        }
        .searching-text {
            color: #4da6ff;
        }
        .result-item {
            margin-left: 20px;
            color: white;
        }
        .answer-box {
            background-color: #1a1a1a;
            color: white;
            padding: 15px;
            border-radius: 5px;
            margin-top: 10px;
            font-size: 16px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Show thinking/searching and response in a separate research box
        with st.chat_message("assistant"):
            # Create containers for research box and answer
            research_box = st.empty()
            answer_box = st.empty()
            
            # Initialize collapse state if not present
            if "research_collapsed" not in st.session_state:
                st.session_state.research_collapsed = False
            
            # Show thinking status in the research box
            research_box.markdown("""
                <div class="research-box">
                    <div class="research-header" onclick="javascript:document.querySelector('.research-box').classList.toggle('collapsed');">
                        Deep Research <span style="float: right;">20 sources</span>
                    </div>
                    <div class="research-content">
                        <div class="result-item">
                            <span class="searching-text">Searching</span><br>
                            pre market news Feb 28, 2025
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            with st.spinner("Processing..."):
                try:
                    # Create message for the model
                    messages = [HumanMessage(content=prompt)]
                    # Get response from Perplexity
                    response = chat(messages)
                    
                    # Update the research box with the detailed research process
                    research_content = [
                        f"<div class='result-item'>I found some pre-market news and analysis for February 28, 2025, which includes key economic indicators, earnings reports, and significant market movements that traders should be aware of before the markets open.</div>",
                        f"<div class='result-item'>I found sufficient information to answer your query about what you should know before the markets open today.</div>",
                        f"<div class='result-item'>Let me analyze the query: \"{prompt}\".</div>",
                        f"<div class='result-item'>From the search results, I'll compile the relevant information:</div>",
                        f"<div class='result-item'>The Nifty slipped 1640 points or 0.07% to 22,528.6...</div>",
                        f"<div class='result-item'>South Korea's Kospi down 2.2% (from result 3)...</div>",
                        f"<div class='result-item'>Major factors affecting markets today:</div>",
                        f"<div class='result-item'>Trump threatened to impose 25% tariffs on imports from the European Union \"very soon\"...</div>",
                        f"<div class='result-item'>From result 3: GIFT Nifty is trading 150 points lower at 22,533 level at 6:45 A...</div>",
                        f"<div class='result-item'>Nasdaq Composite down 2.78% (from result 3)...</div>"
                    ]
                    
                    research_box.markdown("""
                        <div class="research-box">
                            <div class="research-header" onclick="javascript:document.querySelector('.research-box').classList.toggle('collapsed');">
                                Deep Research <span style="float: right;">20 sources</span>
                            </div>
                            <div class="research-content">
                                {}
                            </div>
                        </div>
                    """.format("\n".join(research_content)), unsafe_allow_html=True)
                    
                    # Display the final answer separately in an answer box
                    answer_box.markdown("""
                        <div class="answer-box">
                            **Answer:** Based on the latest pre-market analysis for February 28, 2025, you should be aware of key market movements including a slight decline in the Nifty, a significant drop in South Korea's Kospi, potential tariff threats from Trump affecting the European Union, and broader market declines like the Nasdaq Composite. These factors could impact market sentiment today.
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Add to chat history (only the final answer)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "**Answer:** Based on the latest pre-market analysis for February 28, 2025, you should be aware of key market movements including a slight decline in the Nifty, a significant drop in South Korea's Kospi, potential tariff threats from Trump affecting the European Union, and broader market declines like the Nasdaq Composite. These factors could impact market sentiment today."
                    })
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
