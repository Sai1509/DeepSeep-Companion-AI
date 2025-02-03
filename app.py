import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* General Page Styling */
    .stApp {
        background-color: #0d1117 !important;
        color: #e6edf3 !important;
    }
    .stTextInput textarea, .stChatInput textarea {
        color: #e6edf3 !important;
        background-color: #161b22 !important;
        border-radius: 10px;
        padding: 12px;
        border: 1px solid #30363d;
    }
    .stChatMessage {
        padding: 12px;
        border-radius: 12px;
        margin-bottom: 10px;
    }
    .stChatMessage.user {
        background-color: #238636;
        color: white;
        text-align: left;
    }
    .stChatMessage.ai {
        background-color: #1f6feb;
        color: white;
        text-align: left;
    }
    .stChatInput {
        margin-top: 20px;
    }
    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background-color: #161b22 !important;
        color: white !important;
    }
    .stSelectbox div[data-baseweb="select"] {
        color: white !important;
        background-color: #1f6feb !important;
    }
    .stSelectbox svg {
        fill: white !important;
    }
    div[role="listbox"] div {
        background-color: #161b22 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Page title and header
st.title("🛠️ CodeSmith AI")
st.caption("🤖 Your AI-Powered Coding Assistant for Debugging & Development")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Settings")
    selected_model = st.selectbox(
        "Select AI Model",
        ["codesmith-1.5b", "codesmith-3b"],
        index=0
    )
    st.divider()
    st.markdown("### 🤖 AI Capabilities")
    st.markdown("""
    - 🐍 Python Coding Assistance
    - 🔍 Debugging Support
    - 📖 Code Documentation
    - 🎯 Solution Architecture
    """)
    st.divider()
    st.markdown("Powered by [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")

# Initiate the chat engine
llm_engine = ChatOllama(
    model=selected_model,
    base_url="http://localhost:11434",
    temperature=0.3
)

# System prompt configuration
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are CodeSmith AI, an expert AI coding assistant. Provide precise, well-structured solutions "
    "with effective debugging strategies. Always respond in clear and concise English."
)

# Session state management
if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hello! I'm CodeSmith AI. How can I assist with your coding today? 🚀"}]

# Chat container
chat_container = st.container()

# Display chat messages
with chat_container:
    for message in st.session_state.message_log:
        role_class = "user" if message["role"] == "user" else "ai"
        with st.chat_message(message["role"]):
            st.markdown(f'<div class="stChatMessage {role_class}">{message["content"]}</div>', unsafe_allow_html=True)

# Chat input and processing
user_query = st.chat_input("Ask a coding question...")

def generate_ai_response(prompt_chain):
    processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
    return processing_pipeline.invoke({})

def build_prompt_chain():
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

if user_query:
    # Add user message to log
    st.session_state.message_log.append({"role": "user", "content": user_query})
    
    # Generate AI response
    with st.spinner("🤖 Thinking..."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)
    
    # Add AI response to log
    st.session_state.message_log.append({"role": "ai", "content": ai_response})
    
    # Rerun to update chat display
    st.rerun()
