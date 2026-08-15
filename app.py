import os
import streamlit as st
from chatbot import ask_question
from vector_store import build_vector_store

st.set_page_config(page_title="BMSCE Chatbot", page_icon="🎓", layout="centered")

# Sidebar: Document Uploader & Indexing
with st.sidebar:
    st.image("bmscelogo.png", width=180)
    st.title("📚 Knowledge Base")
    st.write("Upload PDF or TXT files to expand chatbot knowledge.")

    uploaded_files = st.file_uploader("Choose PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)
    
    if st.button("🔄 Rebuild Vector Index", use_container_width=True):
        if uploaded_files:
            os.makedirs("data", exist_ok=True)
            for uploaded_file in uploaded_files:
                file_path = os.path.join("data", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            st.success(f"Saved {len(uploaded_files)} file(s) to knowledge base.")

        with st.spinner("Building vector index with LangChain & FAISS..."):
            vector_db = build_vector_store()
            if vector_db:
                st.success("Vector store built successfully!")
            else:
                st.error("Failed to build vector store. Check documents in data/ directory.")

    st.markdown("---")
    st.caption("Powered by **LangChain**, **PyMuPDF**, **FAISS** & **Groq**")

# Main Interface
st.title("🎓 BMSCE AI Assistant")
st.caption("Ask questions about courses, admissions, hostel, library, and campus facilities.")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am the BMSCE AI Assistant. How can I help you today?"}
    ]

# Display Chat Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User Input
if user_query := st.chat_input("Ask a question..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            response = ask_question(user_query)
            st.write(response)

    # Append assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})