import streamlit as st
from chatbot import ask_question

st.set_page_config(page_title="BMSCE Chatbot", page_icon="🎓")

# Logo
col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.image("bmscelogo.png", width=250)

st.markdown(
    """
    <h1 style='text-align: center; font-size: 60px; color:#4DA3FF;'>
    BMSCE AI Chatbot
    </h1>
    """,
    unsafe_allow_html=True
)

with st.form("chat_form"):
    query = st.text_input("Ask a question about the college")
    submit = st.form_submit_button("Ask")

if submit and query:
    answer = ask_question(query)
    st.success(answer)
st.markdown(
    """
    <hr>
    <p style='text-align: center; font-size:14px; color:gray;'>
    Created by Ayush Godara & Aviral Singh
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .bottom-right {
        position: fixed;
        bottom: 10px;
        right: 15px;
        font-size: 14px;
        color: gray;
    }
    </style>

    <div class="bottom-right">
        XCEL CORP HACKATHON
    </div>
    """,
    unsafe_allow_html=True
)