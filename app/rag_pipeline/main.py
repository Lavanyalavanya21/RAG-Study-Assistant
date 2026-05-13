import streamlit as st
from rag_pipeline import ask_question

st.set_page_config(page_title="IR Study Assistant")

st.title("Information Retrieval Study Assistant")
st.write("Ask questions from your IR notes and slides!")

query = st.text_input("Enter your question:")

if query:
    with st.spinner("Thinking..."):
        response = ask_question(query)
        st.write("Answer:")
        st.write(response)