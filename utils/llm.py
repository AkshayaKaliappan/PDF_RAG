import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

def get_llm():
    """
    Create and return the Groq LLM.
    """
    # Try to get from environment or streamlit secrets
    api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Please set it in your environment variables or Streamlit secrets.")

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=api_key,
        temperature=0
    )

    return llm
