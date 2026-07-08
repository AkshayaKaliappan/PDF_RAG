import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

def get_llm():
    """
    Create and return the Groq LLM.
    """
    # Priority: 1. UI Input, 2. Environment Variable, 3. Streamlit Secrets
    api_key = st.session_state.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Please set it in the sidebar, environment variables, or Streamlit secrets.")

    # Clean the key
    api_key = api_key.strip()

    try:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=api_key,
            temperature=0
        )
        return llm
    except Exception as e:
        if "invalid" in str(e).lower() or "api key" in str(e).lower() or "401" in str(e):
            raise ValueError("The provided Groq API key is invalid. Please check your key at https://console.groq.com/keys")
        raise e
