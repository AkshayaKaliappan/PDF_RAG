import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Try to load .env if it exists
load_dotenv()

def get_llm():
    """
    Create and return the Groq LLM.
    """
    # Priority: 
    # 1. UI Input (Session State)
    # 2. Environment Variable (os.environ)
    # 3. Streamlit Secrets (st.secrets)
    
    api_key = (
        st.session_state.get("GROQ_API_KEY") or 
        os.getenv("GROQ_API_KEY") or 
        st.secrets.get("GROQ_API_KEY")
    )

    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Please provide it in the sidebar or configuration.")

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
        if "invalid" in str(e).lower() or "401" in str(e):
            raise ValueError("The Groq API key provided is invalid. Please check your key.")
        raise e
