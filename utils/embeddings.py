import os
import streamlit as st
from dotenv import load_dotenv
from langchain_voyageai import VoyageAIEmbeddings

# Try to load .env if it exists (local development)
load_dotenv()

def get_embedding_model():
    """
    Create and return the Voyage AI embedding model.
    """
    # Priority: 
    # 1. UI Input (Session State)
    # 2. Environment Variable (os.environ) - includes .env and Streamlit Cloud env
    # 3. Streamlit Secrets (st.secrets)
    
    api_key = (
        st.session_state.get("VOYAGE_API_KEY") or 
        os.getenv("VOYAGE_API_KEY") or 
        st.secrets.get("VOYAGE_API_KEY")
    )
    
    if not api_key:
        raise ValueError("VOYAGE_API_KEY not found. Please provide it in the sidebar or configuration.")

    # Clean the key to prevent whitespace issues
    api_key = api_key.strip()

    try:
        embedding_model = VoyageAIEmbeddings(
            model="voyage-3-lite",
            voyage_api_key=api_key
        )
        return embedding_model
    except Exception as e:
        # Provide a clearer error message for invalid keys
        if "invalid" in str(e).lower() or "401" in str(e):
            raise ValueError("The Voyage AI API key provided is invalid. Please check your key.")
        raise e
