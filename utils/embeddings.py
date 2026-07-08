import os
import streamlit as st
from dotenv import load_dotenv
from langchain_voyageai import VoyageAIEmbeddings

load_dotenv()

def get_embedding_model():
    """
    Create and return the Voyage AI embedding model.
    """
    # Priority: 1. UI Input, 2. Environment Variable, 3. Streamlit Secrets
    api_key = st.session_state.get("VOYAGE_API_KEY") or os.getenv("VOYAGE_API_KEY") or st.secrets.get("VOYAGE_API_KEY")
    
    if not api_key:
        raise ValueError("VOYAGE_API_KEY not found. Please set it in the sidebar, environment variables, or Streamlit secrets.")

    try:
        embedding_model = VoyageAIEmbeddings(
            model="voyage-3-lite",
            voyage_api_key=api_key
        )
        return embedding_model
    except Exception as e:
        if "invalid" in str(e).lower() or "api key" in str(e).lower():
            raise ValueError("The provided Voyage AI API key is invalid. Please check your key and try again.")
        raise e
