import os
import streamlit as st
from dotenv import load_dotenv
from langchain_voyageai import VoyageAIEmbeddings

load_dotenv()

def get_embedding_model():
    """
    Create and return the Voyage AI embedding model.
    """
    # Try to get from environment or streamlit secrets
    api_key = os.getenv("VOYAGE_API_KEY") or st.secrets.get("VOYAGE_API_KEY")
    
    if not api_key:
        raise ValueError("VOYAGE_API_KEY not found. Please set it in your environment variables or Streamlit secrets.")

    embedding_model = VoyageAIEmbeddings(
        model="voyage-3-lite",
        voyage_api_key=api_key
    )

    return embedding_model
