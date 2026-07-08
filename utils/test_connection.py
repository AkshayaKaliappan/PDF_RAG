import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_voyageai import VoyageAIEmbeddings

def test_groq_connection(api_key):
    try:
        if not api_key:
            return False, "Groq API key is empty."
        
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=api_key.strip(),
            temperature=0
        )
        # Try a simple invocation
        llm.invoke("test")
        return True, "Groq connection successful!"
    except Exception as e:
        return False, f"Groq Error: {str(e)}"

def test_voyage_connection(api_key):
    try:
        if not api_key:
            return False, "Voyage AI API key is empty."
        
        embedding_model = VoyageAIEmbeddings(
            model="voyage-3-lite",
            voyage_api_key=api_key.strip()
        )
        # Try a simple embedding
        embedding_model.embed_query("test")
        return True, "Voyage AI connection successful!"
    except Exception as e:
        return False, f"Voyage AI Error: {str(e)}"
