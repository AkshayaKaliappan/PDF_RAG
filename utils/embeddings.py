import os

from dotenv import load_dotenv
from langchain_voyageai import VoyageAIEmbeddings

load_dotenv()


def get_embedding_model():
    """
    Create and return the Voyage AI embedding model.
    """

    embedding_model = VoyageAIEmbeddings(
        model="voyage-3-lite",
        voyage_api_key=os.getenv("VOYAGE_API_KEY")
    )

    return embedding_model