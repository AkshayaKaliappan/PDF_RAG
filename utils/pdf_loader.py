from langchain_community.document_loaders import PyMuPDFLoader


def load_pdf(pdf_path):
    """
    Load a PDF file and return it as a list of LangChain Document objects.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        list: List of Document objects.
    """

    loader = PyMuPDFLoader(pdf_path)

    documents = loader.load()

    return documents