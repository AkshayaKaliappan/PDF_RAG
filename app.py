import streamlit as st
import tempfile
import os

from utils.pdf_loader import load_pdf
from utils.text_splitter import split_documents
from utils.vector_store import create_vector_store
from utils.retriever import get_retriever
from utils.rag_chain import get_rag_chain

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ---------------- SESSION ---------------- #

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "chain" not in st.session_state:
    st.session_state.chain = None

# ---------------- HEADER ---------------- #

st.markdown(
    """
    <h1 style='text-align:center;'>🤖 PDF RAG Chatbot</h1>
    <p style='text-align:center;font-size:18px;color:gray;'>
        Chat with your PDF using <b>Groq</b> • <b>Voyage AI</b> • <b>FAISS</b> • <b>LangChain</b>
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ---------------- SIDEBAR ---------------- #

with st.sidebar:

    st.header("📂 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"]
    )

    process_button = st.button(
        "🚀 Process PDF",
        use_container_width=True
    )

    st.divider()

    st.subheader("⚙ AI Stack")

    st.markdown("""
- 🤖 Groq LLM
- 🧠 Voyage AI
- 📚 FAISS
- 🔗 LangChain
""")

# ---------------- PDF DETAILS ---------------- #

if uploaded_file:

    st.success("✅ PDF Uploaded Successfully")

    col1, col2 = st.columns(2)

    with col1:
        st.info(f"**File:** {uploaded_file.name}")

    with col2:
        st.info(f"**Size:** {uploaded_file.size / 1024:.2f} KB")

# ---------------- PDF PROCESS ---------------- #

if process_button:

    if uploaded_file is None:

        st.error("Please upload a PDF first.")

    else:

        progress = st.progress(0)

        progress.progress(20, text="📄 Loading PDF...")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_pdf_path = tmp_file.name

        documents = load_pdf(temp_pdf_path)

        progress.progress(40, text="✂ Splitting Document...")

        chunks = split_documents(documents)

        progress.progress(70, text="🧠 Creating Vector Store...")

        try:
            vector_store = create_vector_store(chunks)

            progress.progress(85, text="🔍 Creating Retriever...")

            retriever = get_retriever(vector_store)

            chain = get_rag_chain()

            st.session_state.retriever = retriever
            st.session_state.chain = chain

            os.remove(temp_pdf_path)

            progress.progress(100, text="✅ Done")

            st.success("PDF processed successfully!")
        except Exception as e:
            st.error(f"❌ Error during processing: {str(e)}")
            if "API_KEY" in str(e):
                st.info("💡 Tip: Make sure to add your API keys to Streamlit Secrets (Settings > Secrets).")
            progress.empty()

st.divider()

# ---------------- CHAT ---------------- #

st.subheader("💬 Chat with your PDF")

question = st.chat_input(
    "Ask anything from your uploaded PDF..."
)

# ---------------- ANSWER ---------------- #

if question:

    if st.session_state.retriever is None:

        st.warning("Please process a PDF first.")

    elif question.strip() == "":

        st.warning("Please enter a question.")

    else:

        with st.spinner("🤖 Thinking..."):

            docs = st.session_state.retriever.invoke(question)

            context = "\n\n".join(
                doc.page_content for doc in docs
            )

            response = st.session_state.chain.invoke(
                {
                    "context": context,
                    "question": question
                }
            )

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            st.write(response.content)

st.divider()

st.caption(
    "Made with ❤️ using Streamlit | Groq | Voyage AI | FAISS | LangChain"
)