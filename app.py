import streamlit as st
import tempfile
import os

from utils.pdf_loader import load_pdf
from utils.text_splitter import split_documents
from utils.vector_store import create_vector_store
from utils.retriever import get_retriever
from utils.rag_chain import get_rag_chain
from utils.test_connection import test_groq_connection, test_voyage_connection

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
    st.header("🔑 API Configuration")
    
    # Input for Groq
    groq_input = st.text_input(
        "Groq API Key",
        value=st.session_state.get("GROQ_API_KEY", ""),
        type="password",
        help="Get your key from https://console.groq.com/keys"
    )
    if groq_input:
        st.session_state["GROQ_API_KEY"] = groq_input.strip()
    
    # Input for Voyage
    voyage_input = st.text_input(
        "Voyage AI API Key",
        value=st.session_state.get("VOYAGE_API_KEY", ""),
        type="password",
        help="Get your key from https://dashboard.voyageai.com/"
    )
    if voyage_input:
        st.session_state["VOYAGE_API_KEY"] = voyage_input.strip()

    # Connection Tests
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Test Groq"):
            if st.session_state.get("GROQ_API_KEY"):
                with st.spinner("Testing Groq..."):
                    success, msg = test_groq_connection(st.session_state["GROQ_API_KEY"])
                    if success: st.success(msg)
                    else: st.error(msg)
            else:
                st.warning("Please enter a Groq key first.")
                
    with col2:
        if st.button("Test Voyage"):
            if st.session_state.get("VOYAGE_API_KEY"):
                with st.spinner("Testing Voyage..."):
                    success, msg = test_voyage_connection(st.session_state["VOYAGE_API_KEY"])
                    if success: st.success(msg)
                    else: st.error(msg)
            else:
                st.warning("Please enter a Voyage key first.")

    st.divider()

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
    elif not st.session_state.get("GROQ_API_KEY") or not st.session_state.get("VOYAGE_API_KEY"):
        st.error("Please provide both API keys in the sidebar.")
    else:
        progress = st.progress(0)
        progress.progress(20, text="📄 Loading PDF...")

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_pdf_path = tmp_file.name

            # 1. Load PDF
            documents = load_pdf(temp_pdf_path)
            if not documents:
                raise ValueError("The PDF could not be loaded or is empty.")

            # 2. Split Document
            progress.progress(40, text="✂ Splitting Document...")
            chunks = split_documents(documents)
            if not chunks:
                raise ValueError("No text chunks could be extracted from the PDF.")

            # 3. Create Vector Store
            progress.progress(70, text="🧠 Creating Vector Store...")
            vector_store = create_vector_store(chunks)

            # 4. Initialize Chain
            progress.progress(85, text="🔍 Finalizing Setup...")
            retriever = get_retriever(vector_store)
            chain = get_rag_chain()

            # Store in session state
            st.session_state.retriever = retriever
            st.session_state.chain = chain

            # Cleanup
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)

            progress.progress(100, text="✅ Done")
            st.success("PDF processed successfully! You can now ask questions below.")
            
        except Exception as e:
            st.error(f"❌ Error during processing: {str(e)}")
            if "API_KEY" in str(e) or "invalid" in str(e).lower() or "401" in str(e):
                st.info("💡 Tip: Use the 'Test' buttons in the sidebar to verify your keys.")
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
            try:
                # Retrieve
                docs = st.session_state.retriever.invoke(question)
                
                # Format context
                if not docs:
                    context = "No relevant information found in the PDF."
                else:
                    context = "\n\n".join(doc.page_content for doc in docs if hasattr(doc, 'page_content'))

                # Generate
                response = st.session_state.chain.invoke({
                    "context": context,
                    "question": question
                })

                with st.chat_message("user"):
                    st.write(question)

                with st.chat_message("assistant"):
                    st.write(response.content)
            except Exception as e:
                st.error(f"❌ Error during chat: {str(e)}")
                if "API_KEY" in str(e) or "invalid" in str(e).lower() or "401" in str(e):
                    st.info("💡 Tip: Check your API keys in the sidebar.")
                elif "iterable" in str(e).lower():
                    st.info("💡 Tip: Try re-processing the PDF.")

st.divider()

st.caption(
    "Made with ❤️ using Streamlit | Groq | Voyage AI | FAISS | LangChain"
)
