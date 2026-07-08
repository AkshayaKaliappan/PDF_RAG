from utils.pdf_loader import load_pdf
from utils.text_splitter import split_documents
from utils.vector_store import create_vector_store
from utils.retriever import get_retriever
from utils.rag_chain import get_rag_chain

print("Loading PDF...")

documents = load_pdf("data/sample.pdf")

chunks = split_documents(documents)

vector_store = create_vector_store(chunks)

retriever = get_retriever(vector_store)

chain = get_rag_chain()

print("\n===================================")
print("     PDF RAG Chatbot Ready 🤖")
print("Type 'exit' to quit")
print("===================================\n")

while True:

    question = input("You : ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    docs = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in docs)

    response = chain.invoke(
        {
            "context": context,
            "question": question
        }
    )

    print("\nBot :", response.content)
    print()