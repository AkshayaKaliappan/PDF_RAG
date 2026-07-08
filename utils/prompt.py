from langchain_core.prompts import ChatPromptTemplate


def get_prompt():
    """
    Create and return the RAG prompt.
    """

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful AI assistant.

Answer the question ONLY using the provided context.

If the answer is not present in the context, say:
"I couldn't find the answer in the provided PDF."

Context:
{context}

Question:
{question}

Answer:
"""
    )

    return prompt