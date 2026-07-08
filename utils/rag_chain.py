from utils.llm import get_llm
from utils.prompt import get_prompt


def get_rag_chain():

    llm = get_llm()

    prompt = get_prompt()

    chain = prompt | llm

    return chain