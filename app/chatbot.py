
import streamlit as st

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage


@st.cache_resource
def get_llm():

    llm = ChatOllama(
        model="llama3",
        temperature=0.7
    )

    return llm


def generate_response(chat_history):

    llm = get_llm()

    # Convert Streamlit chat history into
    # LangChain message objects
    messages = []

    for message in chat_history:

        if message["role"] == "user":
            messages.append(
                HumanMessage(content=message["content"])
            )

        elif message["role"] == "assistant":
            messages.append(
                AIMessage(content=message["content"])
            )

    # Send conversation to Ollama
    response = llm.invoke(messages)

    return response.content
