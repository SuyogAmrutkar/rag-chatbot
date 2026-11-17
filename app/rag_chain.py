from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from vector_store import search_documents

# 1. Initialize the local LLM
llm = ChatOllama(model="llama3", temperature=0)


# 2. Create the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
        You are a helpful assistant.

        Answer the user's question using ONLY the
        information provided in the context.

        If the answer is not available in the context,
        clearly say:
        "I could not find this information in the
        provided documents."

        Do not make up information.

        Context:
        {context}
        """,
        ),
        ("human", "{question}"),
    ]
)


# 3. Format retrieved documents into context
def format_documents(documents):

    context = ""

    for index, doc in enumerate(documents):

        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page")

        if page is not None:
            page = page + 1

        context += f"""
        Document {index + 1}
        Source: {source}
        Page: {page}

        Content:
        {doc.page_content}

        -------------------------
        """

    return context


# 4. Main RAG function
def ask_question(question):

    # Retrieve relevant chunks from ChromaDB
    documents = search_documents(question, k=3)

    # Check whether relevant documents were found
    if not documents:
        return {
            "answer": "I could not find relevant information in the provided documents.",
            "sources": [],
        }

    # Convert retrieved documents into context
    context = format_documents(documents)

    # Prepare the prompt
    messages = prompt.format_messages(context=context, question=question)

    # Send the prompt to Llama 3
    response = llm.invoke(messages)

    return {"answer": response.content, "sources": documents}


# 5. Test the RAG pipeline
if __name__ == "__main__":

    questions = [
        "What are the effects of deforestation?",
        "How is Nature and Culture International protecting the Amazon?",
        "How large is the Maijuna-Kichwa Reserve?",
        "What food products originally come from the Amazon?",
        "What is the population of Brazil?",
    ]

    for question in questions:

        result = ask_question(question)

        print("\nQUESTION:", question)
        print("ANSWER:", result["answer"])
