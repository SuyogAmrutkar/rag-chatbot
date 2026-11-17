
from langchain_ollama import ChatOllama

# Initialize local LLM
llm = ChatOllama(
    model="llama3",
    temperature=0.7
)

# Send question
question = "Explain RAG in simple language."

response = llm.invoke(question)

# Print response
print("AI Response:")
print(response.content)
