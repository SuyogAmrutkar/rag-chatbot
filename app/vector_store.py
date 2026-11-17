from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# Initialize embedding model
def get_embedding_model():

    return OllamaEmbeddings(model="nomic-embed-text")


# Create vector database
def create_vector_store(chunks):

    embeddings = get_embedding_model()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="company_documents",
        persist_directory="./chroma_db",
    )

    return vector_store


# Load an existing vector database
def load_vector_store():

    embeddings = get_embedding_model()

    vector_store = Chroma(
        collection_name="company_documents",
        persist_directory="./chroma_db",
        embedding_function=embeddings,
    )

    return vector_store


# Search relevant document chunks
def search_documents(query, k=3):

    vector_store = load_vector_store()

    results = vector_store.similarity_search(query, k=k)

    return results
