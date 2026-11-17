from langchain_ollama import OllamaEmbeddings

# Initialize embedding model

embeddings = OllamaEmbeddings(model="nomic-embed-text")

def generate_embeddings(text):

    # Convert embeddings into numerical vectors
    vector = embeddings.embed_query(text)

    return vector

if __name__ == "__main__":

    text = "Saved Tags allows users to save multiple filters."

    vector = generate_embeddings(text)

    print("Embedding:")
    print(vector)

    print("\nVector dimensions:")
    print(len(vector))
