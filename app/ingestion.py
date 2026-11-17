from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from vector_store import create_vector_store, search_documents


def load_pdf(file_path):

    loader = PyPDFLoader(file_path)

    return loader.load()


def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    return splitter.split_documents(documents)


if __name__ == "__main__":

    pdf_path = "data/documents/company.pdf"

    # Step 1: Load PDF
    documents = load_pdf(pdf_path)

    print("Total pages:", len(documents))

    # Step 2: Split PDF
    chunks = split_documents(documents)

    print("Total chunks:", len(chunks))

    # Step 3: Store chunks in ChromaDB
    vector_store = create_vector_store(chunks)

    print("\nDocuments successfully stored!")

    # Step 4: Search documents
    # query = "Which is the world's largest rainforest?"

    # results = search_documents(query)

    # print("\nRelevant search results:")

    # for index, result in enumerate(results):

    #     print("\n------------------------")

    #     print("Result:", index + 1)

    #     print("Content:", result.page_content)

    #     print("Metadata:", result.metadata)

    queries = [
        "Which is the world's largest rainforest?",
        "What percentage of the world's bird species live in the Amazon?",
        "What percentage of the world's known species live in the Amazon?",
        "Which organization is dedicated to conserving Latin America's ecosystems and cultures?",
    ]

    for query in queries:

        print("\nUSER QUESTION:", query)

        results = search_documents(query, k=3)

        for result in results:

            print("RETRIEVED:", result.page_content)
            print("SOURCE:", result.metadata)
