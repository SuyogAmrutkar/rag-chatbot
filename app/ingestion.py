from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(file_path):

    loader = PyPDFLoader(file_path)

    documents = loader.load()

    return documents


def split_documents(documents):

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    # Split documents into smaller chunks
    chunks = text_splitter.split_documents(documents)

    return chunks


if __name__ == "__main__":

    pdf_path = "data/documents/company.pdf"

    # Step 1: Load PDF
    documents = load_pdf(pdf_path)

    print("Total pages:", len(documents))

    # Step 2: Split documents
    chunks = split_documents(documents)

    print("Total chunks:", len(chunks))

    # Step 3: Inspect chunks
    for index, chunk in enumerate(chunks):

        print("\n-----------------------------")

        print("Chunk:", index + 1)

        print("Metadata:", chunk.metadata)

        print("Content:", chunk.page_content)
