import streamlit as st
import hashlib
from pathlib import Path

from ingestion import load_pdf, split_documents
from vector_store import create_vector_store
from rag_chain import ask_question

# Page configuration
st.set_page_config(page_title="My RAG Chatbot", page_icon="📚", layout="wide")

st.title("📚 Document Chatbot")
st.write("Upload a PDF and ask questions about its content.")


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed_file_hash" not in st.session_state:
    st.session_state.indexed_file_hash = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None


# Sidebar: PDF upload
with st.sidebar:

    st.header("Upload Document")

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:

        file_bytes = uploaded_file.getvalue()

        # Identify whether this is a new or changed file
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        if file_hash != st.session_state.indexed_file_hash:

            with st.spinner("Processing PDF..."):

                # Save uploaded PDF
                document_dir = Path("data/documents")
                document_dir.mkdir(parents=True, exist_ok=True)

                file_path = document_dir / uploaded_file.name
                file_path.write_bytes(file_bytes)

                # Load and split PDF
                documents = load_pdf(str(file_path))
                chunks = split_documents(documents)

                # Store chunks in ChromaDB
                create_vector_store(chunks)

                # Remember indexed file
                st.session_state.indexed_file_hash = file_hash
                st.session_state.document_name = uploaded_file.name

                # Start a fresh chat for the new document
                st.session_state.messages = []

            st.success("PDF indexed successfully!")

        else:
            st.success("PDF is already indexed!")

    if st.session_state.document_name:
        st.write("**Current document:**")
        st.caption(st.session_state.document_name)


# Display previous chat messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Display sources for assistant messages
        if message["role"] == "assistant":
            sources = message.get("sources", [])

            if sources:
                with st.expander("View sources"):
                    for doc in sources:
                        page = doc.metadata.get("page")

                        if page is not None:
                            page = page + 1

                        st.write(f"**File:** {doc.metadata.get('source', 'Unknown')}")
                        st.write(f"**Page:** {page}")
                        st.caption(doc.page_content)


# Chat input
question = st.chat_input("Ask something about your document...")

if question:

    if st.session_state.indexed_file_hash is None:

        st.warning("Please upload a PDF document first.")

    else:

        # Display user question
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        # Generate answer using RAG
        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                result = ask_question(question)

                answer = result["answer"]
                sources = result["sources"]

            st.markdown(answer)

            # Display source information
            if sources:

                with st.expander("View sources"):

                    for doc in sources:

                        page = doc.metadata.get("page")

                        if page is not None:
                            page = page + 1

                        st.write(f"**File:** {doc.metadata.get('source', 'Unknown')}")
                        st.write(f"**Page:** {page}")
                        st.caption(doc.page_content)

        # Save assistant response
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
