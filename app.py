import streamlit as st
import tempfile
import os
from utils.file_handlers import extract_text
from utils.indexing import chunk_text, index_documents
from utils.retrieval import rag_answer
from utils.qdrant_ops import QdrantManager

st.set_page_config(page_title="RAG with Gemini & Qdrant", layout="wide")
st.title("📚 RAG Assistant with Gemini & Qdrant")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed" not in st.session_state:
    st.session_state.indexed = False

# Sidebar for document upload and indexing
with st.sidebar:
    st.header("1. Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files (TXT, PDF, DOCX)",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True
    )

    if st.button("🚀 Index Documents", disabled=not uploaded_files):
        if uploaded_files:
            with st.spinner("Extracting text and indexing..."):
                all_chunks = []
                for uploaded_file in uploaded_files:
                    try:
                        text = extract_text(uploaded_file)
                        chunks = chunk_text(text, uploaded_file.name)
                        all_chunks.extend(chunks)
                        st.success(f"✅ {uploaded_file.name}: {len(chunks)} chunks")
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")

                if all_chunks:
                    # Index with progress bar
                    progress_bar = st.progress(0, text="Indexing...")
                    def update_progress(current, total):
                        progress_bar.progress(current / total, text=f"Indexing {current}/{total}")
                    index_documents(all_chunks, progress_callback=update_progress)
                    st.session_state.indexed = True
                    st.success(f"🎉 Indexed {len(all_chunks)} chunks successfully!")
                else:
                    st.warning("No valid chunks to index.")
        else:
            st.info("Please upload at least one file.")

    st.divider()
    st.header("2. Collection Info")
    qdrant = QdrantManager()
    info = qdrant.get_collection_info()
    if info:
        st.write(f"**Collection:** `{qdrant.collection_name}`")
        st.write(f"**Vectors:** {info.points_count}")
    else:
        st.warning("Collection not found. Upload documents to create it.")

    st.divider()
    st.caption("Make sure Qdrant is running on localhost:6333")

# Main chat area
st.header("3. Ask Questions")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = rag_answer(prompt)
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})