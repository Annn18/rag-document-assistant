import os
import io
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

from src.document_loader import DocumentLoader
from src.text_splitter import DocumentSplitter
from src.vector_store import VectorStoreManager
from src.rag_pipeline import RAGPipeline
from src.config import DEFAULT_TOP_K, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, STORAGE_DIR

load_dotenv()

# Streamlit Page Setup
st.set_page_config(
    page_title="RAG Document Q&A Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1E293B;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .citation-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        border-radius: 4px;
        padding: 10px 14px;
        margin: 8px 0;
        font-size: 0.9rem;
    }
    .badge-doc {
        display: inline-block;
        background-color: #E2E8F0;
        color: #1E293B;
        padding: 2px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
        margin-right: 6px;
    }
    .badge-page {
        display: inline-block;
        background-color: #DBEAFE;
        color: #1D4ED8;
        padding: 2px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_manager" not in st.session_state:
    st.session_state.vector_manager = VectorStoreManager()
    # Attempt to load persistent index if available
    try:
        st.session_state.vector_manager.load_index()
    except Exception:
        pass

if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None

if "processed_docs_info" not in st.session_state:
    st.session_state.processed_docs_info = []

# Sidebar Controls
with st.sidebar:
    st.title("📁 Document Ingestion")
    st.markdown("Upload documents (**PDF, DOCX, PPTX**) to build the vector knowledge base.")

    uploaded_files = st.file_uploader(
        "Select documents",
        type=["pdf", "docx", "pptx"],
        accept_multiple_files=True,
        help="Upload one or multiple documents to query simultaneously.",
    )

    st.markdown("---")
    st.subheader("⚙️ Pipeline Configuration")

    chunk_size = st.number_input("Chunk Size (characters)", min_value=200, max_value=2000, value=DEFAULT_CHUNK_SIZE, step=100)
    chunk_overlap = st.number_input("Chunk Overlap (characters)", min_value=0, max_value=500, value=DEFAULT_CHUNK_OVERLAP, step=50)
    top_k = st.slider("Retrieved Context Chunks (Top-K)", min_value=1, max_value=8, value=DEFAULT_TOP_K)

    llm_choice = st.selectbox(
        "LLM Generator Backend",
        options=["HuggingFace Hub (API)", "Local HuggingFace Model (CPU)"],
        index=0,
        help="Choose between Hugging Face Inference API or a local offline Seq2Seq model.",
    )

    hf_token = ""
    hf_model_repo = "mistralai/Mistral-7B-Instruct-v0.2"
    if llm_choice == "HuggingFace Hub (API)":
        env_token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
        hf_token = st.text_input(
            "Hugging Face API Token",
            value=env_token,
            type="password",
            help="Enter your HuggingFace User Access Token (from huggingface.co/settings/tokens)",
        )
        hf_model_repo = st.text_input(
            "HuggingFace Model Repo",
            value="mistralai/Mistral-7B-Instruct-v0.2",
            help="Any text generation model repository on Hugging Face Hub",
        )

    # Ingestion Trigger Button
    process_button = st.button("🚀 Process & Index Documents", type="primary", use_container_width=True)

    if process_button:
        if not uploaded_files:
            st.error("Please upload at least one document (PDF, DOCX, or PPTX).")
        else:
            with st.spinner("Extracting text and metadata from documents..."):
                all_extracted_docs = []
                docs_summary = []

                for uploaded_file in uploaded_files:
                    file_bytes = io.BytesIO(uploaded_file.read())
                    filename = uploaded_file.name
                    extracted = DocumentLoader.load_document(file_bytes, filename)
                    all_extracted_docs.extend(extracted)
                    docs_summary.append({
                        "name": filename,
                        "pages": len(extracted),
                        "type": Path(filename).suffix.upper().replace(".", "")
                    })

            if not all_extracted_docs:
                st.warning("No readable text could be extracted from the uploaded files.")
            else:
                with st.spinner(f"Chunking {len(all_extracted_docs)} document pages/slides..."):
                    splitter = DocumentSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    chunks = splitter.split_documents(all_extracted_docs)

                with st.spinner(f"Generating Sentence-Transformer embeddings for {len(chunks)} chunks & building FAISS index..."):
                    st.session_state.vector_manager.create_vector_store(chunks)
                    # Persist index
                    st.session_state.vector_manager.save_index()
                    st.session_state.processed_docs_info = docs_summary

                # Invalidate existing pipeline so it re-initializes with new vectorstore
                st.session_state.rag_pipeline = None

                st.success(f"Indexed {len(docs_summary)} documents into {len(chunks)} searchable chunks!")

    # Knowledge Base Status in Sidebar
    if st.session_state.processed_docs_info:
        st.markdown("---")
        st.subheader("📊 Indexed Knowledge Base")
        for doc in st.session_state.processed_docs_info:
            st.write(f"- **{doc['name']}** ({doc['type']}) — {doc['pages']} pages/slides")

    # Clear Index Action
    if st.button("🗑️ Reset Vector Store", use_container_width=True):
        st.session_state.vector_manager = VectorStoreManager()
        st.session_state.processed_docs_info = []
        st.session_state.messages = []
        st.session_state.rag_pipeline = None
        # Remove persisted index files if exist
        for f in STORAGE_DIR.glob("*"):
            try:
                f.unlink()
            except Exception:
                pass
        st.rerun()

# Main Header
st.markdown('<div class="main-title">📚 RAG Document Q&A Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Ask natural-language questions over multi-format documents (PDF, DOCX, PPTX) with verified page-level source citations.</div>',
    unsafe_allow_html=True,
)

# Helper function to get or build RAG Pipeline
def get_pipeline():
    if st.session_state.rag_pipeline is None:
        if st.session_state.vector_manager.vector_store is None:
            return None
        
        mode = "huggingface_hub" if llm_choice == "HuggingFace Hub (API)" else "local"
        st.session_state.rag_pipeline = RAGPipeline(
            vector_store_manager=st.session_state.vector_manager,
            llm_type=mode,
            hf_api_token=hf_token if mode == "huggingface_hub" else None,
            model_name=hf_model_repo if mode == "huggingface_hub" else None,
            top_k=top_k,
        )
    return st.session_state.rag_pipeline

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander(f"📖 View {len(message['sources'])} Cited Sources & Page References"):
                for idx, src in enumerate(message["sources"], start=1):
                    doc_label = "Slide" if src.get("doc_type") == "pptx" else "Page"
                    st.markdown(
                        f"""
                        <div class="citation-box">
                            <span class="badge-doc">📄 {src.get('source')}</span>
                            <span class="badge-page">🔖 {doc_label} {src.get('page')}</span>
                            <span style="color:#64748B; font-size:0.8rem; float:right;">Distance Score: {src.get('score')}</span>
                            <p style="margin-top:6px; margin-bottom:0; color:#334155;">{src.get('content')}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# Empty State Welcome Guide
if not st.session_state.messages and not st.session_state.processed_docs_info:
    st.info(
        """
        ### 👋 Welcome to the RAG Document Q&A Assistant!
        To get started:
        1. **Upload documents** (PDF, DOCX, or PPTX) using the sidebar on the left.
        2. Click **'🚀 Process & Index Documents'** to generate dense embeddings and create the FAISS index.
        3. Type your question in the box below to retrieve answers with **page-level citations**.
        """
    )

# Question Input Area
if user_prompt := st.chat_input("Ask a question about your uploaded documents..."):
    # Render user query
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Process query through RAG pipeline
    pipeline = get_pipeline()
    if pipeline is None:
        error_msg = "Please upload and process at least one document before asking questions."
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        with st.chat_message("assistant"):
            st.warning(error_msg)
    else:
        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant passages and synthesizing answer with citations..."):
                try:
                    response = pipeline.answer_query(user_prompt, top_k=top_k)
                    answer = response.get("answer", "No answer generated.")
                    sources = response.get("sources", [])

                    st.markdown(answer)

                    if sources:
                        with st.expander(f"📖 View {len(sources)} Cited Sources & Page References"):
                            for src in sources:
                                doc_label = "Slide" if src.get("doc_type") == "pptx" else "Page"
                                st.markdown(
                                    f"""
                                    <div class="citation-box">
                                        <span class="badge-doc">📄 {src.get('source')}</span>
                                        <span class="badge-page">🔖 {doc_label} {src.get('page')}</span>
                                        <span style="color:#64748B; font-size:0.8rem; float:right;">Distance Score: {src.get('score')}</span>
                                        <p style="margin-top:6px; margin-bottom:0; color:#334155;">{src.get('content')}</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                    # Save to conversation history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })

                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
