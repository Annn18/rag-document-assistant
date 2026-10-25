# 📚 RAG Document Q&A Assistant

A production-grade, modular **Retrieval-Augmented Generation (RAG)** Document Q&A Assistant built with **Python**, **LangChain**, **FAISS**, **Sentence-Transformers**, **HuggingFace**, and **Streamlit**.

The assistant allows users to upload multi-format documents (**PDF**, **DOCX**, **PPTX**) and ask natural-language questions about their contents. Relevant passages are retrieved using dense semantic vector search, and an LLM synthesizes accurate answers grounded strictly in the retrieved context with **verified document and page/slide citations**.

---

## 🌟 Key Features

- **Multi-Format Document Parsing**:
  - **PDF**: Page-by-page text extraction preserving exact page numbers using `pypdf`.
  - **DOCX**: Structured paragraph and table extraction preserving sections and estimated page breaks using `python-docx`.
  - **PPTX**: Slide-by-slide text and table extraction mapping citations to specific slide numbers using `python-pptx`.
- **Intelligent Text Chunking**: Recursive character chunking (`RecursiveCharacterTextSplitter`) with configurable chunk size and overlap, propagating source file and page metadata to every chunk.
- **Dense Vector Search**: Generates 384-dimensional dense semantic embeddings using `sentence-transformers/all-MiniLM-L6-v2` and indexes them into **FAISS** for fast similarity retrieval.
- **Persistent Vector Storage**: Automatically saves and reloads FAISS indexes (`index.faiss` and `index.pkl`) on disk.
- **Flexible LLM Generation**:
  - **Local HuggingFace Model (Offline / CPU)**: Runs lightweight seq2seq models (e.g. `google/flan-t5-base`) locally on CPU without requiring an API key.
  - **HuggingFace Hub (API)**: Connects to high-performance inference endpoints (e.g. `mistralai/Mistral-7B-Instruct-v0.2`, `meta-llama/Llama-3-8B-Instruct`).
- **Verified Source Citations**: Every generated answer provides citation badges indicating the source document name, page/slide number, match distance, and relevant text excerpts.
- **Interactive Streamlit Web Interface**: Complete with drag-and-drop document upload, real-time chunking & indexing metrics, chat interface, and citation inspectors.

---

## 🏗️ Project Architecture

```
rag-document-assistant/
├── .env.example              # Sample environment configuration
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── app.py                    # Streamlit web application
├── src/
│   ├── __init__.py           # Package exports
│   ├── config.py             # Configuration parameters and default settings
│   ├── document_loader.py    # Multi-format document parser (PDF, DOCX, PPTX)
│   ├── text_splitter.py      # Recursive text chunking with metadata tracking
│   ├── vector_store.py       # Sentence-Transformers embeddings & FAISS index manager
│   └── rag_pipeline.py       # LangChain retrieval chain, prompt template & citations
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py      # Automated tests for loader, chunker, vector store, and search
└── data/                     # Vector store and uploads directory (created automatically)
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+** (Tested and compatible with Python 3.10 – 3.14 on Windows, Linux, and macOS)
- `pip` package manager

### 1. Clone or Navigate to the Project

```bash
cd "rag-document-assistant"
```

### 2. Create and Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables (Optional)

If you plan to use the **Hugging Face Hub API** instead of the local offline model, copy `.env.example` to `.env` and provide your Hugging Face API token:

```bash
cp .env.example .env
```

Edit `.env`:
```ini
HUGGINGFACEHUB_API_TOKEN=hf_your_actual_token_here
```
*(You can also input your token directly in the Streamlit UI sidebar).*

---

## 💻 Running the Application

Launch the Streamlit web application:

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

---

## 📖 How to Use

1. **Upload Documents**:
   - In the left sidebar, click **"Select documents"** to upload one or more `.pdf`, `.docx`, or `.pptx` files.
2. **Configure Pipeline** (optional):
   - Adjust chunk size, chunk overlap, or top-k retrieval chunks.
   - Choose between **HuggingFace Hub (API)** or **Local HuggingFace Model (CPU)**.
3. **Index Documents**:
   - Click **"🚀 Process & Index Documents"**. The system extracts text, chunks it, generates dense embeddings, and builds the FAISS index.
4. **Ask Questions**:
   - In the chat input, ask natural-language questions about the contents of your documents.
5. **Inspect Citations**:
   - Under each answer, expand **"📖 View Cited Sources & Page References"** to review the exact source file, page/slide number, similarity score, and excerpt.

---

## 🧪 Running Automated Tests

Run the test suite to verify document extraction, chunking, FAISS index creation, and persistence:

```bash
python -m unittest tests/test_pipeline.py
```

---

## 🛠️ Tech Stack Details

| Component | Library / Framework | Description |
| :--- | :--- | :--- |
| **Pipeline Framework** | `LangChain` | Retrieval orchestration, document modeling, prompt templates |
| **Vector Storage** | `FAISS` (`faiss-cpu`) | High-speed dense similarity search and index persistence |
| **Embedding Model** | `Sentence-Transformers` | `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors) |
| **Document Parsers** | `pypdf`, `python-docx`, `python-pptx` | Format-specific text extraction with page/slide metadata |
| **LLM Generation** | `HuggingFace Hub` & `Transformers` | Online inference endpoint or local CPU text2text pipeline |
| **Web Interface** | `Streamlit` | Interactive dashboard with real-time feedback and citation cards |

---

## 🔮 Roadmap (Upcoming Phases)

- **Phase 2**: Hybrid retrieval combining dense semantic embeddings with BM25 keyword search.
- **Phase 3**: Cross-encoder re-ranking layer (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
- **Phase 4**: Hallucination guardrails and source attribution validation.
