from pathlib import Path
from typing import List, Optional, Tuple, Union
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_TOP_K, STORAGE_DIR

class VectorStoreManager:
    """
    Manages Sentence-Transformers embedding generation and FAISS vector storage,
    including indexing, persistence, and similarity retrieval.
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, storage_dir: Union[str, Path] = STORAGE_DIR):
        self.model_name = model_name
        self.storage_dir = Path(storage_dir)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vector_store: Optional[FAISS] = None

    def create_vector_store(self, chunks: List[Document]) -> FAISS:
        """Embeds document chunks and creates a new FAISS index."""
        if not chunks:
            raise ValueError("No document chunks provided for vector store creation.")
        
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        return self.vector_store

    def add_documents(self, chunks: List[Document]):
        """Adds additional document chunks to an existing FAISS index."""
        if not chunks:
            return
        if self.vector_store is None:
            self.create_vector_store(chunks)
        else:
            self.vector_store.add_documents(chunks)

    def save_index(self, folder_path: Optional[Union[str, Path]] = None):
        """Persists the FAISS index to disk."""
        target_dir = Path(folder_path) if folder_path else self.storage_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        if self.vector_store is None:
            raise ValueError("No active vector store to persist.")
        self.vector_store.save_local(str(target_dir))

    def load_index(self, folder_path: Optional[Union[str, Path]] = None) -> bool:
        """Loads a previously saved FAISS index from disk."""
        target_dir = Path(folder_path) if folder_path else self.storage_dir
        index_file = target_dir / "index.faiss"
        if not index_file.exists():
            return False

        self.vector_store = FAISS.load_local(
            str(target_dir),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        return True

    def similarity_search(self, query: str, k: int = DEFAULT_TOP_K) -> List[Document]:
        """Performs similarity search and returns the top k relevant document chunks."""
        if self.vector_store is None:
            raise ValueError("Vector store is empty. Please upload and process documents first.")
        return self.vector_store.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str, k: int = DEFAULT_TOP_K) -> List[Tuple[Document, float]]:
        """Performs similarity search and returns documents with cosine distance scores."""
        if self.vector_store is None:
            raise ValueError("Vector store is empty. Please upload and process documents first.")
        return self.vector_store.similarity_search_with_score(query, k=k)

    def get_retriever(self, k: int = DEFAULT_TOP_K):
        """Returns a LangChain retriever interface for chaining."""
        if self.vector_store is None:
            raise ValueError("Vector store is empty. Please upload and process documents first.")
        return self.vector_store.as_retriever(search_kwargs={"k": k})
