from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

class DocumentSplitter:
    """
    Splits extracted documents into semantic chunks while preserving source
    and page/slide metadata for citations.
    """

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits a list of Document objects into smaller chunk Documents,
        attaching chunk indices and verifying metadata preservation.
        """
        raw_chunks = self.splitter.split_documents(documents)
        
        # Add a unique chunk index to each chunk for precise traceability
        processed_chunks: List[Document] = []
        for idx, chunk in enumerate(raw_chunks):
            source = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", 1)
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["chunk_id"] = f"{source}_p{page}_c{idx}"
            processed_chunks.append(chunk)

        return processed_chunks
