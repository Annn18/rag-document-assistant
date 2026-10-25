from typing import Any, Dict, List, Optional
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from src.config import DEFAULT_HF_ENDPOINT_MODEL, DEFAULT_LOCAL_LLM, DEFAULT_TOP_K, HF_API_TOKEN
from src.vector_store import VectorStoreManager

# Prompt instructing the LLM to ground its answer strictly on context and cite sources
QA_PROMPT_TEMPLATE = """You are a helpful and accurate document Q&A assistant.
Use ONLY the provided context below to answer the user's question.
If the answer is not contained in the context, state: "I cannot find the answer to this question in the uploaded documents."

Guidelines:
1. Base your answer solely on the provided context. Do NOT make up information.
2. For each key point, cite the document and page/slide number in brackets, for example: [Doc: {sample_doc}, Page: {sample_page}].
3. Be clear, concise, and structured.

Context:
{context}

Question: {question}

Helpful Answer:"""


class RAGPipeline:
    """
    End-to-end RAG Pipeline coordinating retrieval and LLM answer generation
    with source and page-level citations.
    """

    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        llm_type: str = "local",  # "local" or "huggingface_hub"
        hf_api_token: Optional[str] = None,
        model_name: Optional[str] = None,
        top_k: int = DEFAULT_TOP_K,
    ):
        self.vector_store_manager = vector_store_manager
        self.llm_type = llm_type
        self.hf_api_token = hf_api_token or HF_API_TOKEN
        self.model_name = model_name
        self.top_k = top_k
        self.llm = None
        self._initialize_llm()

    def _initialize_llm(self):
        """Initializes the configured LLM engine with fallback and error recording."""
        self.init_error = None
        try:
            if self.llm_type == "huggingface_hub":
                if not self.hf_api_token:
                    self.init_error = "Hugging Face API token is required for HuggingFace Hub mode. Please provide your token in the sidebar or set HUGGINGFACEHUB_API_TOKEN in .env."
                    self.llm = None
                    return

                from langchain_huggingface import HuggingFaceEndpoint

                target_model = self.model_name or DEFAULT_HF_ENDPOINT_MODEL
                self.llm = HuggingFaceEndpoint(
                    repo_id=target_model,
                    huggingfacehub_api_token=self.hf_api_token,
                    temperature=0.1,
                    max_new_tokens=512,
                )
            else:
                # Local pipeline using transformers & lightweight CPU model (default: google/flan-t5-base)
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
                from langchain_huggingface import HuggingFacePipeline

                target_model = self.model_name or DEFAULT_LOCAL_LLM
                tokenizer = AutoTokenizer.from_pretrained(target_model)
                model = AutoModelForSeq2SeqLM.from_pretrained(target_model)
                pipe = pipeline(
                    "text2text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=512,
                    temperature=0.1,
                    repetition_penalty=1.1,
                )
                self.llm = HuggingFacePipeline(pipeline=pipe)
        except Exception as e:
            self.init_error = str(e)
            self.llm = None

    def set_llm_mode(self, llm_type: str, hf_api_token: Optional[str] = None, model_name: Optional[str] = None):
        """Allows dynamic switching between local LLM and Hugging Face API."""
        self.llm_type = llm_type
        if hf_api_token:
            self.hf_api_token = hf_api_token
        if model_name:
            self.model_name = model_name
        self._initialize_llm()

    def format_context(self, docs: List[Document]) -> str:
        """Formats retrieved chunks into a standardized context block with explicit citation headers."""
        context_parts = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown Document")
            page = doc.metadata.get("page", 1)
            doc_type = doc.metadata.get("doc_type", "").upper()
            label = f"Slide: {page}" if doc_type == "PPTX" else f"Page: {page}"
            
            chunk_header = f"[Source: {source} | {label}]"
            context_parts.append(f"{chunk_header}\n{doc.page_content.strip()}")

        return "\n\n---\n\n".join(context_parts)

    def answer_query(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Executes the RAG pipeline:
        1. Retrieves top_k similar chunks from FAISS.
        2. Formats retrieved chunks with citation metadata.
        3. Prompts LLM to synthesize answer with citations.
        4. Returns answer and structured citations list.
        """
        k = top_k or self.top_k
        docs_and_scores = self.vector_store_manager.similarity_search_with_score(query, k=k)
        
        if not docs_and_scores:
            return {
                "answer": "No relevant documents found. Please upload documents first.",
                "sources": [],
                "query": query,
            }

        docs = [item[0] for item in docs_and_scores]
        scores = [item[1] for item in docs_and_scores]

        # Prepare context
        context_text = self.format_context(docs)

        # Build prompt
        prompt = PromptTemplate(
            template=QA_PROMPT_TEMPLATE,
            input_variables=["context", "question", "sample_doc", "sample_page"],
        )
        sample_doc = docs[0].metadata.get("source", "file.pdf")
        sample_page = docs[0].metadata.get("page", 1)
        formatted_prompt = prompt.format(
            context=context_text,
            question=query,
            sample_doc=sample_doc,
            sample_page=sample_page,
        )

        # Generate answer from LLM
        if self.llm is None:
            err_reason = self.init_error or "LLM engine is not initialized."
            answer_text = (
                f"⚠️ **LLM Generation Unavailable**: {err_reason}\n\n"
                "**Retrieved Context Summary**:\n"
                f"Found {len(docs)} relevant passage(s) matching your query. "
                "See the cited sources and excerpts below for the document answers."
            )
        else:
            try:
                raw_response = self.llm.invoke(formatted_prompt)
                answer_text = raw_response if isinstance(raw_response, str) else getattr(raw_response, "content", str(raw_response))
            except Exception as e:
                answer_text = (
                    f"⚠️ **Error during LLM generation**: {str(e)}\n\n"
                    "The retrieved document passages are available below for reference."
                )

        # Build structured sources for UI citation cards
        sources: List[Dict[str, Any]] = []
        for doc, score in zip(docs, scores):
            source_meta = {
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", 1),
                "total_pages": doc.metadata.get("total_pages", 1),
                "doc_type": doc.metadata.get("doc_type", "doc"),
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "score": round(float(score), 4),
                "content": doc.page_content.strip(),
            }
            sources.append(source_meta)

        return {
            "answer": answer_text.strip(),
            "sources": sources,
            "query": query,
        }
