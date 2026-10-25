import io
from pathlib import Path
from typing import List, Union
from langchain_core.documents import Document
import pypdf
import docx
import pptx

class DocumentLoader:
    """
    Multi-format document loader supporting PDF, DOCX, and PPTX files.
    Extracts text page-by-page or slide-by-slide to retain precise citation metadata.
    """

    @classmethod
    def load_pdf(cls, file_source: Union[str, Path, io.BytesIO], filename: str) -> List[Document]:
        """Extracts text from a PDF file page by page."""
        documents: List[Document] = []
        reader = pypdf.PdfReader(file_source)
        total_pages = len(reader.pages)

        for page_idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": filename,
                        "page": page_idx,
                        "total_pages": total_pages,
                        "doc_type": "pdf",
                    },
                )
                documents.append(doc)
        return documents

    @classmethod
    def load_docx(cls, file_source: Union[str, Path, io.BytesIO], filename: str) -> List[Document]:
        """
        Extracts text from a DOCX file.
        Groups paragraphs and table cells, estimating page/section breaks.
        """
        doc_obj = docx.Document(file_source)
        documents: List[Document] = []
        
        # Word documents lack strict page layout without rendering,
        # so we track page break elements, headings, and character count estimation
        current_page = 1
        page_texts: List[str] = []
        char_count = 0
        PAGE_CHAR_THRESHOLD = 2500  # Approx. 400-500 words per page

        def flush_page(p_num: int):
            content = "\n".join(page_texts).strip()
            if content:
                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": filename,
                            "page": p_num,
                            "doc_type": "docx",
                        },
                    )
                )
            page_texts.clear()

        for para in doc_obj.paragraphs:
            text = para.text.strip()
            if not text:
                continue

            # Check for explicit page break in runs
            has_break = False
            for run in para.runs:
                if "\f" in run.text or "lastRenderedPageBreak" in run._element.xml:
                    has_break = True
                    break

            if has_break and page_texts:
                flush_page(current_page)
                current_page += 1
                char_count = 0

            page_texts.append(text)
            char_count += len(text)

            # Heuristic page split if paragraph is very long
            if char_count >= PAGE_CHAR_THRESHOLD:
                flush_page(current_page)
                current_page += 1
                char_count = 0

        # Also extract tables
        for table in doc_obj.tables:
            table_rows = []
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    table_rows.append(row_text)
            if table_rows:
                table_content = "[Table Content]\n" + "\n".join(table_rows)
                page_texts.append(table_content)

        if page_texts:
            flush_page(current_page)

        # Total pages metadata backfill
        total_pages = len(documents)
        for doc in documents:
            doc.metadata["total_pages"] = max(total_pages, 1)

        return documents

    @classmethod
    def load_pptx(cls, file_source: Union[str, Path, io.BytesIO], filename: str) -> List[Document]:
        """Extracts text from a PPTX presentation slide by slide."""
        prs = pptx.Presentation(file_source)
        documents: List[Document] = []
        total_slides = len(prs.slides)

        for slide_idx, slide in enumerate(prs.slides, start=1):
            slide_texts: List[str] = []
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for paragraph in shape.text_frame.paragraphs:
                        text = paragraph.text.strip()
                        if text:
                            slide_texts.append(text)
                elif shape.has_table:
                    for row in shape.table.rows:
                        row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                        if row_text:
                            slide_texts.append(row_text)

            content = "\n".join(slide_texts).strip()
            if content:
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": filename,
                        "page": slide_idx,
                        "total_pages": total_slides,
                        "doc_type": "pptx",
                    },
                )
                documents.append(doc)

        return documents

    @classmethod
    def load_document(cls, file_source: Union[str, Path, io.BytesIO], filename: str) -> List[Document]:
        """Routes file to the proper format extractor based on extension."""
        ext = Path(filename).suffix.lower()
        if ext == ".pdf":
            return cls.load_pdf(file_source, filename)
        elif ext == ".docx":
            return cls.load_docx(file_source, filename)
        elif ext in [".pptx", ".ppt"]:
            return cls.load_pptx(file_source, filename)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Supported formats are .pdf, .docx, .pptx")
