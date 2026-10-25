import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if available
load_dotenv()

# Base project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = DATA_DIR / "vectorstore"
UPLOAD_DIR = DATA_DIR / "uploads"

# Ensure directories exist
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Chunking Configuration
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

# Retrieval Configuration
DEFAULT_TOP_K = 4

# Embeddings Model Configuration
# sentence-transformers/all-MiniLM-L6-v2 is lightweight (80MB), fast on CPU, and high quality
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# HuggingFace LLM Configuration
DEFAULT_LOCAL_LLM = "google/flan-t5-base"
DEFAULT_HF_ENDPOINT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# HuggingFace API Token
HF_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
