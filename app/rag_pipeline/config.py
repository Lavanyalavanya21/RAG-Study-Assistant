import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "raw")
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vectorstore", "faiss_index")

# Models
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "mistral"