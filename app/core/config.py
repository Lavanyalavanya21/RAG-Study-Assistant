import os
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH       = os.path.join(BASE_DIR, "data", "raw")
VECTOR_DB_PATH  = os.path.join(BASE_DIR, "vectorstore", "faiss_index")

# Embedding + LLM
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL       = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")