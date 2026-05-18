RAG Information Retrieval Study Assistant
A Retrieval-Augmented Generation (RAG) system that answers questions based on your IR lecture notes and slides. Upload your PDFs and PPTs, and query them conversationally through a clean chat interface.

Tech Stack
LayerTechnologyFrontendHTML, CSS, JavaScriptBackendFastAPIRAG FrameworkLangChainVector StoreFAISSEmbedding ModelHuggingFace Embeddings (sentence-transformers)LLMGroq (Llama3)ServerUvicorn

Project Structure
IR_RAG_Assistant/
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py
│   ├── rag_pipeline/
│   │   ├── config.py
│   │   ├── ingest.py
│   │   ├── rag_pipeline.py
│   │   └── main.py
│   ├── services/
│   │   ├── __init__.py
│   │   └── qa_services.py
│   ├── static/
│   │   └── index.html
│   └── main.py
├── data/
│   └── raw/          ← Drop your PDFs and PPTs here
├── vector_db/        ← Auto-generated after ingestion
├── requirements.txt
└── README.md

Running Locally
1. Clone the repository
bashgit clone https://github.com/yourusername/IR_RAG_Assistant.git
cd IR_RAG_Assistant
2. Create and activate a virtual environment
bashpython -m venv venv

# Windows
venv\Scripts\activate
3. Install dependencies
bashpip install -r requirements.txt
4. Set up environment variables
Add your Groq API key to app/core/config.py:
pythonGROQ_API_KEY = "your_groq_api_key_here"
Get a free key at console.groq.com
5. Add your study materials
Drop your .pdf and .pptx files into the data/raw/ folder.
6. Run ingestion
bashpython app/rag_pipeline/ingest.py
This processes your documents, generates embeddings using HuggingFace, and saves the FAISS vector database to vector_db/.
7. Start the backend server
bashpython -m uvicorn app.main:app --reload
8. Open the frontend
Go to http://127.0.0.1:8000 in your browser.

Adding New Study Materials

Drop new .pdf or .pptx files into data/raw/
Re-run ingestion to rebuild the vector database:

bashpython app/rag_pipeline/ingest.py

Restart the server:

bashpython -m uvicorn app.main:app --reload
Evaluating Embedding Quality
After ingestion, you can test how well your embedding model retrieves relevant chunks by querying the assistant with specific questions from your notes. Good signs:

Answers reference the correct topic/slide
Sources listed at the bottom match the expected document
Answers are specific, not generic

If answers seem off, try:

Reducing chunk_size in ingest.py (e.g. from 500 to 300)
Increasing k in rag_pipeline.py (retrieve more chunks, e.g. k=5)
Adding more source documents to data/raw/


Hosted Application

Coming soon — deployment on Render (backend) and Netlify (frontend).


API Endpoints
MethodEndpointDescriptionGET/Serves the frontendPOST/askAccepts a question, returns an answer
Request body for /ask:
json{
  "question": "What is TF-IDF?"
}
Response:
json{
  "answer": "TF-IDF stands for..."
}
