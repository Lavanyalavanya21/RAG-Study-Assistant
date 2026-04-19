import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pptx import Presentation
from app.core.config import EMBEDDING_MODEL

VECTOR_BASE_PATH = "vectorstore"


def get_vector_path(subject_id: str):
    return os.path.join(VECTOR_BASE_PATH, subject_id)


def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()


def load_pptx(file_path):
    prs = Presentation(file_path)
    docs = []
    for i, slide in enumerate(prs.slides):
        text = " ".join(
            shape.text for shape in slide.shapes if hasattr(shape, "text")
        )
        if text.strip():
            docs.append(Document(
                page_content=text,
                metadata={"source": os.path.basename(file_path), "slide": i + 1}
            ))
    return docs


def ingest_file(file_path: str, subject_id: str):
    if file_path.endswith(".pdf"):
        docs = load_pdf(file_path)
    elif file_path.endswith(".pptx"):
        docs = load_pptx(file_path)
    else:
        raise ValueError("Unsupported file type")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_path = get_vector_path(subject_id)

    # If vector store exists for this subject, merge. Otherwise create new.
    if os.path.exists(vector_path):
        db = FAISS.load_local(vector_path, embeddings, allow_dangerous_deserialization=True)
        db.add_documents(chunks)
    else:
        db = FAISS.from_documents(chunks, embeddings)

    db.save_local(vector_path)