import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from pptx import Presentation

from config import DATA_PATH, VECTOR_DB_PATH, EMBEDDING_MODEL


# -----------------------------
# 📄 Load PDF files
# -----------------------------
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()


# -----------------------------
# 📊 Load PPTX files
# -----------------------------
def load_pptx(file_path):
    prs = Presentation(file_path)
    docs = []

    for i, slide in enumerate(prs.slides):
        text = ""

        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text += shape.text + " "

        if text.strip():
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": os.path.basename(file_path),
                        "slide": i + 1
                    }
                )
            )

    return docs


# -----------------------------
# 📥 Load all documents
# -----------------------------
def load_documents():
    docs = []

    for file in os.listdir(DATA_PATH):
        file_path = os.path.join(DATA_PATH, file)

        if file.endswith(".pdf"):
            print(f"📄 Loading PDF: {file}")
            docs.extend(load_pdf(file_path))

        elif file.endswith(".pptx"):
            print(f"📊 Loading PPTX: {file}")
            docs.extend(load_pptx(file_path))

    return docs


# -----------------------------
# ✂️ Split into chunks
# -----------------------------
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = []
    for doc in documents:
        split_texts = splitter.split_text(doc.page_content)

        for chunk in split_texts:
            chunks.append(
                Document(
                    page_content=chunk,
                    metadata=doc.metadata  # preserve metadata
                )
            )

    return chunks


# -----------------------------
# 🧠 Create and save vector DB
# -----------------------------
def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    os.makedirs(VECTOR_DB_PATH, exist_ok=True)

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTOR_DB_PATH)


# -----------------------------
# 🚀 Main function
# -----------------------------
def main():
    print("📥 Loading documents...")
    documents = load_documents()

    if not documents:
        print("❌ No documents found in data/raw/")
        return

    print(f"✅ Loaded {len(documents)} documents")

    print("✂️ Splitting documents into chunks...")
    chunks = split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks")

    print("🧠 Creating embeddings and saving vector database...")
    create_vectorstore(chunks)

    print("🎉 Ingestion complete!")


# -----------------------------
# ▶️ Run script
# -----------------------------
if __name__ == "__main__":
    main()