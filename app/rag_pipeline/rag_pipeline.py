import re
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from app.core.config import EMBEDDING_MODEL, LLM_MODEL, GROQ_API_KEY

VECTOR_BASE_PATH = "vectorstore"


# ---------------------------------------------------------------------------
# Detect if the query mentions a specific source filename
# ---------------------------------------------------------------------------

def detect_source_filter(query: str, db: FAISS) -> str | None:
    """
    Scan all document metadata in the FAISS index.
    If the query mentions tokens that match a source filename, return that filename.
    e.g. query "syllabus of unit3 ppt" matches source "Unit3.pptx"
    """
    query_lower = query.lower()

    # Collect all unique source filenames
    all_sources = set()
    for doc_id in db.docstore._dict:
        doc = db.docstore._dict[doc_id]
        source = doc.metadata.get("source", "")
        if source:
            all_sources.add(source)

    best_match = None
    best_score = 0

    for source in all_sources:
        # Normalize filename: remove extension, lowercase, split on spaces/underscores/hyphens
        name = re.sub(r'\.(pptx|pdf|docx|png|jpg|jpeg)$', '', source.lower())
        tokens = re.split(r'[\s_\-]+', name)

        # Score = number of filename tokens found in the query
        score = sum(1 for token in tokens if token and len(token) > 1 and token in query_lower)

        if score > best_score:
            best_score = score
            best_match = source

    return best_match if best_score >= 1 else None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def ask_question(query: str, subject_id: str):
    vector_path = f"{VECTOR_BASE_PATH}/{subject_id}"
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.load_local(vector_path, embeddings, allow_dangerous_deserialization=True)

    source_filter = detect_source_filter(query, db)

    # Detect if this is a "full content" query needing all chunks
    full_content_keywords = ["syllabus", "all topics", "full content", "overview",
                             "summarize all", "list all", "what all", "entire", "complete"]
    is_full_content_query = any(kw in query.lower() for kw in full_content_keywords)

    if source_filter and is_full_content_query:
        # Fetch ALL chunks from this source, not just top-k
        all_docs = list(db.docstore._dict.values())
        filtered_docs = [
            doc for doc in all_docs
            if doc.metadata.get("source") == source_filter
            and doc.metadata.get("type") == "text"  # skip image descriptions for syllabus
        ]
        # Sort by page/slide number for coherent reading order
        filtered_docs.sort(key=lambda d: d.metadata.get("slide") or d.metadata.get("page") or 0)
        context = "\n\n".join(doc.page_content for doc in filtered_docs)
        docs_for_sources = filtered_docs

        llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY)
        prompt_text = (
            f"You are a study assistant. Using ALL the content below from {source_filter}, "
            f"answer the question thoroughly.\n\n"
            f"Content:\n{context}\n\n"
            f"Question: {query}\n\nAnswer:"
        )
        answer = llm.invoke(prompt_text).content

    else:
        # Normal retrieval path
        if source_filter:
            retriever = db.as_retriever(
                search_kwargs={"k": 6, "filter": {"source": source_filter}}
            )
        else:
            retriever = db.as_retriever(search_kwargs={"k": 5})

        llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY)
        source_hint = f"Answer only using content from the file: {source_filter}\n\n" if source_filter else ""

        prompt = PromptTemplate.from_template(
            "You are a study assistant helping a student understand their notes.\n\n"
            "The context below may include:\n"
            "- Text from lecture slides or PDFs\n"
            "- Descriptions of diagrams and figures (prefixed with [Figure ...])\n"
            "- Numerical values, formulas, or equations extracted from figures\n\n"
            "Instructions:\n"
            "1. Use ALL relevant context, including figure descriptions, to answer.\n"
            "2. If the answer involves a diagram or figure, describe what it shows.\n"
            "3. If numerical values or formulas are present in the context, include them accurately.\n"
            "4. If the context does not contain enough information, say so clearly.\n"
            "5. Keep the answer focused and student-friendly.\n\n"
            "{source_hint}"
            "Context:\n{context}\n\n"
            "Question: {question}\n\nAnswer:"
        )

        chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough(),
                "source_hint": RunnableLambda(lambda _: source_hint)
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        answer = chain.invoke(query)
        docs_for_sources = retriever.invoke(query)

    # Build sources footer
    source_text = "\n\n📚 Sources:\n"
    seen = set()
    for doc in docs_for_sources:
        meta = doc.metadata
        chunk_type     = meta.get("type", "text")
        source         = meta.get("source", "Unknown")
        location       = meta.get("slide") or meta.get("page") or "N/A"
        icon           = "🖼️" if chunk_type == "image_description" else "📄"
        key            = f"{source}-{location}-{chunk_type}"
        if key not in seen:
            seen.add(key)
            location_label = "Slide" if "slide" in meta else "Page"
            source_text += f"  {icon} {source} ({location_label}: {location})\n"

    if source_filter:
        source_text += f"\n  🔍 Filtered to: {source_filter}"

    return answer + source_text