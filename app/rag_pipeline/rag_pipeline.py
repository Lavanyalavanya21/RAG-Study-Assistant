import re
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import EMBEDDING_MODEL, LLM_MODEL, GROQ_API_KEY, SUPABASE_URL, SUPABASE_KEY

# ---------------------------------------------------------------------------
# Lazy clients
# ---------------------------------------------------------------------------

_embeddings = None
_supabase_client = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def get_supabase():
    global _supabase_client
    if _supabase_client is None:
        from supabase import create_client
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase_client


# ---------------------------------------------------------------------------
# pgvector similarity search
# ---------------------------------------------------------------------------

def similarity_search(query: str, subject_id: str, k: int = 5, source_filter: str = None) -> list[Document]:
    """
    Embed the query and run a cosine similarity search against the documents table.
    Optionally filter by source filename.
    """
    embedding = get_embeddings().embed_query(query)
    sb = get_supabase()

    # Call the match_documents RPC function (defined in Supabase SQL below)
    params = {
        "query_embedding": embedding,
        "match_subject_id": subject_id,
        "match_count": k,
    }
    if source_filter:
        params["match_source"] = source_filter

    rpc_name = "match_documents_filtered" if source_filter else "match_documents"
    result = sb.rpc(rpc_name, params).execute()

    docs = []
    for row in (result.data or []):
        docs.append(Document(
            page_content=row["content"],
            metadata=row.get("metadata", {})
        ))
    return docs


# ---------------------------------------------------------------------------
# Fetch ALL chunks for a subject+source (full content queries)
# ---------------------------------------------------------------------------

def fetch_all_chunks(subject_id: str, source_filter: str) -> list[Document]:
    """
    Retrieve every text chunk for a given subject and source file,
    sorted by page/slide number for coherent reading order.
    """
    sb = get_supabase()
    result = sb.table("documents") \
        .select("content, metadata") \
        .eq("subject_id", subject_id) \
        .execute()

    docs = []
    for row in (result.data or []):
        meta = row.get("metadata", {})
        if (meta.get("source") == source_filter
                and meta.get("type") == "text"):
            docs.append(Document(page_content=row["content"], metadata=meta))

    # Sort by slide or page number
    docs.sort(key=lambda d: d.metadata.get("slide") or d.metadata.get("page") or 0)
    return docs


# ---------------------------------------------------------------------------
# Detect if query targets a specific source file
# ---------------------------------------------------------------------------

def detect_source_filter(query: str, subject_id: str) -> str | None:
    """
    Fetch all unique source filenames for the subject from Supabase,
    then match against query tokens.
    e.g. query "syllabus of unit3 ppt" matches source "Unit3.pptx"
    """
    sb = get_supabase()
    result = sb.table("documents") \
        .select("metadata") \
        .eq("subject_id", subject_id) \
        .execute()

    all_sources = set()
    for row in (result.data or []):
        source = row.get("metadata", {}).get("source", "")
        if source:
            all_sources.add(source)

    query_lower = query.lower()
    best_match = None
    best_score = 0

    for source in all_sources:
        name = re.sub(r'\.(pptx|pdf|docx|png|jpg|jpeg)$', '', source.lower())
        tokens = re.split(r'[\s_\-]+', name)
        score = sum(1 for token in tokens if token and len(token) > 1 and token in query_lower)
        if score > best_score:
            best_score = score
            best_match = source

    return best_match if best_score >= 1 else None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def ask_question(query: str, subject_id: str) -> str:
    source_filter = detect_source_filter(query, subject_id)

    full_content_keywords = [
        "syllabus", "all topics", "full content", "overview",
        "summarize all", "list all", "what all", "entire", "complete"
    ]
    is_full_content_query = any(kw in query.lower() for kw in full_content_keywords)

    llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY)

    if source_filter and is_full_content_query:
        # Fetch ALL text chunks from this source in reading order
        filtered_docs = fetch_all_chunks(subject_id, source_filter)

        if not filtered_docs:
            return f"No content found for '{source_filter}' in this subject."

        context = "\n\n".join(doc.page_content for doc in filtered_docs)
        prompt_text = (
            f"You are a study assistant. Using ALL the content below from {source_filter}, "
            f"answer the question thoroughly.\n\n"
            f"Content:\n{context}\n\n"
            f"Question: {query}\n\nAnswer:"
        )
        answer = llm.invoke(prompt_text).content
        docs_for_sources = filtered_docs

    else:
        # Normal retrieval: cosine similarity search
        k = 6 if source_filter else 5
        docs_for_sources = similarity_search(query, subject_id, k=k, source_filter=source_filter)

        if not docs_for_sources:
            return "No relevant content found. Please upload notes for this subject first."

        context = "\n\n".join(doc.page_content for doc in docs_for_sources)
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
                "context": RunnableLambda(lambda _: context),
                "question": RunnablePassthrough(),
                "source_hint": RunnableLambda(lambda _: source_hint),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        answer = chain.invoke(query)

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