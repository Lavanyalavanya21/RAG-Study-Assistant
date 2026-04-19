from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from app.core.config import EMBEDDING_MODEL, LLM_MODEL, GROQ_API_KEY

VECTOR_BASE_PATH = "vectorstore"


def load_pipeline(subject_id: str):
    vector_path = f"{VECTOR_BASE_PATH}/{subject_id}"
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.load_local(vector_path, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY)

    prompt = PromptTemplate.from_template(
        "Use the following context to answer the question.\n\n"
        "Context: {context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def ask_question(query: str, subject_id: str):
    chain, retriever = load_pipeline(subject_id)
    answer = chain.invoke(query)
    docs = retriever.invoke(query)

    source_text = "\n\n📚 Sources:\n"
    for doc in docs:
        meta = doc.metadata
        source_text += f"- {meta.get('source', 'Unknown')} (Slide/Page: {meta.get('slide', 'N/A')})\n"

    return answer + source_text