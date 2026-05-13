import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from app.core.config import EMBEDDING_MODEL, LLM_MODEL, GROQ_API_KEY

#CONFIG
SUBJECT_ID = "7f255c91-5679-4a14-afc1-28492b825dd7"
VECTOR_PATH = f"vectorstore/{SUBJECT_ID}"

# TEST QUESTIONS
test_data = [
    {
        "question": "What exactly is Information Retrieval?",
        "ground_truth": "Information retrieval is defined as the process of accessing and retrieving the most appropriate information from text based on a particular query given by the user, with the help of context-based indexing or metadata."
    },
    {
        "question": "Explain the architecture and functioning of an LSI-based search engine.",
        "ground_truth": "1. Document Collection, 2. Term-Document Matrix, 3. Singular Value Decomposition (SVD), 4. Dimensionality Reduction, 5. Semantic Space Representation, 6. Ranking of Documents"
    },
    {
        "question": "What is Singular Value Decomposition (SVD), explain the numerical?",
        "ground_truth": "Any table of numbers can be broken into three simpler pieces. The formula A=UΣVᵀ where U represents directions of important patterns, Σ represents how strong each pattern is, and Vᵀ represents how the data is combined."
    },
    {
        "question": "Comparison with Traditional Keyword-based Search Engines",
        "ground_truth": "LSI search engines consider semantic similarity between terms and documents, are more robust to vocabulary variations, and can scale efficiently compared to traditional keyword-based search engines that rely on exact keyword matching."
    },
]


def get_contexts(question: str, retriever):
    docs = retriever.invoke(question)
    return [doc.page_content for doc in docs]


def get_answer(question: str, chain):
    return chain.invoke(question)


def main():
    print("Loading pipeline...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.load_local(VECTOR_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY)

    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

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

    print("Running evaluation questions...")
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for item in test_data:
        q = item["question"]
        print(f"  ➜ {q}")
        answer = get_answer(q, chain)
        context = get_contexts(q, retriever)

        questions.append(q)
        answers.append(answer)
        contexts.append(context)
        ground_truths.append(item["ground_truth"])

        time.sleep(2)  # avoid Groq rate limits

    # Build RAGAS dataset
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    print("\n Running RAGAS evaluation")
    results = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=llm,
        embeddings=embeddings
    )

    print("\nRAGAS Evaluation Results:")
    print("=" * 50)
    df = results.to_pandas()
    print("Available columns:", df.columns.tolist())
    print(df.to_string())

    print("\nAverage Scores:")
    for col in df.columns:
        if df[col].dtype in ['float64', 'float32']:
            print(f"  {col}: {df[col].mean():.3f}")

    df.to_csv("app/evaluation/evaluation_results.csv", index=False)
    print("\n Results saved to app/evaluation/evaluation_results.csv")


if __name__ == "__main__":
    main()