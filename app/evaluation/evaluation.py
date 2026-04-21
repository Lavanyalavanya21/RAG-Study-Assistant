import os
import sys
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

# ── CONFIG ──
SUBJECT_ID = "7f255c91-5679-4a14-afc1-28492b825dd7"  # paste your subject ID from Supabase
VECTOR_PATH = f"vectorstore/{SUBJECT_ID}"

# ── TEST QUESTIONS ──
# Add your own questions and reference answers based on your notes
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
        "ground_truth": " Any table of numbers can be broken into three simpler pieces that are easier to understand.Imagine you have a complicated machine. Instead of trying to understand it all at once, you take it apart into:A part that rotates or rearranges things A part that stretches or shrinks things Another part that rotates again That’s exactly what SVD does to a matrix (a grid of numbers).The simple formula 𝐴=𝑈Σ𝑉^𝑇 A → your original data U → directions of important patterns Σ (Sigma) → how strong each pattern is (stretching amount)Vᵀ → how the data is combined. Eigenvector for λ = 4, Step 1: Subtract 4, Step 2: Solve, Step 3: Choose value, Step 4: Normalize, Step 5: Find U, "
    },
    {
        "question": "Comparison with Traditional Keyword-based Search Engines",
        "ground_truth": "1. Semantic Relevance vs. Exact Match: Traditional keyword-based search engines rely on exact keyword matching to retrieve documents, which may lead to issues such as synonymy, polysemy, and poor retrieval of semantically related documents. In contrast, LSI search engines consider the semantic similarity between terms and documents, allowing them to retrieve relevant documents even if they do not contain exact keyword matches. 2. Robustness to Vocabulary Variations: LSI search engines are more robust to variations in vocabulary, spelling errors, and linguistic nuances compared to traditional keyword-based search engines. They can capture the underlying semantic meaning of terms and recognize documents that use different terms to express similar concepts. 3. Scalability and Efficiency: While LSI-based search engines require additional computational overhead for performing SVD and representing documents in the latent semantic space, they can scale efficiently to large document collections. Traditional keyword-based search engines may struggle with scalability due to the need to match each query term against the entire document collection. In summary, LSI-based search engines offer improved retrieval accuracy by considering the semantic relationships between terms and documents, making them a valuable alternative to traditional keyword-based search engines, especially in scenarios where precise keyword matches are not sufficient for capturing document relevance."
    },

]

def get_contexts(question: str, retriever):
    docs = retriever.invoke(question)
    return [doc.page_content for doc in docs]

def get_answer(question: str, chain):
    return chain.invoke(question)

def main():
    print("🔄 Loading pipeline...")
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

    print("📝 Running evaluation questions...")
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

    # Build RAGAS dataset
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    print("\n🧠 Running RAGAS evaluation...")
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

    print("\n📊 RAGAS Evaluation Results:")
    print("=" * 50)
    df = results.to_pandas()
    print(df[["question", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]].to_string())
    print("\n📈 Average Scores:")
    print(f"  Faithfulness:      {df['faithfulness'].mean():.3f}")
    print(f"  Answer Relevancy:  {df['answer_relevancy'].mean():.3f}")
    print(f"  Context Precision: {df['context_precision'].mean():.3f}")
    print(f"  Context Recall:    {df['context_recall'].mean():.3f}")

    # Save results to CSV
    df.to_csv("app/evaluation/evaluation_results.csv", index=False)
    print("\n✅ Results saved to app/evaluation/evaluation_results.csv")

if __name__ == "__main__":
    main()