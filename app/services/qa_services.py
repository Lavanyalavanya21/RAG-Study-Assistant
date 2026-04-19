from app.rag_pipeline.rag_pipeline import ask_question

def get_answer(question: str, subject_id: str):
    result = ask_question(question, subject_id)
    return {"answer": result}