from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel
from app.services.qa_services import get_answer
from app.services.ingest_service import ingest_file
import shutil, os

router = APIRouter()

class Query(BaseModel):
    question: str
    subject_id: str

@router.post("/ask")
def ask(query: Query):
    answer = get_answer(query.question, query.subject_id)
    return {"answer": answer}

@router.post("/upload")
async def upload(file: UploadFile = File(...), subject_id: str = Form(...)):
    temp_path = f"temp_{file.filename}"
    original_name = file.filename
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    ingest_file(temp_path, subject_id, original_name=file.filename)
    os.remove(temp_path)
    
    return {"message": f"{file.filename} ingested successfully"}