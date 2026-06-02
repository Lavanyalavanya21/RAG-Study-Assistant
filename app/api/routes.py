from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from app.services.qa_services import get_answer
from app.services.ingest_service import ingest_file
import shutil
import os

router = APIRouter()


class Query(BaseModel):
    question: str
    subject_id: str


# ---------------------------------------------------------------------------
# Ask a question
# ---------------------------------------------------------------------------

@router.post("/ask")
def ask(query: Query):
    try:
        answer = get_answer(query.question, query.subject_id)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Upload + ingest a file
# ---------------------------------------------------------------------------

@router.post("/upload")
async def upload(file: UploadFile = File(...), subject_id: str = Form(...)):
    allowed_extensions = {".pdf", ".pptx", ".docx", ".png", ".jpg", ".jpeg"}
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(allowed_extensions)}"
        )

    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = ingest_file(temp_path, subject_id, original_name=file.filename)

        return {
            "message": f"{file.filename} ingested successfully",
            "subject_id": subject_id,
            "chunks_created": result.get("chunks", 0),
            "text_docs": result.get("text_docs", 0),
            "image_descriptions": result.get("image_descriptions", 0),
            "storage_url": result.get("storage_url", "")
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ---------------------------------------------------------------------------
# Check ingestion status for a subject
# ---------------------------------------------------------------------------

@router.get("/status/{subject_id}")
def ingestion_status(subject_id: str):
    try:
        from app.services.ingest_service import get_supabase
        sb = get_supabase()
        result = sb.table("documents") \
            .select("id", count="exact") \
            .eq("subject_id", subject_id) \
            .execute()

        count = result.count or 0
        if count == 0:
            return {
                "subject_id": subject_id,
                "status": "not_found",
                "message": "No notes ingested for this subject yet."
            }

        return {
            "subject_id": subject_id,
            "status": "ready",
            "chunks": count,
            "message": f"{count} chunks ingested and ready for querying."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Health check (used by Render)
# ---------------------------------------------------------------------------

@router.get("/health")
def health():
    return {"status": "ok"}