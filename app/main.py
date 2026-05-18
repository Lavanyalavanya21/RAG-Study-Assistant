from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router

app = FastAPI(title="RAG IR Study-Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://rag-study-assistant.netlify.app"],  # ← your Netlify URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)