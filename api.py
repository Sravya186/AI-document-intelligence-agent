from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import tempfile
import os

from rag import (
    extract_text_from_pdf,
    create_chunks,
    create_embeddings,
    create_vector_database,
    create_llm
)

from agent import run_agent


app = FastAPI(
    title="AI Document Intelligence API",
    description="RAG-based AI Document Intelligence and Decision Agent",
    version="1.0.0"
)


# -----------------------------
# Global application state
# -----------------------------

vector_database = None
llm = None


# -----------------------------
# Request Models
# -----------------------------

class QuestionRequest(BaseModel):
    question: str


class AnalysisRequest(BaseModel):
    situation: str


class ChatRequest(BaseModel):
    question: str
    chat_history: list = []


# -----------------------------
# Health Check
# -----------------------------

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "service": "AI Document Intelligence API"
    }


# -----------------------------
# Upload PDF
# -----------------------------

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    global vector_database
    global llm

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    try:

        file_bytes = await file.read()

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(file_bytes)
            temp_pdf_path = temp_file.name

        # Extract PDF text
        documents = extract_text_from_pdf(temp_pdf_path)

        if not documents:
            raise HTTPException(
                status_code=400,
                detail="No readable text found in the PDF."
            )

        # Create chunks
        chunks = create_chunks(documents)

        # Create embeddings
        embeddings = create_embeddings()

        # Create FAISS database
        vector_database = create_vector_database(
            chunks,
            embeddings
        )

        # Create LLM
        llm = create_llm()

        # Remove temporary file
        os.remove(temp_pdf_path)

        return {
            "message": "Document uploaded successfully.",
            "filename": file.filename,
            "pages": len(documents),
            "chunks": len(chunks)
        }

    except Exception as e:

        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# -----------------------------
# Ask Question
# -----------------------------

@app.post("/ask")
def ask_question(request: QuestionRequest):

    if vector_database is None or llm is None:

        raise HTTPException(
            status_code=400,
            detail="Please upload a document first."
        )

    try:

        answer, pages = run_agent(
            question=request.question,
            mode="question",
            vector_database=vector_database,
            llm=llm,
            chat_history=[]
        )

        return {
            "answer": answer,
            "source_pages": pages
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# -----------------------------
# Analyze Situation
# -----------------------------

@app.post("/analyze")
def analyze_situation(request: AnalysisRequest):

    if vector_database is None or llm is None:

        raise HTTPException(
            status_code=400,
            detail="Please upload a document first."
        )

    try:

        answer, pages = run_agent(
            question=request.situation,
            mode="analysis",
            vector_database=vector_database,
            llm=llm,
            chat_history=[]
        )

        return {
            "analysis": answer,
            "source_pages": pages
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# -----------------------------
# Chat
# -----------------------------

@app.post("/chat")
def chat(request: ChatRequest):

    if vector_database is None or llm is None:

        raise HTTPException(
            status_code=400,
            detail="Please upload a document first."
        )

    try:

        answer, pages = run_agent(
            question=request.question,
            mode="chat",
            vector_database=vector_database,
            llm=llm,
            chat_history=request.chat_history
        )

        return {
            "answer": answer,
            "source_pages": pages
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )