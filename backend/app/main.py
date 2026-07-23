from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.chat_service import answer_question
from app.config import get_settings
from app.db import SessionLocal, get_db, init_db
from app.ingestion import process_dataset
from app.models import ChatJob, ChatMessage, ChatSession, ChatSessionDataset, Dataset, DatasetEmbedding, SemanticCatalog


settings = get_settings()
app = FastAPI(title="DataAI4 MVP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatJobRequest(BaseModel):
    chat_session_id: str
    dataset_id: str
    question: str


class ChatSessionCreateRequest(BaseModel):
    title: str | None = None


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/admin/reset-memory")
def reset_memory(db: Session = Depends(get_db)) -> dict:
    duckdb_dir = settings.storage_dir / "duckdb"
    uploads_dir = settings.storage_dir / "uploads"

    db.query(ChatJob).delete()
    db.query(ChatMessage).delete()
    db.query(ChatSessionDataset).delete()
    db.query(ChatSession).delete()
    db.query(SemanticCatalog).delete()
    db.query(DatasetEmbedding).delete()
    db.execute(Dataset.__table__.delete())
    db.commit()

    for directory in [duckdb_dir, uploads_dir]:
        if not directory.exists():
            continue
        for path in directory.iterdir():
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)

    return {"status": "ok", "detail": "Memoria y archivos temporales reiniciados."}


@app.post("/datasets")
def upload_dataset(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    description: str = Form(""),
    chat_session_id: str = Form(""),
    db: Session = Depends(get_db),
) -> dict:
    if not file.filename or Path(file.filename).suffix.lower() not in {".csv", ".xlsx"}:
        raise HTTPException(status_code=400, detail="Carga un archivo .csv o .xlsx.")
    chat_session = None
    if chat_session_id:
        chat_session = db.get(ChatSession, chat_session_id)
        if not chat_session:
            raise HTTPException(status_code=404, detail="Chat no encontrado.")
    dataset = Dataset(
        filename=file.filename,
        description=description,
        status="uploaded",
        progress_step="uploaded",
        progress_detail="Archivo recibido.",
        progress_current=0,
        progress_total=7,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    if chat_session:
        db.add(ChatSessionDataset(chat_session_id=chat_session.id, dataset_id=dataset.id))
        chat_session.updated_at = dataset.updated_at
        db.commit()
    suffix = Path(file.filename).suffix.lower()
    target = settings.storage_dir / "uploads" / f"{dataset.id}{suffix}"
    with target.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    background_tasks.add_task(process_dataset, dataset.id, target)
    return {"dataset_id": dataset.id, "status": dataset.status, "chat_session_id": chat_session.id if chat_session else None}


@app.get("/datasets")
def list_datasets(chat_session_id: str | None = None, db: Session = Depends(get_db)) -> list[dict]:
    query = db.query(Dataset)
    if chat_session_id:
        query = (
            query.join(ChatSessionDataset, ChatSessionDataset.dataset_id == Dataset.id)
            .filter(ChatSessionDataset.chat_session_id == chat_session_id)
        )
    datasets = query.order_by(Dataset.created_at.desc()).limit(50).all()
    return [dataset_payload(item) for item in datasets]


@app.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str, db: Session = Depends(get_db)) -> dict:
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset no encontrado.")
    return dataset_payload(dataset)


@app.get("/datasets/{dataset_id}/catalog")
def get_catalog(dataset_id: str, db: Session = Depends(get_db)) -> list[dict]:
    rows = (
        db.query(SemanticCatalog)
        .filter(SemanticCatalog.dataset_id == dataset_id)
        .order_by(SemanticCatalog.sheet_name, SemanticCatalog.id)
        .all()
    )
    return [
        {
            "sheet_name": row.sheet_name,
            "duckdb_table": row.duckdb_table,
            "column_name": row.column_name,
            "duckdb_column": row.duckdb_column,
            "data_type": row.data_type,
            "semantic_type": row.semantic_type,
            "sample_values": row.sample_values,
            "metrics": row.metrics_json,
            "is_hidden": row.is_hidden,
        }
        for row in rows
    ]


def dataset_payload(item: Dataset) -> dict:
    return {
        "id": item.id,
        "filename": item.filename,
        "description": item.description,
        "status": item.status,
        "progress_step": item.progress_step,
        "progress_detail": item.progress_detail,
        "progress_current": item.progress_current,
        "progress_total": item.progress_total,
        "summary": item.summary,
        "profile": item.profile_json,
        "error_message": item.error_message,
        "created_at": item.created_at,
        "updated_at": item.updated_at,
    }


def session_title_from_message(question: str) -> str:
    cleaned = " ".join(question.strip().split())
    return (cleaned[:77] + "...") if len(cleaned) > 80 else (cleaned or "Nuevo chat")


def session_payload(session: ChatSession, db: Session) -> dict:
    dataset_links = (
        db.query(ChatSessionDataset)
        .filter(ChatSessionDataset.chat_session_id == session.id)
        .order_by(ChatSessionDataset.created_at.asc())
        .all()
    )
    datasets = [dataset_payload(link.dataset) for link in dataset_links if link.dataset]
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.chat_session_id == session.id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )
    last_message = messages[-1] if messages else None
    return {
        "id": session.id,
        "title": session.title,
        "datasets": datasets,
        "messages": [
            {
                "id": message.id,
                "role": message.role,
                "content": message.payload_json if message.role == "assistant" else message.content_text,
                "created_at": message.created_at,
            }
            for message in messages
        ],
        "last_message_preview": (
            (last_message.content_text or "")[:120]
            if last_message and last_message.role == "user"
            else str((last_message.payload_json or {}).get("response_text") or "")[:120]
            if last_message
            else ""
        ),
        "created_at": session.created_at,
        "updated_at": session.updated_at,
    }


def session_dataset_ids(db: Session, chat_session_id: str) -> list[str]:
    return [
        row.dataset_id
        for row in db.query(ChatSessionDataset.dataset_id)
        .filter(ChatSessionDataset.chat_session_id == chat_session_id)
        .all()
    ]


@app.get("/chat/sessions")
def list_chat_sessions(db: Session = Depends(get_db)) -> list[dict]:
    sessions = db.query(ChatSession).order_by(ChatSession.updated_at.desc(), ChatSession.created_at.desc()).all()
    return [session_payload(session, db) for session in sessions]


@app.post("/chat/sessions")
def create_chat_session(request: ChatSessionCreateRequest, db: Session = Depends(get_db)) -> dict:
    session = ChatSession(title=(request.title or "Nuevo chat").strip() or "Nuevo chat")
    db.add(session)
    db.commit()
    db.refresh(session)
    return session_payload(session, db)


@app.get("/chat/sessions/{chat_session_id}")
def get_chat_session(chat_session_id: str, db: Session = Depends(get_db)) -> dict:
    session = db.get(ChatSession, chat_session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat no encontrado.")
    return session_payload(session, db)


def process_chat_job(job_id: str) -> None:
    db = SessionLocal()
    try:
        job = db.get(ChatJob, job_id)
        if not job:
            return
        session = db.get(ChatSession, job.chat_session_id) if job.chat_session_id else None
        scoped_dataset_ids = session_dataset_ids(db, session.id) if session else None
        job.status = "running"
        job.detail = "Analizando la pregunta."
        db.commit()
        response = answer_question(db, job.dataset_id, job.question, scoped_dataset_ids)
        job = db.get(ChatJob, job_id)
        if job:
            job.status = "complete"
            job.detail = "Respuesta lista."
            job.response_json = response
            if session:
                db.add(ChatMessage(chat_session_id=session.id, role="assistant", content_text=None, payload_json=response))
                session.updated_at = func.now()
            db.commit()
    except Exception as exc:
        db.rollback()
        job = db.get(ChatJob, job_id)
        if job:
            job.status = "error"
            job.detail = "No se pudo completar la respuesta."
            job.error_message = str(exc)
            db.commit()
    finally:
        db.close()


@app.post("/chat/jobs")
def create_chat_job(
    request: ChatJobRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> dict:
    chat_session = db.get(ChatSession, request.chat_session_id)
    if not chat_session:
        raise HTTPException(status_code=404, detail="Chat no encontrado.")
    dataset = db.get(Dataset, request.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset no encontrado.")
    scoped_dataset_ids = set(session_dataset_ids(db, chat_session.id))
    if dataset.id not in scoped_dataset_ids:
        raise HTTPException(status_code=400, detail="El archivo no pertenece a este chat.")
    ready_count = (
        db.query(Dataset)
        .join(ChatSessionDataset, ChatSessionDataset.dataset_id == Dataset.id)
        .filter(ChatSessionDataset.chat_session_id == chat_session.id, Dataset.status == "ready")
        .count()
    )
    if ready_count == 0:
        raise HTTPException(status_code=400, detail="Todavía no hay archivos listos para responder preguntas.")
    db.add(ChatMessage(chat_session_id=chat_session.id, role="user", content_text=request.question, payload_json=None))
    if chat_session.title == "Nuevo chat" and request.question.strip():
        chat_session.title = session_title_from_message(request.question)
    job = ChatJob(
        chat_session_id=chat_session.id,
        dataset_id=request.dataset_id,
        question=request.question,
        status="queued",
        detail="Pregunta recibida.",
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    background_tasks.add_task(process_chat_job, job.id)
    return {"job_id": job.id, "status": job.status, "detail": job.detail, "chat_session_id": chat_session.id}


@app.get("/chat/jobs/{job_id}")
def get_chat_job(job_id: str, db: Session = Depends(get_db)) -> dict:
    job = db.get(ChatJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Trabajo de chat no encontrado.")
    return {
        "id": job.id,
        "chat_session_id": job.chat_session_id,
        "dataset_id": job.dataset_id,
        "question": job.question,
        "status": job.status,
        "detail": job.detail,
        "response": job.response_json,
        "error_message": job.error_message,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }
