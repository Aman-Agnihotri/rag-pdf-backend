from __future__ import annotations

import datetime as dt
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.core.config import get_settings
from app.services.generation import generate_answer
from app.services.mistral_client import MistralClient
from app.services.pdf_ingest import chunk_text, extract_text_from_pdf
from app.services.policies import check_refusal_policy, detect_intent, evidence_check, rewrite_query
from app.services.search import ScoredChunk, hybrid_search, rerank, tokenize, cosine_similarity
from app.storage.store import ChunkRecord, DocumentMeta, IndexStore

settings = get_settings()
store = IndexStore(settings.index_path)
store.load()
client = MistralClient(settings)

app = FastAPI(title="RAG PDF Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = settings.base_dir / "app" / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


class IngestResponse(BaseModel):
    documents_ingested: int
    chunks_added: int


class QueryRequest(BaseModel):
    question: str


class Citation(BaseModel):
    source: str
    pages: str
    chunk_id: str
    score: float
    excerpt: str


class QueryResponse(BaseModel):
    answer: str
    intent: str
    rewritten_query: Optional[str]
    citations: List[Citation]
    unsupported_sentences: List[str]


@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    index_path = static_dir / "index.html"
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.post("/ingest", response_model=IngestResponse)
async def ingest(files: List[UploadFile] = File(...)) -> IngestResponse:
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    chunks_added = 0
    for upload in files:
        doc_id = str(uuid.uuid4())
        safe_name = upload.filename or f"upload-{doc_id}.pdf"
        target = settings.uploads_dir / f"{doc_id}-{safe_name}"
        with target.open("wb") as handle:
            handle.write(await upload.read())

        pages = extract_text_from_pdf(str(target))
        if not pages:
            continue

        doc_meta = DocumentMeta(
            doc_id=doc_id,
            name=safe_name,
            path=str(target),
            pages=len(pages),
            uploaded_at=dt.datetime.utcnow().isoformat(),
        )
        store.add_document(doc_meta)

        raw_chunks = chunk_text(pages, settings.chunk_size, settings.chunk_overlap)
        texts = [chunk["text"] for chunk in raw_chunks]
        try:
            embeddings = client.embed(texts) if texts else []
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Embedding request failed: {exc}") from exc

        for chunk, embedding in zip(raw_chunks, embeddings):
            chunk_id = str(uuid.uuid4())
            tokens = tokenize(chunk["text"])
            token_counts = {}
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
            store.update_doc_freq(token_counts.keys())
            record = ChunkRecord(
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=chunk["text"],
                page_start=chunk["page_start"],
                page_end=chunk["page_end"],
                embedding=embedding,
                token_counts=token_counts,
                token_len=len(tokens),
            )
            store.add_chunk(record)
            chunks_added += 1

    store.save()
    return IngestResponse(documents_ingested=len(files), chunks_added=chunks_added)


@app.post("/query", response_model=QueryResponse)
async def query(payload: QueryRequest) -> QueryResponse:
    question = payload.question.strip()
    refusal, refusal_msg = check_refusal_policy(question)
    if refusal:
        return QueryResponse(
            answer=refusal_msg,
            intent="refused",
            rewritten_query=None,
            citations=[],
            unsupported_sentences=[],
        )

    intent_result = detect_intent(question)
    if not intent_result.should_search:
        return QueryResponse(
            answer="Hello! Ask a question about your uploaded PDFs.",
            intent=intent_result.intent,
            rewritten_query=None,
            citations=[],
            unsupported_sentences=[],
        )

    rewritten = rewrite_query(client, question)
    try:
        query_embedding = client.embed([rewritten])[0]
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Embedding request failed: {exc}") from exc

    scored = hybrid_search(
        store=store,
        query_embedding=query_embedding,
        query_text=rewritten,
        alpha=settings.hybrid_alpha,
        top_k=settings.top_k,
    )

    scored = rerank(scored, rewritten)
    if not scored:
        return QueryResponse(
            answer="insufficient evidence",
            intent=intent_result.intent,
            rewritten_query=rewritten,
            citations=[],
            unsupported_sentences=[],
        )

    max_raw = max(cosine_similarity(query_embedding, item.chunk.embedding) for item in scored)
    if max_raw < settings.min_similarity:
        return QueryResponse(
            answer="insufficient evidence",
            intent=intent_result.intent,
            rewritten_query=rewritten,
            citations=[],
            unsupported_sentences=[],
        )

    top_chunks = [item.chunk for item in scored]
    answer_payload = generate_answer(client, question, intent_result.intent, top_chunks, store.state.documents)
    filtered_answer, unsupported = evidence_check(answer_payload.answer, top_chunks)

    citations: List[Citation] = []
    for item in scored:
        doc = store.get_document(item.chunk.doc_id)
        source = doc.name if doc else item.chunk.doc_id
        pages = f"{item.chunk.page_start}-{item.chunk.page_end}"
        citations.append(
            Citation(
                source=source,
                pages=pages,
                chunk_id=item.chunk.chunk_id,
                score=round(item.hybrid_score, 4),
                excerpt=item.chunk.text[:280],
            )
        )

    return QueryResponse(
        answer=filtered_answer,
        intent=intent_result.intent,
        rewritten_query=rewritten,
        citations=citations,
        unsupported_sentences=unsupported,
    )
