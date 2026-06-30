from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class DocumentMeta:
    doc_id: str
    name: str
    path: str
    pages: int
    uploaded_at: str


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    text: str
    page_start: int
    page_end: int
    embedding: List[float]
    token_counts: Dict[str, int]
    token_len: int


@dataclass
class IndexState:
    documents: Dict[str, DocumentMeta] = field(default_factory=dict)
    chunks: Dict[str, ChunkRecord] = field(default_factory=dict)
    doc_freq: Dict[str, int] = field(default_factory=dict)
    chunk_count: int = 0


class IndexStore:
    def __init__(self, index_path: Path) -> None:
        self.index_path = index_path
        self.state = IndexState()

    def load(self) -> None:
        if not self.index_path.exists():
            return
        raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        documents = {}
        for doc_id, meta in raw.get("documents", {}).items():
            documents[doc_id] = DocumentMeta(**meta)
        chunks = {}
        for chunk_id, chunk in raw.get("chunks", {}).items():
            chunks[chunk_id] = ChunkRecord(**chunk)
        self.state = IndexState(
            documents=documents,
            chunks=chunks,
            doc_freq=raw.get("doc_freq", {}),
            chunk_count=raw.get("chunk_count", 0),
        )

    def save(self) -> None:
        payload = {
            "documents": {doc_id: vars(meta) for doc_id, meta in self.state.documents.items()},
            "chunks": {chunk_id: vars(chunk) for chunk_id, chunk in self.state.chunks.items()},
            "doc_freq": self.state.doc_freq,
            "chunk_count": self.state.chunk_count,
        }
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    def add_document(self, meta: DocumentMeta) -> None:
        self.state.documents[meta.doc_id] = meta

    def add_chunk(self, chunk: ChunkRecord) -> None:
        self.state.chunks[chunk.chunk_id] = chunk
        self.state.chunk_count += 1

    def update_doc_freq(self, terms: Iterable[str]) -> None:
        unique_terms = set(terms)
        for term in unique_terms:
            self.state.doc_freq[term] = self.state.doc_freq.get(term, 0) + 1

    def get_chunks(self) -> List[ChunkRecord]:
        return list(self.state.chunks.values())

    def get_document(self, doc_id: str) -> Optional[DocumentMeta]:
        return self.state.documents.get(doc_id)