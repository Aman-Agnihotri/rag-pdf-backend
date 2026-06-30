from __future__ import annotations

from dataclasses import dataclass
from typing import List

from app.services.mistral_client import MistralClient
from app.storage.store import ChunkRecord, DocumentMeta


@dataclass
class AnswerPayload:
    answer: str
    prompt: str


def build_context(chunks: List[ChunkRecord], documents: dict) -> str:
    lines = []
    for idx, chunk in enumerate(chunks, start=1):
        doc: DocumentMeta | None = documents.get(chunk.doc_id)
        source = doc.name if doc else chunk.doc_id
        lines.append(
            f"[S{idx}] {source} p.{chunk.page_start}-{chunk.page_end}: {chunk.text}"
        )
    return "\n".join(lines)


def select_template(intent: str) -> str:
    if intent == "list":
        return (
            "Answer using a short bullet list. Each bullet must end with citations like [S1]. "
            "Only use the provided sources."
        )
    if intent == "table":
        return (
            "Answer using a markdown table. Every row must include citations in the last column. "
            "Only use the provided sources."
        )
    return (
        "Answer in 3-6 sentences. Each sentence must end with citations like [S1]. "
        "Only use the provided sources."
    )


def generate_answer(
    client: MistralClient,
    question: str,
    intent: str,
    chunks: List[ChunkRecord],
    documents: dict,
) -> AnswerPayload:
    context = build_context(chunks, documents)
    instruction = select_template(intent)
    prompt = (
        "You are a careful assistant. If the sources do not contain the answer, reply exactly "
        "with: insufficient evidence.\n\n"
        f"Sources:\n{context}\n\n"
        f"Question: {question}\n{instruction}"
    )
    messages = [
        {"role": "system", "content": "You answer only from provided sources."},
        {"role": "user", "content": prompt},
    ]
    answer = client.chat(messages, temperature=0.2)
    return AnswerPayload(answer=answer, prompt=prompt)