from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

from app.services.mistral_client import MistralClient
from app.services.search import STOPWORDS, tokenize
from app.storage.store import ChunkRecord


GREETINGS = {"hello", "hi", "hey", "good morning", "good evening", "good afternoon"}


@dataclass
class IntentResult:
    intent: str
    should_search: bool


def detect_intent(query: str) -> IntentResult:
    cleaned = query.strip().lower()
    if not cleaned:
        return IntentResult(intent="empty", should_search=False)
    if cleaned in GREETINGS or cleaned.split()[0] in {"hi", "hello", "hey"}:
        return IntentResult(intent="smalltalk", should_search=False)
    if len(cleaned) < 4:
        return IntentResult(intent="smalltalk", should_search=False)
    if cleaned.startswith("list") or " list " in cleaned:
        return IntentResult(intent="list", should_search=True)
    if cleaned.startswith("table") or " table " in cleaned:
        return IntentResult(intent="table", should_search=True)
    if cleaned.startswith("compare"):
        return IntentResult(intent="table", should_search=True)
    return IntentResult(intent="question", should_search=True)


def check_refusal_policy(query: str) -> Tuple[bool, str]:
    lowered = query.lower()
    pii_terms = ["ssn", "social security", "credit card", "password", "bank account", "passport"]
    if any(term in lowered for term in pii_terms):
        return True, "I can't help with requests involving personal or sensitive data."
    medical_terms = ["diagnose", "treatment", "prescribe", "symptoms", "medical advice"]
    legal_terms = ["lawsuit", "legal advice", "contract", "liability", "attorney"]
    if any(term in lowered for term in medical_terms):
        return True, "I can't provide medical advice. Please consult a qualified professional."
    if any(term in lowered for term in legal_terms):
        return True, "I can't provide legal advice. Please consult a qualified professional."
    return False, ""


def rewrite_query(client: MistralClient, query: str) -> str:
    prompt = (
        "Rewrite the user question into a short, keyword-rich search query. "
        "Keep entities and technical terms. Return only the rewritten query."
    )
    messages = [
        {"role": "system", "content": "You are a query rewriting assistant."},
        {"role": "user", "content": f"Question: {query}"},
        {"role": "user", "content": prompt},
    ]
    try:
        rewritten = client.chat(messages, temperature=0.1)
        return rewritten or query
    except Exception:
        return query


def split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def evidence_check(answer: str, chunks: List[ChunkRecord]) -> Tuple[str, List[str]]:
    unsupported: List[str] = []
    sentences = split_sentences(answer)
    for sentence in sentences:
        sentence_tokens = [t for t in tokenize(sentence) if t not in STOPWORDS]
        if not sentence_tokens:
            continue
        supported = False
        for chunk in chunks:
            chunk_tokens = set(tokenize(chunk.text))
            overlap = len(set(sentence_tokens) & chunk_tokens)
            if overlap >= 2:
                supported = True
                break
        if not supported:
            unsupported.append(sentence)

    if not unsupported:
        return answer, []

    filtered = " ".join([s for s in sentences if s not in unsupported]).strip()
    if not filtered:
        return "insufficient evidence", unsupported
    return filtered, unsupported