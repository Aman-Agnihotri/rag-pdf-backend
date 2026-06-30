from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from app.storage.store import ChunkRecord, IndexStore


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in",
    "is", "it", "its", "of", "on", "that", "the", "to", "was", "were", "will", "with",
    "i", "you", "your", "we", "they", "this", "those", "these", "or", "not", "but",
}


@dataclass
class ScoredChunk:
    chunk: ChunkRecord
    semantic_score: float
    keyword_score: float
    hybrid_score: float


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [token for token in tokens if token not in STOPWORDS]


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def bm25_score(query_tokens: List[str], chunk: ChunkRecord, store: IndexStore) -> float:
    if not query_tokens:
        return 0.0
    avg_len = _avg_chunk_len(store)
    k1 = 1.5
    b = 0.75
    score = 0.0
    for term in query_tokens:
        df = store.state.doc_freq.get(term, 0)
        if df == 0:
            continue
        idf = math.log(1 + (store.state.chunk_count - df + 0.5) / (df + 0.5))
        tf = chunk.token_counts.get(term, 0)
        if tf == 0:
            continue
        denom = tf + k1 * (1 - b + b * (chunk.token_len / avg_len))
        score += idf * (tf * (k1 + 1) / denom)
    return score


def _avg_chunk_len(store: IndexStore) -> float:
    if store.state.chunk_count == 0:
        return 1.0
    total = sum(chunk.token_len for chunk in store.state.chunks.values())
    return max(total / store.state.chunk_count, 1.0)


def normalize(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    min_score = min(scores.values())
    max_score = max(scores.values())
    if math.isclose(min_score, max_score):
        return {key: 1.0 for key in scores}
    return {key: (value - min_score) / (max_score - min_score) for key, value in scores.items()}


def hybrid_search(
    store: IndexStore,
    query_embedding: List[float],
    query_text: str,
    alpha: float,
    top_k: int,
) -> List[ScoredChunk]:
    query_tokens = tokenize(query_text)
    semantic_scores: Dict[str, float] = {}
    keyword_scores: Dict[str, float] = {}

    for chunk in store.get_chunks():
        semantic_scores[chunk.chunk_id] = cosine_similarity(query_embedding, chunk.embedding)
        keyword_scores[chunk.chunk_id] = bm25_score(query_tokens, chunk, store)

    semantic_scores = normalize(semantic_scores)
    keyword_scores = normalize(keyword_scores)

    scored: List[ScoredChunk] = []
    for chunk in store.get_chunks():
        semantic = semantic_scores.get(chunk.chunk_id, 0.0)
        keyword = keyword_scores.get(chunk.chunk_id, 0.0)
        hybrid = alpha * semantic + (1 - alpha) * keyword
        scored.append(ScoredChunk(chunk=chunk, semantic_score=semantic, keyword_score=keyword, hybrid_score=hybrid))

    scored.sort(key=lambda item: item.hybrid_score, reverse=True)
    return scored[:top_k]


def rerank(scored: List[ScoredChunk], query_text: str) -> List[ScoredChunk]:
    query_tokens = set(tokenize(query_text))
    reranked: List[Tuple[float, ScoredChunk]] = []
    for item in scored:
        chunk_tokens = set(tokenize(item.chunk.text))
        coverage = len(query_tokens & chunk_tokens) / max(len(query_tokens), 1)
        boost = 0.1 * coverage
        reranked.append((item.hybrid_score + boost, item))
    reranked.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in reranked]