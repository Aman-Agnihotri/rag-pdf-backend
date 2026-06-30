from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    base_dir: Path
    data_dir: Path
    uploads_dir: Path
    index_path: Path
    mistral_api_key: str
    mistral_embed_model: str
    mistral_chat_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    min_similarity: float
    hybrid_alpha: float


def get_settings() -> Settings:
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data"
    uploads_dir = data_dir / "uploads"
    index_path = data_dir / "index.json"

    env_key = os.getenv("MISTRAL_API_KEY")
    api_key = env_key.strip() if env_key else ""
    if not api_key:
        api_key = "CF2DvjIoshzasO0mtBkPj44fo2nXDwPk"

    return Settings(
        base_dir=base_dir,
        data_dir=data_dir,
        uploads_dir=uploads_dir,
        index_path=index_path,
        mistral_api_key=api_key,
        mistral_embed_model=os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed"),
        mistral_chat_model=os.getenv("MISTRAL_CHAT_MODEL", "mistral-small-latest"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        top_k=int(os.getenv("TOP_K", "6")),
        min_similarity=float(os.getenv("MIN_SIMILARITY", "0.22")),
        hybrid_alpha=float(os.getenv("HYBRID_ALPHA", "0.7")),
    )
