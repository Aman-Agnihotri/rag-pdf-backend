from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List

from pypdf import PdfReader


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str


def extract_text_from_pdf(path: str) -> List[PageText]:
    reader = PdfReader(path)
    pages: List[PageText] = []
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        cleaned = re.sub(r"\s+", " ", text).strip()
        pages.append(PageText(page_number=idx, text=cleaned))
    return pages


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def chunk_text(pages: Iterable[PageText], chunk_size: int, chunk_overlap: int) -> List[dict]:
    chunks: List[dict] = []
    buffer = ""
    buffer_start_page = None
    buffer_end_page = None

    for page in pages:
        sentences = split_sentences(page.text)
        for sentence in sentences:
            if buffer_start_page is None:
                buffer_start_page = page.page_number
            buffer_end_page = page.page_number
            if len(buffer) + len(sentence) + 1 <= chunk_size:
                buffer = f"{buffer} {sentence}".strip()
                continue

            if buffer:
                chunks.append(
                    {
                        "text": buffer,
                        "page_start": buffer_start_page,
                        "page_end": buffer_end_page,
                    }
                )
                if chunk_overlap > 0:
                    overlap_text = buffer[-chunk_overlap:]
                    buffer = overlap_text.strip()
                    buffer_start_page = buffer_end_page
                else:
                    buffer = ""
                    buffer_start_page = None

            if sentence:
                buffer = f"{buffer} {sentence}".strip()
                if buffer_start_page is None:
                    buffer_start_page = page.page_number
                buffer_end_page = page.page_number

    if buffer:
        chunks.append(
            {
                "text": buffer,
                "page_start": buffer_start_page or 1,
                "page_end": buffer_end_page or buffer_start_page or 1,
            }
        )

    return chunks