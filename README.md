# RAG PDF Backend (FastAPI + Mistral)

A minimal Retrieval-Augmented Generation (RAG) pipeline that ingests PDF files, performs hybrid retrieval (semantic + keyword), and answers questions with citations.

## System design

```
Upload PDFs
   |
   v
PDF text extraction -> chunking -> embeddings -> local JSON index
   |
   v
Query -> intent + rewrite -> hybrid search -> rerank -> evidence check
   |
   v
Mistral answer w/ citations -> response
```

### Data ingestion flow
1. PDF upload via `/ingest`.
2. Text extraction with `pypdf`.
3. Chunking by sentences with overlap.
4. Embeddings computed via Mistral `embeddings` endpoint.
5. Store chunks + embeddings + term stats in a local JSON index (`data/index.json`).

### Chunking considerations
- **Sentence-aware**: sentences are kept together to preserve meaning.
- **Max size + overlap**: limits per chunk avoid too-long context; overlap preserves continuity across boundaries.
- **PDF artifacts**: whitespace is normalized to mitigate line-break noise.
- **Trade-offs**: smaller chunks improve recall; larger chunks improve context but can dilute relevance.

### Query processing
- **Intent detection**: greetings/small-talk skip retrieval.
- **Query rewrite**: LLM rewrites to a compact, keyword-rich search string.
- **Policy checks**: PII and legal/medical requests are refused.

### Retrieval
- **Semantic search**: cosine similarity over Mistral embeddings.
- **Keyword search**: BM25-style scoring over chunk term frequencies.
- **Hybrid combination**: normalized scores with `alpha` weighting.
- **Rerank**: boost based on query term coverage.

### Post-processing + generation
- **Citations required**: if max similarity < threshold, return `insufficient evidence`.
- **Answer shaping**: list/table templates selected by intent.
- **Hallucination filter**: post-hoc evidence check prunes unsupported sentences.

## How to run

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000` for the UI.

## API endpoints

### `POST /ingest`
Upload one or more PDF files.

- form-data: `files` (one or more PDFs)
- response:

```json
{ "documents_ingested": 1, "chunks_added": 24 }
```

### `POST /query`
Query the system.

- body:

```json
{ "question": "What are the key product requirements?" }
```

- response:

```json
{
  "answer": "...",
  "intent": "question",
  "rewritten_query": "...",
  "citations": [
    {
      "source": "doc.pdf",
      "pages": "3-4",
      "chunk_id": "...",
      "score": 0.82,
      "excerpt": "..."
    }
  ],
  "unsupported_sentences": []
}
```

## Configuration
Environment variables (optional):

- `MISTRAL_API_KEY` (defaults to provided key)
- `MISTRAL_EMBED_MODEL` (default `mistral-embed`)
- `MISTRAL_CHAT_MODEL` (default `mistral-small-latest`)
- `CHUNK_SIZE` (default `1000`)
- `CHUNK_OVERLAP` (default `200`)
- `TOP_K` (default `6`)
- `MIN_SIMILARITY` (default `0.22`)
- `HYBRID_ALPHA` (default `0.7`)

## Libraries and links

- FastAPI: https://fastapi.tiangolo.com/
- Mistral AI API: https://docs.mistral.ai/
- PyPDF: https://pypdf.readthedocs.io/
- HTTPX: https://www.python-httpx.org/

## Notes
- No external search or RAG libraries are used.
- No third-party vector database is used (local JSON index only).
- For larger corpora, move storage to SQLite or a dedicated index file per document.