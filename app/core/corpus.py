"""The shared corpus: the only thing in this app every demo may read.

Input here is a task, not a document (ground rule 1), so there is no upload and
no `doc_id`. Instead `samples/corpus/` ships with the repo, gets chunked and
embedded once into `corpus_chunks`, and `search_corpus` queries it. A reader who
drops another `.md` or `.txt` in there gets it indexed on the next run: files are
tracked by content hash, so only what changed is re-embedded.

Read-only for every demo. Nothing writes to `corpus_chunks` except this module.
"""

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from .config import REPO_ROOT, get_settings
from .embeddings import embed
from .mongo import CORPUS_COLLECTION, ensure_indexes, get_collection, vector_search

CORPUS_DIR = REPO_ROOT / "samples" / "corpus"
CORPUS_SUFFIXES = (".md", ".txt")


@dataclass
class CorpusStats:
    files: int
    chunks: int
    indexed: int  # files (re)embedded on this pass; 0 means the index was already current
    latency_ms: float


def corpus_files() -> list[Path]:
    return sorted(p for p in CORPUS_DIR.rglob("*") if p.suffix in CORPUS_SUFFIXES and p.is_file())


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _chunk(text: str) -> list[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.corpus_chunk_size, chunk_overlap=settings.corpus_chunk_overlap
    )
    return splitter.split_text(text)


def build_index(force: bool = False) -> CorpusStats:
    """Indexes anything new or changed, and waits for the vector index to be queryable.

    Deliberately incremental: re-embedding a corpus that has not changed costs
    nothing but time on a CPU-only box, and this runs on the first search of every
    session.
    """
    start = time.monotonic()
    collection = get_collection(CORPUS_COLLECTION)
    files = corpus_files()

    stored: dict[str, str] = {}
    if not force:
        for record in collection.aggregate(
            [{"$group": {"_id": "$source", "file_hash": {"$first": "$file_hash"}}}]
        ):
            stored[record["_id"]] = record["file_hash"]
    else:
        collection.delete_many({})

    current = {p.name: p for p in files}
    # Files deleted from the corpus directory leave the index with them.
    for gone in set(stored) - set(current):
        collection.delete_many({"source": gone})

    indexed = 0
    for name, path in current.items():
        text = path.read_text(encoding="utf-8", errors="replace")
        file_hash = _hash(text)
        if stored.get(name) == file_hash:
            continue

        collection.delete_many({"source": name})
        chunks = _chunk(text)
        if not chunks:
            continue
        vectors = embed(chunks)
        collection.insert_many(
            [
                {
                    "source": name,
                    "file_hash": file_hash,
                    "chunk_index": i,
                    "text": chunk,
                    "embedding": vector,
                }
                for i, (chunk, vector) in enumerate(zip(chunks, vectors))
            ]
        )
        indexed += 1

    total_chunks = collection.count_documents({})
    if total_chunks:
        ensure_indexes(collection, filter_fields=("source",))

    return CorpusStats(
        files=len(files),
        chunks=total_chunks,
        indexed=indexed,
        latency_ms=(time.monotonic() - start) * 1000,
    )


@st.cache_resource(show_spinner="Indexing the corpus...")
def _ensure_indexed() -> CorpusStats:
    """Runs build_index once per session. Cached on the stats it returns, so the
    first search pays for indexing and later ones do not."""
    return build_index()


def reset_index_cache() -> None:
    """Forgets the once-per-session indexing result, so the next search re-checks the
    directory. Scoped to this cache rather than st.cache_resource.clear(), which would
    also drop the embedding model and the Mongo client."""
    _ensure_indexed.clear()


def search_corpus(query: str, *, top_k: int | None = None) -> list[dict]:
    """The read path behind the `search_corpus` tool. Returns text, source and score."""
    settings = get_settings()
    top_k = top_k or settings.corpus_top_k

    stats = _ensure_indexed()
    if not stats.chunks:
        return []

    # A database error is deliberately allowed to propagate: the tool wrapper turns it
    # into an ERROR[...] observation, which is true, where an empty list would read as
    # "the corpus has nothing on this".
    collection = get_collection(CORPUS_COLLECTION)
    hits = vector_search(collection, embed([query])[0], top_k=top_k)
    return [{"text": h["text"], "source": h["source"], "score": h["score"]} for h in hits]
