"""MongoDB Atlas Local: run records, long-term memory, and the shared corpus index.

Collection naming is the data-isolation rule in code form: every demo owns
`<demo>_runs`, `<demo>_checkpoints` (+ `_checkpoint_writes`) and, where it has
long-term memory, `<demo>_memory`. No demo reads another's collections. The one
shared collection is `corpus_chunks`, which is read-only for every demo.
"""

import time
from typing import Any, Iterable

import streamlit as st
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError
from pymongo.operations import SearchIndexModel

from .config import get_settings
from .embeddings import EMBEDDING_DIMS
from .types import AgentRun

# The corpus index every demo may read and no demo may write (core/corpus.py builds it).
CORPUS_COLLECTION = "corpus_chunks"

# A paused or unreachable database should say so in a second, not hang the page for
# thirty. Every call that touches Mongo is wrapped by the caller in a way that turns
# this into a UI message (ground rule 10).
_SERVER_SELECTION_TIMEOUT_MS = 3000


class MongoUnavailable(RuntimeError):
    """Raised instead of leaking a driver traceback into the page."""


@st.cache_resource(show_spinner=False)
def get_client() -> MongoClient:
    settings = get_settings()
    return MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=_SERVER_SELECTION_TIMEOUT_MS)


def get_collection(name: str) -> Collection:
    settings = get_settings()
    return get_client()[settings.mongodb_db][name]


def ping() -> tuple[bool, str]:
    """(reachable, message). Used by the pages to degrade politely rather than crash."""
    try:
        get_client().admin.command("ping")
        return True, "MongoDB is reachable."
    except PyMongoError as exc:
        return False, f"MongoDB is not reachable: {type(exc).__name__}."


# --- collection names per demo -------------------------------------------------


def runs_collection(demo: str) -> Collection:
    return get_collection(f"{demo}_runs")


def memory_collection(demo: str) -> Collection:
    return get_collection(f"{demo}_memory")


def checkpoint_collection_names(demo: str) -> tuple[str, str]:
    """(checkpoints, checkpoint_writes) -- the two collections MongoDBSaver writes."""
    return f"{demo}_checkpoints", f"{demo}_checkpoint_writes"


# --- search indexes ------------------------------------------------------------


def ensure_indexes(
    collection: Collection,
    *,
    vector_dims: int = EMBEDDING_DIMS,
    filter_fields: Iterable[str] = (),
    timeout_s: float = 60.0,
) -> None:
    """Create `vector_index` if missing, and block until it is queryable.

    The waiting is the point: Atlas search indexes build in the background, and a
    $vectorSearch sent straight after create_search_index returns an empty result
    set rather than an error -- which reads exactly like "retrieval found nothing".
    This bit the sibling RAG lab twice, so it is carried over deliberately.
    """
    existing = {ix["name"] for ix in collection.list_search_indexes()}

    if "vector_index" not in existing:
        definition = {
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": vector_dims,
                    "similarity": "cosine",
                },
                *({"type": "filter", "path": field} for field in filter_fields),
            ]
        }
        collection.create_search_index(
            SearchIndexModel(definition, name="vector_index", type="vectorSearch")
        )

    _wait_until_queryable(collection, "vector_index", timeout_s)


def _wait_until_queryable(collection: Collection, name: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for ix in collection.list_search_indexes(name):
            if ix.get("queryable"):
                return
        time.sleep(1)
    raise TimeoutError(f"Search index '{name}' did not become queryable within {timeout_s}s.")


def vector_search(
    collection: Collection,
    query_vector: list[float],
    *,
    top_k: int = 5,
    match_filter: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """$vectorSearch with the embedding stripped from the results and the score attached."""
    stage: dict[str, Any] = {
        "index": "vector_index",
        "path": "embedding",
        "queryVector": query_vector,
        "numCandidates": max(top_k * 10, 100),
        "limit": top_k,
    }
    if match_filter:
        stage["filter"] = match_filter

    pipeline = [
        {"$vectorSearch": stage},
        {"$project": {"embedding": 0, "score": {"$meta": "vectorSearchScore"}}},
    ]
    return list(collection.aggregate(pipeline))


# --- run records ---------------------------------------------------------------


def save_run(run: AgentRun) -> None:
    """Stores a finished run in its own demo's collection. Never raises into the page:
    a run that completed is worth showing even if the record could not be written."""
    if not run.demo:
        raise ValueError("AgentRun.demo must be set before the run can be stored.")
    try:
        runs_collection(run.demo).insert_one({"_id": run.run_id, **run.model_dump()})
    except PyMongoError:
        pass


def recent_runs(demo: str, limit: int = 10) -> list[dict[str, Any]]:
    """Most recent runs first. run_id sorts by creation, so no index is needed."""
    try:
        return list(runs_collection(demo).find().sort("_id", -1).limit(limit))
    except PyMongoError:
        return []


def clear_demo_data(demo: str) -> dict[str, int]:
    """Deletes this demo's runs, checkpoints and memory. What `Clear my data` calls.

    Returns a per-collection count of what went, so the page can say what it did
    rather than flashing a bare "done".
    """
    checkpoints, writes = checkpoint_collection_names(demo)
    names = [f"{demo}_runs", checkpoints, writes, f"{demo}_memory"]
    deleted: dict[str, int] = {}
    for name in names:
        try:
            collection = get_collection(name)
            count = collection.count_documents({})
            if count:
                collection.delete_many({})
            deleted[name] = count
        except PyMongoError:
            deleted[name] = 0
    return deleted
