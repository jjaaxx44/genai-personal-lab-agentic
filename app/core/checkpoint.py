"""Durable graph state, one checkpoint namespace per demo.

This is what makes a gate survivable: the run pauses at `interrupt()`, Streamlit
reruns (or the reader refreshes the browser, or the container restarts), and
`resume()` picks the graph up from the checkpoint instead of starting again.

`langgraph-checkpoint-mongodb==0.5.0` writes two collections per saver, so a demo
gets `<demo>_checkpoints` and `<demo>_checkpoint_writes` and touches nothing else.
"""

from typing import Any

import streamlit as st
from langgraph.checkpoint.mongodb import MongoDBSaver

from .config import get_settings
from .mongo import checkpoint_collection_names, get_client
from .types import thread_id_for  # noqa: F401  (re-exported: demos build thread ids here)


@st.cache_resource(show_spinner=False)
def get_checkpointer(demo: str) -> MongoDBSaver:
    """The saver for one demo. Cached, because its constructor creates its indexes."""
    settings = get_settings()
    checkpoints, writes = checkpoint_collection_names(demo)
    return MongoDBSaver(
        get_client(),
        db_name=settings.mongodb_db,
        checkpoint_collection_name=checkpoints,
        writes_collection_name=writes,
    )


def thread_config(thread_id: str, **extra: Any) -> dict[str, Any]:
    """The `config` every checkpointed invoke/stream needs. One thread per run."""
    return {"configurable": {"thread_id": thread_id, **extra}}


def has_checkpoint(demo: str, thread_id: str) -> bool:
    """Whether this thread has state to resume from -- checked before offering resume,
    so a stale thread id from session state can't produce a confusing empty resume."""
    return get_checkpointer(demo).get(thread_config(thread_id)) is not None
