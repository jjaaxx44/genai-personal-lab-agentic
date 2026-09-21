"""The one local embedding model, loaded once.

Used by the corpus search tool, by long-term memory recall and by evaluation. CPU
only, and small on purpose: on an 8 GB box it shares the machine with MongoDB and
(when it is the fallback in use) a 7B local chat model.
"""

import streamlit as st
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from .config import get_settings

# bge-small-en-v1.5. Every vector index in this app is built for this width.
EMBEDDING_DIMS = 384


@st.cache_resource(show_spinner="Loading embedding model...")
def _get_model() -> SentenceTransformer:
    settings = get_settings()
    return SentenceTransformer(settings.embedding_model)


def embed(texts: list[str]) -> list[list[float]]:
    model = _get_model()
    return model.encode(texts, normalize_embeddings=True).tolist()


class LocalEmbeddings(Embeddings):
    """Wraps the shared sentence-transformers model so it's loaded once."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return embed([text])[0]
