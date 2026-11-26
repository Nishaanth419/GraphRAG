"""Local sentence embeddings (all-MiniLM-L6-v2, 384 dims, ~90 MB, runs on CPU). Vectors are L2-normalised, so dot product = cosine.
Entity and community vectors are persisted in Neo4j (see neo4j_store.py); nothing is cached on disk here."""
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_model = None


def embed(texts):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
