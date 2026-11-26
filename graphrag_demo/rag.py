"""Baseline RAG: index each document, retrieve the top-k most similar to the question.

Documents here are tiny, so one document == one chunk. In real RAG you would split
long documents into ~500 token chunks first; the failure mode below is the same.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .common import load_docs, tokenize


class RAG:
    def __init__(self, dense: bool = False):
        self.dense = dense
        self.docs = load_docs()
        self.ids = list(self.docs)
        self.vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, stop_words=None)
        self.matrix = self.vec.fit_transform(self.docs.values())
        if dense:
            from .embed import embed
            self.emb = embed(self.docs.values())

    def retrieve(self, question: str, k: int = 4) -> list[str]:
        if self.dense:
            from .embed import embed
            sims = self.emb @ embed([question])[0]
        else:
            sims = cosine_similarity(self.vec.transform([question]), self.matrix)[0]
        ranked = sorted(zip(self.ids, sims), key=lambda x: -x[1])[:k]
        return [doc_id for doc_id, _ in ranked]

    def context(self, question: str, k: int = 4) -> tuple[str, set[str]]:
        ids = self.retrieve(question, k)
        return "\n".join(f"[{i}] {self.docs[i]}" for i in ids), set(ids)
