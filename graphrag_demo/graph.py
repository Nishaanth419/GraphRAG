"""GraphRAG: knowledge graph + local (neighbourhood) search + global (community) search."""
import re
from collections import deque

import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .common import load_triples, tokenize


STOP = {"the", "a", "an", "of", "if", "is", "are", "who", "what", "whose", "which", "when", "does", "do", "to", "for", "on",
        "in", "that", "this", "our", "we", "by", "and", "or", "it", "goes", "down", "owns", "report", "reports", "head", "heads"}


def question_spans(question: str) -> list[str]:
    """Every 1-4 word phrase of the question that doesn't start or end with a stopword."""
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'.-]*", question)
    return [" ".join(words[i:j]) for i in range(len(words)) for j in range(i + 1, min(len(words), i + 4) + 1)
            if not all(w.lower() in STOP for w in words[i:j]) and words[i].lower() not in STOP and words[j - 1].lower() not in STOP]


class GraphRAG:
    def __init__(self, triples=None, summarizer=None, dense: bool = False, link_threshold: float = 0.75, scorer=None, min_score: int = 5):
        self.dense, self.link_threshold = dense, link_threshold
        self.scorer, self.min_score = scorer, min_score
        self.g = nx.MultiDiGraph()
        for s, r, o, doc in triples or load_triples():
            self.g.add_edge(s, o, rel=r, doc=doc)
        self._build_communities(summarizer)
        if dense:
            from .embed import embed
            self.names = list(self.g.nodes)
            self.nemb = embed(self.names)
            self.cemb = embed(c["summary"] for c in self.communities)

    # ---------- indexing: communities + summaries ----------
    def _build_communities(self, summarizer):
        undirected = nx.Graph(self.g.to_undirected())
        groups = nx.community.louvain_communities(undirected, seed=7)
        self.communities = []
        for members in sorted(groups, key=lambda m: sorted(m)):
            edges = [(u, d["rel"], v, d["doc"]) for u, v, d in self.g.edges(data=True) if u in members and v in members]
            facts = "; ".join(f"{u} {r} {v}" for u, r, v, _ in edges)
            summary = summarizer(sorted(members), facts) if summarizer else f"Entities: {', '.join(sorted(members))}. Facts: {facts}."
            self.communities.append({"members": members, "edges": edges, "summary": summary})
        self.cvec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
        self.cmatrix = self.cvec.fit_transform(c["summary"] for c in self.communities)

    # ---------- local search: entities in the question -> expand the neighbourhood ----------
    def link_entities(self, question: str) -> list[str]:
        if self.dense:
            return self.link_entities_dense(question)
        q = question.lower()
        return [n for n in self.g.nodes if n.lower() in q]

    def link_entities_dense(self, question: str) -> list[str]:
        """Embed every 1-4 word phrase of the question and match it to the nearest entity name (cosine >= threshold)."""
        from .embed import embed
        spans = question_spans(question)
        sims = embed(spans) @ self.nemb.T
        best = {}
        for k in range(len(spans)):
            m = int(sims[k].argmax())
            if sims[k][m] >= self.link_threshold:
                best[self.names[m]] = max(best.get(self.names[m], 0), float(sims[k][m]))
        return sorted(best, key=lambda n: -best[n])

    def local_search(self, question: str, hops: int = 2):
        seeds = self.link_entities(question)
        dist = {s: 0 for s in seeds}
        queue = deque(seeds)
        und = self.g.to_undirected(as_view=True)
        while queue:
            n = queue.popleft()
            if dist[n] == hops:
                continue
            for nb in und.neighbors(n):
                if nb not in dist:
                    dist[nb] = dist[n] + 1
                    queue.append(nb)
        edges = {(u, d["rel"], v, d["doc"]) for u, v, d in self.g.edges(data=True)
                 if u in dist and v in dist and min(dist[u], dist[v]) < hops}
        return seeds, sorted(edges)

    def local_context(self, question: str, hops: int = 2):
        seeds, edges = self.local_search(question, hops)
        text = f"Seed entities: {seeds}\n" + "\n".join(f"{u} --{r}--> {v}   [{d}]" for u, r, v, d in edges)
        return text, {d for *_, d in edges}

    # ---------- global search: rank community summaries ----------
    def global_context(self, question: str, top: int = 2):
        if self.scorer:  # map step: an LLM rates every community; keep all that are useful enough (at least the best one)
            scores = [self.scorer(question, c["summary"]) for c in self.communities]
            best = [i for i in sorted(range(len(scores)), key=lambda i: -scores[i]) if scores[i] >= self.min_score] or [max(range(len(scores)), key=lambda i: scores[i])]
            text = "\n".join(f"Community {i} (score {scores[i]}): {self.communities[i]['summary']}" for i in best)
            return text, {d for i in best for *_, d in self.communities[i]["edges"]}
        if self.dense:
            from .embed import embed
            sims = self.cemb @ embed([question])[0]
        else:
            sims = cosine_similarity(self.cvec.transform([question]), self.cmatrix)[0]
        best = sorted(range(len(self.communities)), key=lambda i: -sims[i])[:top]
        text = "\n".join(f"Community {i}: {self.communities[i]['summary']}" for i in best)
        docs = {d for i in best for *_, d in self.communities[i]["edges"]}
        return text, docs
