"""Generate report.html: how the graph is built + RAG vs GraphRAG comparison.
Uses triples_llm.json if present (from build_index), else triples.json.   python -m graphrag_demo.make_report"""
import json
from collections import deque

import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

from .common import ROOT, load_docs
from .compare import QUESTIONS, run
from .graph import GraphRAG
from .rag import RAG


docs, rag, rag_d = load_docs(), RAG(), RAG(dense=True)
try:  # Neo4j is optional: without it the report simply omits the Neo4j column
    from .neo4j_store import Neo4jGraphRAG
    NEO = Neo4jGraphRAG()
    NEO.stats()
except BaseException as e:  # SystemExit from connect() included
    print("Neo4j unavailable, skipping Neo4j results:", e)
    NEO = None


def trace_local(grag, q, hops, eidx):
    """Replay local search hop by hop so the report can animate it."""
    seeds = grag.link_entities(q)
    und = grag.g.to_undirected(as_view=True)
    dist, queue = {s: 0 for s in seeds}, deque(seeds)
    while queue:
        n = queue.popleft()
        if dist[n] == hops:
            continue
        for nb in und.neighbors(n):
            if nb not in dist:
                dist[nb] = dist[n] + 1
                queue.append(nb)
    steps = [{"title": "Find entities in the question",
              "text": f"Match known entity names inside the question text: {seeds}. These are the starting points of the walk. (This is why entity names must be consistent: a name the graph doesn't contain can't be matched.)",
              "nodes": seeds, "edges": [], "docs": []}]
    seen = set()
    for h in range(1, hops + 1):
        es = [(u, d["rel"], v, d["doc"]) for u, v, d in grag.g.edges(data=True)
              if u in dist and v in dist and min(dist[u], dist[v]) == h - 1]
        nodes = sorted(n for n, dv in dist.items() if dv == h)
        new_docs = sorted({e[3] for e in es} - seen)
        seen |= set(new_docs)
        reached = sum(1 for v in dist.values() if v < h)
        steps.append({"title": f"Hop {h}: walk one step outward",
                      "text": f"From the {reached} entities reached so far, follow every relationship one step further. That reaches {len(nodes)} new entities and collects {len(es)} facts, which bring in {len(new_docs)} new source document(s).",
                      "nodes": nodes, "edges": sorted({eidx[e] for e in es}), "docs": new_docs})
    steps.append({"title": "Collect sources and answer", "final": True,
                  "text": "Every fact remembers which document it came from. All those documents together are the context given to the LLM, which writes the answer and cites them. Chips below: green = needed and found, grey = found but not needed, red dashed = needed but missed.",
                  "nodes": [], "edges": [], "docs": []})
    return steps


def trace_global(grag, q, es, best, sims, eidx):
    scores = [{"c": i, "score": round(float(x), 3), "members": sorted(grag.communities[i]["members"])} for i, x in enumerate(sims)]
    members = sorted(grag.communities[best]["members"])
    docs = sorted({e[3] for e in es})
    return [
        {"title": "Compare the question with every community summary",
         "text": "No single entity is named, so there is nothing to start a walk from. Instead each community summary is scored against the question (TF-IDF here; an LLM scores them in full GraphRAG). Scores shown below.",
         "nodes": [], "edges": [], "docs": [], "scores": scores},
        {"title": "Take the best-scoring community",
         "text": f"Community {best} scores highest. All its entities and facts become the context: {len(members)} entities, {len(es)} facts, {len(docs)} source document(s).",
         "nodes": members, "edges": sorted({eidx[e] for e in es}), "docs": docs, "scores": scores},
        {"title": "Collect sources and answer", "final": True,
         "text": "The summaries and their source documents go to the LLM, which writes a corpus-wide answer. Chips below: green = needed and found, grey = found but not needed, red dashed = needed but missed.",
         "nodes": [], "edges": [], "docs": []},
    ]


def build(path, label):
    triples = [tuple(t) for t in json.loads(path.read_text())]
    grag = GraphRAG(triples=triples)
    grag_d = GraphRAG(triples=triples, dense=True)
    neo_stats = NEO.build(grag_d).stats() if NEO else None  # Neo4j holds one graph at a time: load this tab's graph, then query it
    comm_of = {n: i for i, c in enumerate(grag.communities) for n in c["members"]}
    pos = nx.spring_layout(nx.Graph(grag.g.to_undirected()), seed=11, k=1.4, iterations=300)
    nodes = [{"id": n, "x": round(560 + 520 * pos[n][0]), "y": round(330 + 290 * pos[n][1]), "c": comm_of.get(n, 0)} for n in grag.g.nodes]
    edges = [{"s": u, "t": v, "rel": d["rel"], "doc": d["doc"]} for u, v, d in grag.g.edges(data=True)]
    eidx = {(e["s"], e["rel"], e["t"], e["doc"]): i for i, e in enumerate(edges)}
    names = sorted(grag.g.nodes, key=str.lower)
    dupes = sorted({tuple(sorted((a, b))) for a in names for b in names
                    if a != b and len(a) > 3 and (a.lower() in b.lower().split(" ") or a.lower() == b.lower() or b.lower().startswith(a.lower() + " "))})
    out = []
    for it in QUESTIONS:
        _, r_docs = rag.context(it["q"], 4)
        if it["method"] == "local":
            seeds, es = grag.local_search(it["q"], it["hops"])
            how = f"Found {seeds} in the question, then walked {it['hops']} hop(s) outward along edges."
            trace = trace_local(grag, it["q"], it["hops"], eidx)
        else:
            sims = cosine_similarity(grag.cvec.transform([it["q"]]), grag.cmatrix)[0]
            best = max(range(len(grag.communities)), key=lambda i: sims[i])
            es = grag.communities[best]["edges"]
            how = f"Ranked the community summaries against the question and took the best one (community {best})."
            trace = trace_global(grag, it["q"], es, best, sims, eidx)
        out.append({**{k: it[k] for k in ("q", "kind", "method", "hops")}, "gold": sorted(it["gold"]), "rag": sorted(r_docs),
                    "graph": sorted({d for *_, d in es}), "g_edges": [eidx[e] for e in es], "how": how, "trace": trace,
                    "rag_d": sorted(rag_d.context(it["q"], 4)[1]), "graph_d": sorted(run(grag_d, rag_d, it)),
                    "neo4j": sorted(run(NEO, rag_d, it)) if NEO else None})
    return {"source": label, "neo4j_stats": neo_stats, "nodes": nodes, "edges": edges, "questions": out, "dupes": dupes,
            "communities": [{"members": sorted(c["members"]), "summary": c["summary"]} for c in grag.communities]}


def _same(a, b):
    return a.exists() and b.exists() and a.read_bytes() == b.read_bytes()


sources = []
cur, v1, v1r, v2 = ROOT / "triples_llm.json", ROOT / "triples_llm_v1.json", ROOT / "triples_llm_v1_resolved.json", ROOT / "triples_llm_v2.json"
if cur.exists() and not _same(cur, v1) and not _same(cur, v1r) and not _same(cur, v2):
    sources.append(build(cur, "LLM v3: tense-aware prompt (latest)"))
if v2.exists():
    sources.append(build(v2, "LLM v2: improved prompt"))
if v1r.exists():
    sources.append(build(v1r, "LLM v1 + merge step only"))
if v1.exists():
    sources.append(build(v1, "LLM v1, raw first attempt"))
hw = ROOT / "triples_handwritten.json"
sources.append(build(hw if hw.exists() else ROOT / "triples.json", "Hand-written (reference)"))
if NEO and cur.exists():  # leave the latest LLM graph loaded in Neo4j, so `ask --neo4j` sees it
    NEO.build(GraphRAG(triples=[tuple(t) for t in json.loads(cur.read_text())], dense=True))
data = {"docs": docs, "graphs": sources}
html = (ROOT / "graphrag_demo" / "report_template.html").read_text().replace("__DATA__", json.dumps(data))
(ROOT / "report.html").write_text(html)
print("wrote report.html with graphs:", [(g["source"], len(g["nodes"]), len(g["edges"])) for g in sources])
