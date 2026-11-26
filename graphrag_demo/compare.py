"""Run the same questions through RAG and GraphRAG (word-count vs embedding retrieval) and score which source documents each retrieved.
   python -m graphrag_demo.compare [triples_file] [-v] [--llm-summaries] [--neo4j]"""
import json

from .graph import GraphRAG
from .rag import RAG

# gold = documents you must read to answer correctly
QUESTIONS = [
    {"q": "When did the Corvus Cloud outage happen?", "kind": "simple lookup",
     "gold": {"d04_corvus_outage"}, "method": "local", "hops": 1},
    {"q": "Who is the manager of the team that owns Checkout?", "kind": "multi-hop (3 docs)",
     "gold": {"d01_payments_team", "d05_org_engineering"}, "method": "local", "hops": 3},
    {"q": "Whose teams are affected if Corvus Cloud goes down?", "kind": "multi-hop (deep chain)",
     "gold": {"d03_ledgerdb_hosting", "d06_search_service", "d02_checkout_architecture", "d09_billing_reports",
              "d07_mobile_app", "d01_payments_team", "d05_org_engineering", "d08_org_product"},
     "method": "local", "hops": 5},
    {"q": "What are the biggest vendor risks for the company?", "kind": "global / thematic",
     "gold": {"d11_vendor_risk"}, "method": "global", "hops": None},
    # added for the embeddings comparison
    {"q": "Who does the head of the payments group report to?", "kind": "multi-hop, paraphrased entity",
     "gold": {"d01_payments_team", "d05_org_engineering"}, "method": "local", "hops": 2},
    {"q": "What are the biggest vendor risks across our systems?", "kind": "global, original wording",
     "gold": {"d11_vendor_risk"}, "method": "global", "hops": None},
]


def score(retrieved: set[str], gold: set[str]) -> tuple[float, float]:
    hit = len(retrieved & gold)
    return hit / len(gold), (hit / len(retrieved) if retrieved else 0.0)


def run(grag, rag, item):
    if item["method"] == "local":
        _, docs = grag.local_context(item["q"], item["hops"])
    else:
        _, docs = grag.global_context(item["q"], top=1)
    return docs


def main(triples_path=None, show_missed=False, llm_summaries=False, neo4j=False):
    from .common import ROOT
    path = ROOT / (triples_path or "triples_llm.json")
    triples = [tuple(t) for t in json.loads(path.read_text())]
    rag_t, rag_d = RAG(), RAG(dense=True)
    g_t, g_d = GraphRAG(triples=triples), GraphRAG(triples=triples, dense=True)
    g_l = g_m = None
    if llm_summaries:
        import os
        if not os.getenv("GROQ_API_KEY"):
            raise SystemExit("--llm-summaries needs GROQ_API_KEY")
        from . import llm
        summ = llm.cached_summarizer(ROOT / "summaries_cache.json")
        g_l = GraphRAG(triples=triples, dense=True, summarizer=summ)
        g_m = GraphRAG(triples=triples, dense=True, summarizer=summ, scorer=llm.cached_scorer(ROOT / "scores_cache.json"))
    print(f"Graph: {path.name} ({g_t.g.number_of_nodes()} entities, {g_t.g.number_of_edges()} relationships)\n")
    systems = [("RAG tfidf", lambda it: rag_t.context(it["q"], 4)[1]), ("RAG emb", lambda it: rag_d.context(it["q"], 4)[1]),
               ("Graph exact+tfidf", lambda it: run(g_t, rag_t, it)), ("Graph emb", lambda it: run(g_d, rag_d, it))]
    if neo4j:
        from .neo4j_store import Neo4jGraphRAG
        store = Neo4jGraphRAG().build(g_d)  # reloads the Neo4j graph from the same triples and offline summaries as "Graph emb"
        systems.append(("Neo4j emb", lambda it: run(store, rag_d, it)))
    if g_l:
        systems.append(("Graph emb+LLM sum", lambda it: run(g_l, rag_d, it)))
        systems.append(("Graph LLM-scored", lambda it: run(g_m, rag_d, it)))
    print(f"{'question (recall / precision)':<54}" + "".join(f"{n:>19}" for n, _ in systems))
    print("-" * (54 + 19 * len(systems)))
    tot = {n: [0, 0] for n, _ in systems}
    for it in QUESTIONS:
        row, miss = f"{it['q'][:52]:<54}", []
        for n, fn in systems:
            docs = fn(it)
            r, p = score(docs, it["gold"])
            tot[n][0] += r; tot[n][1] += p
            row += f"{r:>10.0%} /{p:>5.0%}  "
            miss.append((n, sorted(d[:3] for d in it["gold"] - docs)))
        print(row)
        print(f"{'   (' + it['kind'] + ')':<54}")
        if show_missed:
            print("      missed:", "; ".join(f"{n}={m}" for n, m in miss if m))
    print("-" * (54 + 19 * len(systems)))
    print(f"{'average':<54}" + "".join(f"{tot[n][0] / len(QUESTIONS):>10.0%} /{tot[n][1] / len(QUESTIONS):>5.0%}  " for n, _ in systems))
    print("\nrecall = share of needed documents retrieved; precision = share of retrieved documents that were needed")


if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    main(args[0] if args else None, show_missed="-v" in sys.argv, llm_summaries="--llm-summaries" in sys.argv, neo4j="--neo4j" in sys.argv)
