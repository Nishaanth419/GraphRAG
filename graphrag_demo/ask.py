"""Ask one question.
   python -m graphrag_demo.ask "question" --mode rag|local|global [--hops N] [--dense] [--llm] [--triples FILE]"""
import argparse
import os

from .common import ROOT
from .graph import GraphRAG
from .rag import RAG


def main():
    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("--mode", choices=["rag", "local", "global"], default="local")
    p.add_argument("--hops", type=int, default=3)
    p.add_argument("--dense", action="store_true", help="use embeddings (RAG retrieval, entity linking, community ranking)")
    p.add_argument("--llm", action="store_true", help="LLM-written community summaries (global) and an LLM-written answer")
    p.add_argument("--neo4j", action="store_true", help="query the graph stored in Neo4j (load it first with neo4j_store load)")
    p.add_argument("--triples", default="triples_llm.json", help="graph file in the project folder (default: the Groq v2 graph)")
    a = p.parse_args()
    if a.llm and not os.getenv("GROQ_API_KEY"):
        raise SystemExit("--llm needs GROQ_API_KEY")
    if a.llm:
        from . import llm
    if a.mode == "rag":
        ctx, docs = RAG(dense=a.dense).context(a.question)
    else:
        import json
        triples = None if a.neo4j else [tuple(t) for t in json.loads((ROOT / a.triples).read_text())]
        summarizer = llm.cached_summarizer(ROOT / "summaries_cache.json") if a.llm and a.mode == "global" else None
        scorer = llm.cached_scorer(ROOT / "scores_cache.json") if a.llm and a.mode == "global" else None
        if a.neo4j:
            from .neo4j_store import Neo4jGraphRAG
            g = Neo4jGraphRAG(scorer=scorer)
        else:
            g = GraphRAG(triples=triples, summarizer=summarizer, dense=a.dense, scorer=scorer)
        ctx, docs = g.local_context(a.question, a.hops) if a.mode == "local" else g.global_context(a.question)
    print(ctx, "\n\nSource documents:", sorted(docs))
    if a.llm:
        print("\nAnswer:", llm.answer(a.question, ctx))


if __name__ == "__main__":
    main()
