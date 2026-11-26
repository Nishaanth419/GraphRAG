# GraphRAG vs RAG: a small, runnable comparison

A 14-document corpus about a fictional company (Helio Labs) where the facts needed to answer a
question are scattered across several files. Six questions go through plain RAG and several GraphRAG
variants, and each is scored on which source documents it retrieved. `report.html` shows it all visually,
including a step-by-step trace of one question walking through the graph.

## Setup and run

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m graphrag_demo.compare triples_handwritten.json -v   # offline, no API key needed
.venv/bin/python -m graphrag_demo.make_report                           # builds report.html
```

With a Groq key (`export GROQ_API_KEY=...`, optional `GROQ_MODEL`, default `openai/gpt-oss-120b`):

```bash
.venv/bin/python -m graphrag_demo.build_index                           # LLM extracts the graph -> triples_llm.json
.venv/bin/python -m graphrag_demo.compare triples_llm.json -v --llm-summaries
.venv/bin/python -m graphrag_demo.ask "What are the biggest vendor risks across our systems?" --mode global --dense --llm
.venv/bin/python -m graphrag_demo.ask "Whose teams are affected if Corvus Cloud goes down?" --mode local --hops 5 --dense
```

## Neo4j (graph and embeddings stored in a database)

The graph, the entity and community embeddings and the community summaries can live in Neo4j instead of memory.
Entities are `(:Entity {name, embedding})` nodes, triples are `[:REL {rel, doc}]` edges, communities are
`(:Community {summary, embedding})` nodes. Two vector indexes (`entity_embedding`, `community_embedding`) handle entity
linking and global ranking, and local search is a Cypher variable-length path query.

```bash
# one-time: a dedicated container on localhost only, data in a named volume (survives restarts)
printf 'NEO4J_AUTH=neo4j/%s\n' "$(openssl rand -hex 12)" > neo4j.env && chmod 600 neo4j.env
docker run -d --name graphrag-neo4j --env-file neo4j.env -p 127.0.0.1:7474:7474 -p 127.0.0.1:7687:7687 -v graphrag_neo4j_data:/data neo4j:5-community

.venv/bin/python -m graphrag_demo.neo4j_store load triples_llm.json                  # write the graph into Neo4j
.venv/bin/python -m graphrag_demo.compare triples_llm.json --neo4j                   # adds a "Neo4j emb" column
.venv/bin/python -m graphrag_demo.ask "Whose teams are affected if Corvus Cloud goes down?" --neo4j --dense --hops 5
```

Browse it at http://localhost:7474 (user `neo4j`, password in `neo4j.env`), for example
`MATCH p=(:Entity)-[:REL]->(:Entity) RETURN p`. `neo4j.env` holds the password; keep it out of version control.

Results match the in-memory version on all six questions (100% / 28%). Two things to know:
- **The vector index is approximate.** Asking for just the top 1 result returned the wrong community once, so the code fetches 10
  candidates and re-ranks them. Neo4j's cosine score is `(1 + cosine) / 2`, which the code converts back.
- **Community detection still runs in Python** (networkx Louvain) and the result is stored. Running it inside Neo4j needs the
  separate Graph Data Science plugin.

## RAG vs GraphRAG

| | RAG | GraphRAG |
|---|---|---|
| Index | embed chunks | LLM extracts entities + relationships into a graph, the graph is clustered into communities, each community summarized |
| Retrieve | chunks most similar to the question | local: entities in the question, then walk the graph. global: rank community summaries |
| Vectors | the main retrieval method | support only: match question phrases to entities, rank summaries, find duplicate entities |
| Good at | "what does doc X say about Y" | multi-hop ("manager of the team that owns Checkout"), "what is affected by Z", themes across the corpus |
| Cost | cheap to index | expensive to index (an LLM call per chunk, plus summaries) |

Similarity search finds text that *looks like* the question. The deep-chain question
("Corvus Cloud down -> LedgerDB/IndexStore -> Checkout/Search/Billing -> owning teams -> managers")
shares almost no words with the org-chart documents it needs, so RAG misses them. Walking the graph finds them.

## Results

Average over six questions, on the LLM-extracted graph (`triples_llm.json`, recall / precision of retrieved documents):

| RAG (word counts) | RAG (embeddings) | GraphRAG (exact names) | GraphRAG (embeddings) | + LLM summaries | + LLM scoring |
|---|---|---|---|---|---|
| 81% / 38% | 67% / 21% | 67% / 19% | **100% / 28%** | 100% / 28% | 100% / 28% |

- **Multi-hop is where GraphRAG wins.** Deep chain: 100% vs 38% (RAG with embeddings: 0%). Manager of the team that
  owns Checkout: 100% vs 50%.
- **Simple lookups are a tie**, and RAG has better precision (25% vs 17%). GraphRAG buys recall and pays in noise.
- **Embeddings fix paraphrased entities.** "the payments group" never matches the node "Payments team" by string,
  so exact matching scores 0%; embedding-based entity linking scores 100%. Not everything: "the purchase service"
  (meaning Checkout) still isn't matched.
- **Embeddings alone hurt plain RAG on the deep chain** (0%): dense retrieval matches by topic, not by chain of connections.
- **LLM summaries and LLM scoring added nothing measurable here**, but the scorer is the safer design: it rated the
  right community 8-9 out of 10 and the rest 0, while embeddings won by a margin of about 0.1 (0.01 on an earlier graph,
  where they picked the wrong community).

### Extraction quality moved the results more than anything else

| Graph | Entities | What happened |
|---|---|---|
| Hand-written (reference) | 20 | 100% on the original four questions |
| LLM v1 (first prompt) | 57 | "Checkout" vs "Checkout service", "nightly" and "slow" as entities. Manager question 0%, vendor-risk question 0% |
| LLM v1 + merge step (`resolve.py`) | 28 | Fixed those two, but dropped the outage date: the model hung it off a node called "major outage", which the cleanup removed |
| LLM v2 (stricter prompt) | 22 | 100% on all four. The prompt alone did what the merge step was trying to do |
| LLM v3 (tense-aware prompt) | 21 | "will migrate" had been extracted as `migrates`, so summaries said the migration was done. Now `plans to migrate`. 100% on all six |

The v2 problem was only visible in the summaries: they repeated a fact that the extraction had wrongly turned
from a plan into a completed action. An LLM can only summarize what it is given.

## Files
- `corpus/` the documents. `triples_handwritten.json` the hand-written graph. `triples_llm*.json` the Groq extractions
  (v1 raw, v1 merged, v2, latest). `triples.json` is a copy of the hand-written graph.
- `graphrag_demo/rag.py` baseline (word counts or embeddings). `graph.py` GraphRAG: graph, Louvain communities,
  local and global search, embedding-based entity linking, optional LLM scoring.
- `graphrag_demo/embed.py` local sentence embeddings (all-MiniLM-L6-v2). `resolve.py` entity merge and cleanup.
- `graphrag_demo/neo4j_store.py` the Neo4j version of GraphRAG (load, vector-index entity linking, Cypher local search, global search).
- `graphrag_demo/llm.py` Groq extraction, summaries, scoring, answers. `build_index.py` runs extraction.
- `graphrag_demo/compare.py` the scoreboard. `ask.py` one-off questions. `make_report.py` builds `report.html`.
- `summaries_cache.json`, `scores_cache.json` cached LLM calls, so reruns are free.

## Limitations
- **Small and self-made.** Six questions on 14 documents, all written by the same person who wrote the corpus.
  Recall is measured on retrieved *documents*, not on whether a final answer is correct.
- **Tuned on the test set.** The entity-matching threshold (0.75) and the extraction prompt were adjusted while
  looking at these questions. The real test is a different corpus with new questions.
- **The merge step is tuned to this corpus** (its suffix list and the rule that drops lowercase names). The v2 and
  v3 prompts made it unnecessary here; a general version would use embedding similarity or an LLM merge pass.
- **Community summaries lose edges between communities.** Corvus Cloud and IndexStore landed in different
  communities in one graph, so a summary of the vendor risk mentioned only LedgerDB. Local search doesn't have this problem.
- **Hop count is a manual knob** (the manager question needs 3, the deep chain 5). Production systems let an LLM decide.
- **Entity resolution is the hard part of real extraction.** It broke the first LLM graph and is only partly handled.

## Next steps
1. Run on a different corpus (your own notes or a book) with new questions, and see whether the prompt still holds.
2. Add the Graph Data Science plugin so Louvain runs inside Neo4j, and store the LLM-written summaries there.
3. Compare with Microsoft's `graphrag` package on the same documents.
