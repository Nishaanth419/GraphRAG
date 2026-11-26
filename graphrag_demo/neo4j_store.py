"""GraphRAG on Neo4j.

Graph model:   (:Entity {name, embedding})-[:REL {rel, doc}]->(:Entity)      one edge per extracted triple
               (:Entity)-[:IN_COMMUNITY]->(:Community {id, summary, embedding})
Vector indexes: entity_embedding, community_embedding (cosine, 384 dims).
Entity linking uses the entity vector index, local search is a Cypher variable-length path query,
global search uses the community vector index (or an LLM scorer). Community detection (Louvain) still runs in
Python/networkx, because Neo4j's graph-algorithm library (GDS) is a separate plugin; the results are stored here.

    python -m graphrag_demo.neo4j_store load [triples_file]      # (re)build the graph in Neo4j
"""
import os
import time

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

from .common import ROOT
from .embed import embed
from .graph import GraphRAG, question_spans

DIM = 384
OVERFETCH = 10  # the vector index is approximate (HNSW): asking for k=1 can miss the true nearest neighbour, so fetch more and re-rank


def _credentials():
    auth = os.getenv("NEO4J_AUTH")
    env_file = ROOT / "neo4j.env"
    if not auth and env_file.exists():
        auth = dict(l.split("=", 1) for l in env_file.read_text().splitlines() if "=" in l).get("NEO4J_AUTH")
    if not auth:
        raise SystemExit("Set NEO4J_AUTH=user/password or create neo4j.env")
    user, password = auth.split("/", 1)
    return os.getenv("NEO4J_URI", "bolt://localhost:7687"), (user, password)


def connect(wait: int = 60):
    uri, auth = _credentials()
    driver = GraphDatabase.driver(uri, auth=auth)
    deadline = time.time() + wait
    while True:
        try:
            driver.verify_connectivity()
            return driver
        except (ServiceUnavailable, OSError):
            if time.time() > deadline:
                raise SystemExit(f"Neo4j not reachable at {uri}")
            time.sleep(2)


class Neo4jGraphRAG:
    def __init__(self, driver=None, link_threshold: float = 0.75, scorer=None, min_score: int = 5):
        self.driver = driver or connect()
        self.link_threshold, self.scorer, self.min_score = link_threshold, scorer, min_score

    def _run(self, query, **params):
        with self.driver.session() as s:
            return s.run(query, **params).data()

    # ---------- build ----------
    def build(self, grag: GraphRAG):
        """Write an in-memory GraphRAG (graph, communities, summaries) into Neo4j with embeddings."""
        names = list(grag.g.nodes)
        comm_of = {n: i for i, c in enumerate(grag.communities) for n in c["members"]}
        nvec, cvec = embed(names), embed(c["summary"] for c in grag.communities)
        with self.driver.session() as s:
            s.run("MATCH (n) WHERE n:Entity OR n:Community DETACH DELETE n").consume()
            s.run("CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE").consume()
            for label, index in (("Entity", "entity_embedding"), ("Community", "community_embedding")):
                s.run(f"CREATE VECTOR INDEX {index} IF NOT EXISTS FOR (n:{label}) ON (n.embedding) "
                      f"OPTIONS {{indexConfig: {{`vector.dimensions`: {DIM}, `vector.similarity_function`: 'cosine'}}}}").consume()
            s.run("UNWIND $rows AS r CREATE (e:Entity {name: r.name}) WITH e, r "
                  "CALL db.create.setNodeVectorProperty(e, 'embedding', r.vec)",
                  rows=[{"name": n, "vec": [float(x) for x in v]} for n, v in zip(names, nvec)]).consume()
            s.run("UNWIND $rows AS r MATCH (a:Entity {name: r.s}), (b:Entity {name: r.t}) "
                  "CREATE (a)-[:REL {rel: r.rel, doc: r.doc}]->(b)",
                  rows=[{"s": u, "t": v, "rel": d["rel"], "doc": d["doc"]} for u, v, d in grag.g.edges(data=True)]).consume()
            s.run("UNWIND $rows AS r CREATE (c:Community {id: r.id, summary: r.summary}) WITH c, r "
                  "CALL db.create.setNodeVectorProperty(c, 'embedding', r.vec)",
                  rows=[{"id": i, "summary": c["summary"], "vec": [float(x) for x in v]}
                        for i, (c, v) in enumerate(zip(grag.communities, cvec))]).consume()
            s.run("UNWIND $rows AS r MATCH (e:Entity {name: r.name}), (c:Community {id: r.c}) CREATE (e)-[:IN_COMMUNITY]->(c)",
                  rows=[{"name": n, "c": comm_of[n]} for n in names]).consume()
            s.run("CALL db.awaitIndexes(120)").consume()
        return self

    def stats(self):
        return self._run("MATCH (e:Entity) WITH count(e) AS entities "
                         "MATCH ()-[r:REL]->() WITH entities, count(r) AS rels "
                         "MATCH (c:Community) RETURN entities, rels, count(c) AS communities")[0]

    # ---------- local search ----------
    def link_entities(self, question: str) -> list[str]:
        """Entity linking through the vector index. Neo4j cosine scores are (1+cos)/2, so convert back to cosine."""
        spans = question_spans(question)
        best = {}
        for span, vec in zip(spans, embed(spans)):
            rows = self._run("CALL db.index.vector.queryNodes('entity_embedding', $k, $v) YIELD node, score "
                             "RETURN node.name AS name, score ORDER BY score DESC LIMIT 1", k=OVERFETCH, v=[float(x) for x in vec])
            if rows and 2 * rows[0]["score"] - 1 >= self.link_threshold:
                best[rows[0]["name"]] = max(best.get(rows[0]["name"], 0), rows[0]["score"])
        return sorted(best, key=lambda n: -best[n])

    def local_search(self, question: str, hops: int = 2):
        seeds = self.link_entities(question)
        if not seeds:
            return seeds, []
        hops = int(hops)
        dist = {r["name"]: r["d"] for r in self._run(
            f"MATCH (s:Entity) WHERE s.name IN $seeds MATCH p = (s)-[:REL*0..{hops}]-(n:Entity) "
            "RETURN n.name AS name, min(length(p)) AS d", seeds=seeds)}
        rows = self._run(
            "MATCH (a:Entity)-[r:REL]->(b:Entity) WHERE a.name IN keys($dist) AND b.name IN keys($dist) "
            "AND (CASE WHEN $dist[a.name] < $dist[b.name] THEN $dist[a.name] ELSE $dist[b.name] END) < $hops "
            "RETURN a.name AS u, r.rel AS rel, b.name AS v, r.doc AS doc", dist=dist, hops=hops)
        return seeds, sorted({(r["u"], r["rel"], r["v"], r["doc"]) for r in rows})

    def local_context(self, question: str, hops: int = 2):
        seeds, edges = self.local_search(question, hops)
        text = f"Seed entities: {seeds}\n" + "\n".join(f"{u} --{r}--> {v}   [{d}]" for u, r, v, d in edges)
        return text, {d for *_, d in edges}

    # ---------- global search ----------
    def _community_docs(self, ids):
        rows = self._run("MATCH (c:Community) WHERE c.id IN $ids MATCH (c)<-[:IN_COMMUNITY]-(a)-[r:REL]->(b)-[:IN_COMMUNITY]->(c) "
                         "RETURN DISTINCT r.doc AS doc", ids=list(ids))
        return {r["doc"] for r in rows}

    def global_context(self, question: str, top: int = 2):
        if self.scorer:
            comms = self._run("MATCH (c:Community) RETURN c.id AS id, c.summary AS summary ORDER BY id")
            scores = {c["id"]: self.scorer(question, c["summary"]) for c in comms}
            ids = [i for i in sorted(scores, key=lambda i: -scores[i]) if scores[i] >= self.min_score] or [max(scores, key=scores.get)]
            summ = {c["id"]: c["summary"] for c in comms}
            return "\n".join(f"Community {i} (score {scores[i]}): {summ[i]}" for i in ids), self._community_docs(ids)
        vec = [float(x) for x in embed([question])[0]]
        rows = self._run("CALL db.index.vector.queryNodes('community_embedding', $k, $v) YIELD node, score "
                         "RETURN node.id AS id, node.summary AS summary ORDER BY score DESC LIMIT $top", k=max(OVERFETCH, top), top=top, v=vec)
        return "\n".join(f"Community {r['id']}: {r['summary']}" for r in rows), self._community_docs([r["id"] for r in rows])


if __name__ == "__main__":
    import json
    import sys
    cmd, *rest = sys.argv[1:] or ["load"]
    if cmd != "load":
        raise SystemExit(__doc__)
    path = ROOT / (rest[0] if rest else "triples_llm.json")
    store = Neo4jGraphRAG().build(GraphRAG(triples=[tuple(t) for t in json.loads(path.read_text())]))
    print(f"loaded {path.name} into Neo4j:", store.stats())
