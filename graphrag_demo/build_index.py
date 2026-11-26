"""Extract the knowledge graph with an LLM -> triples_llm.json  (needs GROQ_API_KEY).
Documents are processed in order and the entities found so far are passed to each call, so names stay consistent.
The raw model output is kept in triples_llm_raw.json; triples_llm.json is the cleaned version (see resolve.py)."""
import json

from . import llm
from .common import ROOT, load_docs
from .resolve import resolve

def main():
    raw, known = [], set()
    for doc_id, text in load_docs().items():
        try:
            got = llm.extract_triples(doc_id, text, sorted(known))
        except Exception as e:  # bad JSON etc: skip the doc, keep going
            print(f"  {doc_id}: FAILED ({type(e).__name__}: {e})")
            continue
        raw += got
        known |= {x for s, _, o, _ in got for x in (s, o)}
        print(f"  {doc_id}: {len(got)} triples")
    (ROOT / "triples_llm_raw.json").write_text(json.dumps(raw, indent=1))
    clean = resolve(raw)
    (ROOT / "triples_llm.json").write_text(json.dumps(clean, indent=1))
    print(f"raw {len(raw)} triples -> cleaned {len(clean)} triples, {len({x for s, _, o, _ in clean for x in (s, o)})} entities")


if __name__ == "__main__":
    main()
