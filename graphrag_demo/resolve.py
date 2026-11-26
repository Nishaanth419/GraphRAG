"""Entity resolution + cleanup applied to LLM-extracted triples.

Heuristics (deliberately simple, tuned to this corpus; real systems add embedding similarity or an LLM merge pass):
  1. merge names that differ only by case ("Payments Team" / "Payments team")
  2. strip generic trailing words ("Checkout service" -> "Checkout", "Corvus Cloud outage" -> "Corvus Cloud")
  3. strip version numbers ("Mobile App 4.2" -> "Mobile App")
  4. drop non-entities: names with no capital letter ("nightly", "slow", "customer orders", "8 percent")
  5. drop self-loops and duplicate edges
"""
import re
from collections import Counter

GENERIC_SUFFIX = {"service", "outage", "down", "release", "incident", "renewal", "contract"}


def _strip(name: str) -> str:
    words = name.split()
    while len(words) > 1 and (words[-1].lower() in GENERIC_SUFFIX or re.fullmatch(r"\d+(\.\d+)+", words[-1])):
        words.pop()
    return " ".join(words)


def resolve(triples):
    triples = [tuple(t) for t in triples]
    forms = {}  # casefolded stripped name -> Counter of stripped surface forms (to pick the canonical casing)
    for s, _, o, _ in triples:
        for name in (s, o):
            forms.setdefault(_strip(name).casefold(), Counter())[_strip(name)] += 1
    canon = {k: c.most_common(1)[0][0] for k, c in forms.items()}
    out, seen = [], set()
    for s, r, o, d in triples:
        cs, co = canon[_strip(s).casefold()], canon[_strip(o).casefold()]
        if not (re.search(r"[A-Z]", cs) and re.search(r"[A-Z]", co)) or cs.casefold() == co.casefold():
            continue
        if (cs, r, co, d) not in seen:
            seen.add((cs, r, co, d))
            out.append([cs, r, co, d])
    return out


if __name__ == "__main__":
    import json
    import sys
    src, dst = sys.argv[1], sys.argv[2]
    res = resolve(json.load(open(src)))
    json.dump(res, open(dst, "w"), indent=1)
    ents = {x for s, _, o, _ in res for x in (s, o)}
    print(f"{src} -> {dst}: {len(res)} triples, {len(ents)} entities")
