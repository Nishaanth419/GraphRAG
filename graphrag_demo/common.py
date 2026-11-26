"""Shared helpers: corpus loading and a tiny tokenizer used by every TF-IDF index."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "corpus"
TRIPLES_PATH = ROOT / "triples.json"


def load_docs() -> dict[str, str]:
    return {p.stem: p.read_text().strip() for p in sorted(CORPUS_DIR.glob("*.txt"))}


def load_triples() -> list[tuple[str, str, str, str]]:
    return [tuple(t) for t in json.loads(TRIPLES_PATH.read_text())]


def tokenize(text: str) -> list[str]:
    # lowercase words with a crude plural strip ("risks" -> "risk"); no real stemmer needed here
    return [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z0-9]+", text.lower())]
