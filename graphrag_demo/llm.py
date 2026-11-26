"""Optional LLM layer using Groq. Needs GROQ_API_KEY. Not used by the offline demo."""
import json
import os

from groq import Groq, NotFoundError

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
_client = None


def _ask(prompt: str, max_tokens: int = 1500) -> str:
    global _client
    _client = _client or Groq()
    try:
        resp = _client.chat.completions.create(
            model=MODEL, max_tokens=max_tokens + 3000, temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
    except NotFoundError:
        ids = sorted(m.id for m in _client.models.list().data)
        raise SystemExit(f"Model '{MODEL}' not available to your key. Set GROQ_MODEL to one of:\n  " + "\n  ".join(ids))
    return resp.choices[0].message.content


EXTRACT_PROMPT = """You build a knowledge graph from a document. Output ONLY a JSON list of [subject, relation, object] triples.

Rules:
1. Subjects and objects must be NAMED entities: a Person, Team, Service/System, Vendor/Company, or Date. Write them as proper names with capital letters. NEVER use common nouns ("major outage", "customers", "status updates", "nightly") as entities.
2. Do not create nodes for events or attributes. Fold them into the relation instead.
   "Zeta Hosting suffered a long outage on May 2" -> ["Zeta Hosting", "had outage on", "May 2"]
3. Use the shortest canonical name: "Checkout" not "Checkout service", "Mobile App" not "Mobile App 4.2".
4. If an entity matches one in KNOWN ENTITIES, reuse that exact string.
5. Keep tense and certainty in the relation. Plans and promises are NOT facts that already happened:
   "Dana will migrate PayStore to Omega Cloud by Q4" -> ["Dana", "plans to migrate", "PayStore"], ["PayStore", "planned to move to", "Omega Cloud"]; never "migrates" or "migrated".
6. Skip durations, counts, percentages and other attributes. Relations are short lowercase verb phrases ("owns", "depends on", "hosted on", "reports to", "leads").

Example. Text: "Dana runs the Billing team. Billing uses PayStore, hosted on Zeta Hosting."
Output: [["Dana", "leads", "Billing team"], ["Billing team", "uses", "PayStore"], ["PayStore", "hosted on", "Zeta Hosting"]]

KNOWN ENTITIES: {known}

Text:
{text}"""


def extract_triples(doc_id: str, text: str, known: list[str] | None = None) -> list[list[str]]:
    """The expensive step of GraphRAG: an LLM reads every document and emits entities + relationships."""
    out = _ask(EXTRACT_PROMPT.format(known=json.dumps(sorted(known or [])), text=text))
    out = out[out.index("["): out.rindex("]") + 1]
    return [[s, r, o, doc_id] for s, r, o in json.loads(out)]


SUMMARY_PROMPT = """Write a 2-3 sentence summary of this group of related entities, using ONLY the facts given.
Do not turn plans into completed actions ("will migrate" must not become "began migrating"). Do not add details that are not in the facts.
Mention every entity that plays a risk, dependency or ownership role.
Entities: {members}
Facts: {facts}"""


def summarize_community(members: list[str], facts: str) -> str:
    return _ask(SUMMARY_PROMPT.format(members=members, facts=facts), 300)


def cached_summarizer(path):
    """summarize_community with an on-disk cache, so reruns and comparisons don't repeat the API calls."""
    import hashlib
    cache = json.loads(path.read_text()) if path.exists() else {}

    def fn(members, facts):
        key = hashlib.sha256(json.dumps([MODEL, SUMMARY_PROMPT, members, facts]).encode()).hexdigest()
        if key not in cache:
            cache[key] = summarize_community(members, facts)
            path.write_text(json.dumps(cache, indent=1))
        return cache[key]
    return fn


def answer(question: str, context: str) -> str:
    return _ask(f"Answer using ONLY this context. If it is insufficient, say so.\n\nContext:\n{context}\n\nQuestion: {question}", 500)


SCORE_PROMPT = """You are scoring a summary of one part of a knowledge graph for how useful it is for answering a question.
Reply with ONLY an integer from 0 (irrelevant) to 10 (contains the key information).

Question: {question}

Summary: {summary}"""


def score_community(question: str, summary: str) -> int:
    """The 'map' step of global search: an LLM rates each community summary against the question."""
    import re
    out = _ask(SCORE_PROMPT.format(question=question, summary=summary), 20)
    m = re.search(r"\d+", out or "")
    return min(10, int(m.group())) if m else 0


def cached_scorer(path):
    import hashlib
    cache = json.loads(path.read_text()) if path.exists() else {}

    def fn(question, summary):
        key = hashlib.sha256(json.dumps([MODEL, SCORE_PROMPT, question, summary]).encode()).hexdigest()
        if key not in cache:
            cache[key] = score_community(question, summary)
            path.write_text(json.dumps(cache, indent=1))
        return cache[key]
    return fn
