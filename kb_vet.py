"""Retrieval over the pet-health guidance PetsAIHealth already curates (RAG for the nurse chat).

Corpus (all paraphrased, each with its MSD / ESCCAP page where there is one):
  * vet_knowledge.TOPICS : owner-level facts per topic
  * vet_rules.RULES      : the one-line guidance behind every safety rule (rules that already fired as alerts are skipped)

`retrieve()` returns the prompt block plus the chunks, so the caller can later call `cite()` on the model's reply.
Kill switch: env KB_RAG=off keeps only the old keyword topics (vet_knowledge.relevant), exactly as before.
No Streamlit, no network: unit-testable (dev/test_kb.py).
"""
import os

import kb
import vet_knowledge
import vet_rules

_INDEX = None
_MAX = 5


def _label(url, basis):
    if not url:
        return "PetsAIHealth safety policy"
    if "esccap" in url:
        return "ESCCAP guidelines"
    return kb.title_from_url(url)


def chunks():
    out = []
    seen = {}
    for tid, _rx, sp, txt, url in vet_knowledge._COMPILED:
        seen[tid] = seen.get(tid, 0) + 1          # a topic id may appear twice in TOPICS: keep chunk ids unique
        cid = f"topic:{tid}" if seen[tid] == 1 else f"topic:{tid}#{seen[tid]}"
        out.append(kb.Chunk(cid, txt, _label(url, "msd"), url, tuple(sp)))
    for r in vet_rules.RULES:
        out.append(kb.Chunk(f"rule:{r.id}", r.en, _label(r.source, r.basis), r.source, tuple(r.species)))
    return out


def index():
    global _INDEX
    if _INDEX is None:
        # Topics stay on their keyword gates (vet_knowledge.relevant: deterministic, tested); BM25 searches the rule guidance.
        _INDEX = kb.Index([c for c in chunks() if c.id.startswith("rule:")])
    return _INDEX


def enabled():
    return os.environ.get("KB_RAG", "on").strip().lower() not in ("off", "0", "false", "no")


def retrieve(species_key, texts, alert_ids=(), limit=_MAX):
    """-> (prompt_block, found). Keyword topics (the old, deterministic path) always count; BM25 adds what they miss."""
    texts = [t for t in (texts or []) if t]
    found = []
    seen = set()
    for tid, txt, url in vet_knowledge.relevant(species_key, texts, limit):
        c = kb.Chunk(f"topic:{tid}", txt, _label(url, "msd"), url)
        found.append(kb.Found(c, 99.0, 99))
        seen.add(c.id)
    if enabled():
        skip = {f"rule:{i}" for i in alert_ids} | seen
        for f in index().search(texts, species=species_key, k=limit, min_cover=0.2, exclude=skip):
            if len(found) >= limit:
                break
            found.append(f)
    found = found[:limit]
    if not found:
        return "", []
    block = vet_knowledge.ALWAYS + "\n\n" + kb.prompt_block(found)
    return block, found


def cite(reply, found, lang="en"):
    return kb.cite(reply, found, lang)


def ids(found):
    return ",".join(f.chunk.id for f in found)[:200]
