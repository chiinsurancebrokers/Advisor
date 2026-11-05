"""Retrieval over the clinical guidance Asklepios already curates (RAG for the nurse chat).

Corpus (paraphrased, every chunk keeps its published source):
  * triage_rules.RULES      : the guidance sentence behind each red-flag rule, with its published source (PubMed record, NHS or MedlinePlus page)
  * lifestyle_info.ITEMS    : the reviewed-source lifestyle texts (only items that carry a verified source)
Rules that already fired as alerts for this chat are skipped (the alert block states them).

Small on purpose: this is the structure (retrieve -> tag -> cite), not yet a large knowledge base. Adding sources means adding
Chunks here, each with a URL and a licence check (docs/SOURCES.md). No Streamlit, no network: unit-testable (tests/test_kb_med.py).
Kill switch: env KB_RAG=off.
"""
import os

import kb
import lifestyle_info
import triage_rules

# Human symptom words the shared glossary does not know. Added here so kb.py stays identical in every repo.
kb.GLOSSARY.update({
    "οστεοπορ": "osteoporosis bone", "οστεοπεν": "osteopenia bone", "οστικ": "bone density", "πυκνοτ": "density",
    "ασβεστ": "calcium", "βιταμιν": "vitamin", "ασκηση": "exercise", "πτωσ": "falls fall", "πεσιμ": "falls fall",
    "θωρακ": "chest", "στηθ": "chest", "πιεση": "pressure", "σφιξ": "tightness pressure", "ιδρω": "sweat sweating", "ναυτι": "nausea",
    "βραχιον": "arm", "χερι": "arm", "σαγον": "jaw", "παραλυ": "weakness paralysis", "μουδι": "numb numbness", "αμιλητ": "speech",
    "θολη": "blurred vision", "οραση": "vision", "κεφαλ": "head headache", "πονοκεφ": "headache", "τραυματ": "injury trauma",
    "εγκυμ": "pregnant pregnancy", "μωρο": "infant baby", "βρεφ": "infant baby", "παιδι": "child", "αλλεργ": "allergy allergic",
    "προσωπ": "face", "μιλα": "speech speak", "μιλω": "speech speak", "ομιλ": "speech", "κρεμ": "droop", "οιδημ": "swelling", "πρηξ": "swelling", "ανασ": "breathing", "αναπνο": "breathing", "κρισ": "attack", "ψυχ": "mental",
})

# PubMed id -> short label shown to the user (titles verified in triage_rules.py)
_TITLES = {
    "37622654": "ESC 2023 guidelines: acute coronary syndromes",
    "31504429": "ESC 2019 guidelines: acute pulmonary embolism",
    "25173340": "ESC 2014 guidelines: aortic diseases",
    "31662037": "AHA/ASA 2019: early management of acute ischemic stroke",
    "12511753": "Face Arm Speech Test (Stroke 2003)",
    "33773835": "ERC 2021 guidelines: basic life support",
    "33773828": "ERC 2021 guidelines: first aid",
    "33773825": "ERC 2021 guidelines: adult advanced life support",
    "33204386": "WAO anaphylaxis guidance 2020",
    "34343358": "EAACI guidelines: anaphylaxis (2021)",
    "27062097": "ESCMID guideline: acute bacterial meningitis",
    "22556195": "AHA/ASA 2012: aneurysmal subarachnoid haemorrhage",
    "23432764": "Scandinavian guidelines: mild/moderate head injury in adults",
    "29562304": "ESC 2018 guidelines: syncope",
    "39052901": "Consensus report: hyperglycaemic crises in adults with diabetes",
    "26900382": "AES guideline: convulsive status epilepticus",
    "34281996": "AAP guideline: febrile infants 8 to 60 days",
}

_TOPIC = {"bone": "bone health, osteoporosis, osteopenia, bone density"}
_INDEX = None


def _label(url):
    if not url:
        return "Asklepios safety policy"
    tail = url.rstrip("/").rsplit("/", 1)[-1]
    if "nhs.uk" in url:
        return "NHS: " + tail.replace("-", " ")
    if "medlineplus.gov" in url:
        return "MedlinePlus (US National Library of Medicine)"
    return _TITLES.get(tail, f"PubMed {tail}")


def chunks():
    out = []
    for r in triage_rules.RULES:
        out.append(kb.Chunk(f"rule:{r.id}", r.en, _label(r.source), r.source, (), False))
    reviewed = bool(lifestyle_info.REVIEWED_BY)
    for topic, items in lifestyle_info.ITEMS.items():
        for it in items:
            srcs = [s for s in it["sources"] if s in lifestyle_info.SOURCES]
            if not srcs or not it["en"]:
                continue
            s = lifestyle_info.SOURCES[srcs[0]]
            out.append(kb.Chunk(f"life:{topic}:{it['id']}", it["en"], f"General information on {_TOPIC.get(topic, topic)}: " + s["cite"].split(".")[0][:90],
                                f"https://pubmed.ncbi.nlm.nih.gov/{s['pmid']}/", (), reviewed))
    return out


def index():
    global _INDEX
    if _INDEX is None:
        _INDEX = kb.Index(chunks())
    return _INDEX


def enabled():
    return os.environ.get("KB_RAG", "on").strip().lower() not in ("off", "0", "false", "no")


def retrieve(texts, alert_ids=(), limit=4):
    """-> (prompt_block, found) for what the user wrote. Empty when nothing is relevant or KB_RAG=off."""
    texts = [t for t in (texts or []) if t]
    if not enabled() or not texts:
        return "", []
    found = index().search(texts, k=limit, min_cover=0.25, exclude={f"rule:{i}" for i in alert_ids})
    if not found:
        return "", []
    return kb.prompt_block(found), found


def cite(reply, found, lang="en"):
    return kb.cite(reply, found, lang)


def ids(found):
    return ",".join(f.chunk.id for f in found)[:200]
