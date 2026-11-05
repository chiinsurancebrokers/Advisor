"""kb.py: a small retrieval layer (RAG) shared by PetsAIHealth, Asklepios and the Advisor.

What it does
  1. A corpus of short, paraphrased, SOURCED chunks (each has a source URL and a label).
  2. Retrieval: accent-folded BM25 (Greek + English) with a small Greek-to-English symptom glossary, so a Greek
     message finds English guidance and the other way round. Pure Python, no network, no keys: the same query
     always returns the same chunks, and it is unit-testable.
  3. Grounding: the caller puts the retrieved chunks in the system prompt tagged [S1], [S2]...; the model cites the
     tag after a sentence that relies on it; `cite()` then removes the tags and appends a "Sources" line that lists
     ONLY the chunks the model actually cited.

What it is not: a diagnosis engine. Chunks hold owner-level guidance; the deterministic safety rules stay the floor.
No dependencies beyond the standard library. Copied unchanged into every repo that uses it (like triage_rules.py).
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    title: str            # short label shown to the user, e.g. "MSD Veterinary Manual: food hazards"
    source: str = ""      # URL ("" for internal policy text: shown by title only, never as a link)
    species: tuple = ()   # () = every species
    reviewed: bool = False  # True only after a vet / clinician signed the text off


@dataclass
class Found:
    chunk: Chunk
    score: float
    terms: int            # distinct query stems that matched
    cover: float = 0.0    # share of the query's (idf-weighted) vocabulary that this chunk matched, 0..1


# --------------------------------------------------------------------------------------------- text handling
def fold(text):
    """Lower-case, strip accents, final sigma -> sigma. Greek and English both come out plain."""
    t = unicodedata.normalize("NFD", text or "")
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    return t.replace("ς", "σ")


_STOP = set("""
the and for with that this from have has had was were are not but you your his her its they them what when where which who how
into onto over under than then also very just can could should would will been being does did done any all some more most
our out off per via about after before again once only own same such too s t a an as at be by do he if in is it me my no of
on or so to up us we
και να το τα του της των τον την στο στη στην στα στον με σε απο για που πως ειναι εχει εχουν ηταν θα δεν μου σου μας σας
ο η οι ενα μια ενας αυτο αυτα αυτος αυτη αυτοι επισης οπως αλλα ομως ειτε ωστε μετα πριν
""".split())

_GREEK = re.compile(r"[α-ω]")


def stem(tok):
    """Cheap stemmer: Greek = first 5 letters; English = drop common suffixes, then first 7 letters."""
    if tok.isdigit():
        return tok
    if _GREEK.search(tok):
        return tok[:5]
    tok = tok.replace("oe", "e")
    for suf in ("ing", "ied", "ies", "ed", "es", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 4:
            tok = tok[: -len(suf)]
            break
    return tok[:7]


def tokens(text):
    out = []
    for raw in re.findall(r"[a-zα-ω0-9]+", fold(text)):
        if raw in _STOP or (len(raw) < 3 and not raw.isdigit()):
            continue
        out.append(stem(raw))
    return out


# Greek word-start (folded) -> English terms. Lets a Greek owner message reach English guidance. Kept to symptoms,
# body parts, common hazards and species: it is a bridge, not a dictionary.
GLOSSARY = {
    "εμετ": "vomit vomiting retching", "ξεραμ": "retching vomit", "ξερναει": "vomit", "διαρρ": "diarrhea diarrhoea", "κοπρα": "stool diarrhea",
    "αιμα": "blood bleeding", "αιματ": "blood bleeding", "κουρασ": "lethargy tired weak", "λεθαργ": "lethargy", "υπνηλ": "lethargy drowsy",
    "αδυνα": "weak weakness", "φαγητ": "eating appetite food", "ορεξ": "appetite eating", "τρωει": "eating appetite", "νερο": "water drinking thirst",
    "διψα": "thirst drinking", "ουρα": "urine urinate urination", "κατουρ": "urine urinate urination", "κατουρη": "urine urinate", "αδυνατ": "weight loss thin", "χασει": "loss", "ουρη": "urine urinate", "κοιλι": "abdomen belly stomach", "στομαχ": "stomach",
    "πονα": "pain painful", "πονο": "pain painful", "πυρετ": "fever temperature", "βηχα": "cough coughing", "βηχε": "cough", "αναπνο": "breathing breath",
    "δυσπν": "breathing breath difficulty", "λαχαν": "panting breathing", "σπασμ": "seizure convulsion", "κουτσ": "limp lameness", "χωλ": "limp lameness",
    "ματι": "eye", "μματι": "eye", "αυτι": "ear", "δερμα": "skin", "φαγουρ": "itch itching scratching", "ξυνεται": "itch scratching",
    "τριχοπτ": "hair loss", "τρομο": "tremor shaking", "τρεμ": "tremor shaking", "δηλητηρ": "poison poisoning toxic", "τοξικ": "toxic poisoning",
    "σοκολατ": "chocolate", "σταφυλ": "grapes", "σταφιδ": "raisins", "κρινο": "lily", "φαρμακ": "medication drug medicine", "χαπι": "pill medication",
    "εμβολ": "vaccine vaccination", "ψυλλ": "flea", "κροτων": "tick", "σκουληκ": "worm parasite", "γαστρεντ": "gastroenteritis vomit diarrhea",
    "αφυδατ": "dehydration", "πρησ": "swollen swelling", "φουσκ": "bloat swollen belly", "σκληρ": "hard firm", "χλωμ": "pale", "ουλα": "gums",
    "γλωσσ": "tongue", "σαλια": "drooling saliva", "βηχ": "cough", "φτερν": "sneeze", "μυτη": "nose", "τραυμ": "wound injury", "πληγη": "wound",
    "δαγκω": "bite", "δαγκα": "bite", "φιδι": "snake snakebite", "οχια": "snake viper", "καψιμ": "burn", "πνιγ": "choking choke", "πνιξ": "choking choke",
    "λιποθυμ": "faint collapse", "καταρρ": "collapse", "ζαλαδ": "dizziness", "πονοκεφ": "headache", "εξανθ": "rash", "στηθο": "chest", "καρδι": "heart",
    "εγκυμ": "pregnant pregnancy", "γεννα": "labour birth", "κουταβ": "puppy", "γατακ": "kitten", "σκυλ": "dog", "γατα": "cat", "γατο": "cat",
    "κουνελ": "rabbit", "παπαγαλ": "parrot bird", "πουλι": "bird", "χελων": "tortoise turtle", "χαμστερ": "hamster", "ινδικο": "guinea pig",
    "καναρ": "canary bird", "ερπετ": "reptile", "νεφρ": "kidney", "ηπαρ": "liver", "ουροδ": "urinary bladder", "κυστη": "bladder",
    "ιδρω": "sweat", "ναυτι": "nausea", "ζαχαρ": "sugar", "διαβητ": "diabetes", "ξυλιτολ": "xylitol", "τσιχλ": "gum xylitol", "αντιψυκ": "antifreeze",
    "ποντικ": "rat poison rodenticide", "ραιτ": "rat poison", "δολωμ": "bait poison", "καθαριστ": "cleaner bleach", "χλωριν": "bleach",
    "μπαταρ": "battery", "οστο": "bone", "καταγμ": "fracture broken", "σπασιμ": "fracture broken", "ραχη": "spine back", "πλατη": "back",
}


# Everyday English wording -> the term the guidance uses.
SYNONYMS = {
    "hurt": "pain", "hurts": "pain", "hurting": "pain", "ache": "pain", "aches": "pain", "aching": "pain", "sore": "pain",
    "puke": "vomit", "puking": "vomit", "threw": "vomit", "throwing": "vomit", "sick": "vomit",
    "poop": "stool", "poo": "stool", "pee": "urine", "peeing": "urine", "wee": "urine", "tummy": "abdomen belly",
    "breathless": "breathing", "wheezing": "breathing", "limping": "limp lameness", "scratching": "itch",
}


def expand(text):
    """Query tokens plus the English terms the glossary and synonym table map the words to."""
    base = tokens(text)
    extra = []
    for raw in re.findall(r"[a-z]+", fold(text)):
        if raw in SYNONYMS:
            extra += tokens(SYNONYMS[raw])
    for raw in re.findall(r"[α-ω]+", fold(text)):
        for pre, en in GLOSSARY.items():
            if raw.startswith(pre):
                extra += tokens(en)
    return base + extra


# --------------------------------------------------------------------------------------------- the index
class Index:
    """BM25 over chunks. Build once (cheap), search many times."""

    def __init__(self, chunks, k1=1.4, b=0.75):
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self._tf = []
        self._len = []
        df = {}
        for c in self.chunks:
            toks = tokens(f"{c.title} {c.text}")
            tf = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            self._tf.append(tf)
            self._len.append(len(toks) or 1)
            for t in tf:
                df[t] = df.get(t, 0) + 1
        n = max(len(self.chunks), 1)
        self._avg = sum(self._len) / n if self.chunks else 1
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, texts, species=None, k=4, min_score=3.0, min_terms=2, min_cover=0.0, exclude=()):
        """Best chunks for what the user wrote (a string or a list of strings). Deterministic.

        min_terms: at least this many DISTINCT query stems must match, so one common word never retrieves a chunk.
        """
        q = expand(" \n ".join(t for t in ([texts] if isinstance(texts, str) else texts) if t))
        if not q:
            return []
        qset = list(dict.fromkeys(q))
        mass = sum(self._idf.get(t, 0.0) for t in qset) or 1.0   # terms the corpus has never seen do not count against a chunk
        out = []
        for i, c in enumerate(self.chunks):
            if c.id in exclude or (c.species and species and species not in c.species):
                continue
            tf, ln = self._tf[i], self._len[i]
            score, hit, got = 0.0, 0, 0.0
            for t in qset:
                f = tf.get(t)
                if not f:
                    continue
                hit += 1
                got += self._idf.get(t, 0.0)
                score += self._idf.get(t, 0.0) * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * ln / self._avg))
            if hit >= min(min_terms, len(qset)) and score >= min_score and got / mass >= min_cover:
                out.append(Found(c, round(score, 3), hit, round(got / mass, 3)))
        out.sort(key=lambda f: (-f.score, f.chunk.id))
        return out[:k]


# --------------------------------------------------------------------------------------------- grounding
GROUNDING_EN = (
    "RETRIEVED REFERENCE FACTS (curated and paraphrased; each has a tag). Use them to ask better questions and to give safe, "
    "specific wording. They are not a diagnosis. Rules: (1) state a threshold, dose, drug, time window or number ONLY if it is in a "
    "fact below or in the SAFETY alerts; otherwise say you cannot verify it and send the owner to the professional. (2) When a sentence "
    "relies on a fact, put its tag right after it, like [S2]; never invent a tag; do not cite for general questions. (3) Facts marked "
    "(pending clinician review) are internal guidance: use them cautiously and never to lower urgency."
)


def prompt_block(found, header=GROUNDING_EN):
    """System-prompt text with the chunks tagged S1..Sn. Empty string when nothing was retrieved."""
    if not found:
        return ""
    lines = [header]
    for i, f in enumerate(found, 1):
        flag = "" if f.chunk.reviewed else " (pending clinician review)"
        lines.append(f"[S{i}] {f.chunk.title}{flag}: {f.chunk.text}")
    return "\n".join(lines)


_TAG = re.compile(r"[ \t]*\[S(\d+)\]")


def cite(reply, found, lang="en"):
    """Remove [S#] tags from the reply and append a Sources line for the chunks that were really cited."""
    if not reply:
        return reply
    used = []
    for m in _TAG.finditer(reply):
        n = int(m.group(1))
        if 1 <= n <= len(found) and n not in used:
            used.append(n)
    clean = _TAG.sub("", reply)
    if not used:
        return clean
    parts, seen = [], set()
    for n in used:
        c = found[n - 1].chunk
        label = f"{c.title} ({c.source})" if c.source else c.title
        if label not in seen:
            seen.add(label)
            parts.append(label)
    head = "Πηγές" if lang == "el" else "Sources"
    return clean.rstrip() + f"\n\n_{head}: " + "; ".join(parts) + "_"


def title_from_url(url, prefix="MSD Veterinary Manual"):
    """'.../special-pet-topics/poisoning/food-hazards' -> 'MSD Veterinary Manual: food hazards'."""
    if not url:
        return prefix
    tail = [p for p in url.split("?")[0].rstrip("/").split("/") if p][-1]
    return f"{prefix}: {tail.replace('-', ' ')}"
