"""Personal Advisor: one chat that routes to the right project agent.

Agents wired in this version:
  asklepios  - AI nurse. Kira's triage prompt + Kira's deterministic safety floor (triage_rules.py).
  nomikos    - Greek legal-information agent. Calls the live Nomikosarogos /api/analyze (RAG over the Codes).
pets, ev, money, astro live in agents_tools.py (pets uses the vet safety floor; the others call the live project apps).

Safety: every user message goes through the Asklepios red-flag rules FIRST, whichever agent is active.
If an act-now rule fires, the fixed emergency message is returned and no AI is called.
"""
import json
import os
import re

import anthropic
import httpx
import jwt
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import agents_tools as at
import triage_rules
import vet_rules

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")
SUPABASE_JWT_SECRET = os.environ.get("SUPABASE_JWT_SECRET", "")
NOMIKOS_API_URL = os.environ.get("NOMIKOS_API_URL", "https://nomikosarogos-production.up.railway.app").rstrip("/")
MODEL = os.environ.get("ADVISOR_MODEL", "claude-sonnet-4-6")

AGENTS = {
    "asklepios": {"name": "Asklepios", "icon": "🩺", "live": True},
    "nomikos": {"name": "Nomikosarogos", "icon": "⚖️", "live": True},
    "pets": {"name": "PetsAIHealth", "icon": "🐾", "live": True, "url": "https://app.petsaihealth.com"},
    "ev": {"name": "EV Charger", "icon": "🔌", "live": True, "url": "https://evchargers.up.railway.app"},
    "money": {"name": "Smart Money", "icon": "💰", "live": True, "url": "https://smartmoneywatcher.up.railway.app"},
    "astro": {"name": "Astro Forecast", "icon": "🔮", "live": True, "url": "https://astro-forecast.up.railway.app"},
}

ROUTER_SYSTEM = """You route a user's message to one specialist agent. Reply with ONLY one word from this list:
asklepios  - the user's own health, symptoms, medicines, feeling unwell, mental wellbeing
nomikos    - legal issues, law, contracts, courts, police, tenants/landlords, fines, inheritance, divorce (Greek law)
pets       - a pet's health or care (dog, cat, etc.)
ev         - electric car charging, charging stations, EV trips
money      - money, investing, markets, savings, budgeting
astro      - astrology, horoscope, birth chart, forecasts, tarot
general    - anything else, greetings, or unclear
Messages may be in English or Greek. If the conversation is already with an agent and the newest message is a
follow-up on the same topic, answer with that same agent."""

ASKLEPIOS_SYSTEM = """You are Asklepios, an AI nurse for users in Greece. Clinically accurate, direct, supportive.
Role: symptom triage (ask ONE question at a time), vitals interpretation, medications, the Greek health system (EOPYY, EODY, EOF).
Reply in the language the user writes in (Greek or English).
Rules: always recommend a professional. Red flags -> call 166 (EKAB) or 112. You do not diagnose.
Triage - NON-EMERGENCY rule: recommend seeing a doctor (not self-care) when ANY of these apply:
  - A prescription medication is needed (antibiotic, steroid, antifungal, etc.)
  - Imaging is needed to rule out fracture or serious injury
  - The diagnosis needs professional confirmation before treatment
  - There is a risk of deterioration without monitoring
  - Symptoms persist more than 48-72 hours without improvement
Keep answers short and easy to read on a phone. This is a plain text chat: never mention maps, buttons, body diagrams
or any screen element that does not exist here; when you need a location, just ask the user to describe it in words.
Users can attach a PDF or photo with the paperclip. A full clinical report lives in the full Asklepios app
(https://asklepiosainurse.up.railway.app)."""

app = FastAPI(title="Personal Advisor")
_claude = None
_jwks = None


def claude():
    global _claude
    if _claude is None:
        _claude = anthropic.Anthropic()
    return _claude


def _verify(token):
    """Verify a Supabase access token (asymmetric via JWKS first, legacy HS256 as fallback)."""
    global _jwks
    if not SUPABASE_URL:
        raise HTTPException(500, "Server is missing SUPABASE_URL")
    try:
        if _jwks is None:
            _jwks = jwt.PyJWKClient(f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json")
        key = _jwks.get_signing_key_from_jwt(token).key
        return jwt.decode(token, key, algorithms=["ES256", "RS256"], audience="authenticated")
    except Exception:
        if SUPABASE_JWT_SECRET:
            return jwt.decode(token, SUPABASE_JWT_SECRET, algorithms=["HS256"], audience="authenticated")
        raise


def require_user(authorization: str = Header(default="")):
    token = authorization[7:].strip() if authorization.startswith("Bearer ") else ""
    if not token:
        raise HTTPException(401, "Authentication required")
    try:
        payload = _verify(token)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(401, "Invalid or expired session")
    if not payload.get("sub"):
        raise HTTPException(401, "Invalid session")
    return {"id": payload["sub"], "token": token}


class Msg(BaseModel):
    role: str
    content: str


class Doc(BaseModel):
    data: str  # base64
    mediaType: str
    name: str = ""


class ChatIn(BaseModel):
    messages: list[Msg]
    agent: str | None = None  # agent currently in the conversation (sticky)
    lang: str | None = None
    document: Doc | None = None  # optional attachment on the newest user message


DOC_TYPES = {"application/pdf", "image/jpeg", "image/png", "image/gif", "image/webp"}
MAX_DOC_B64 = 14_000_000  # about 10 MB of file


def clean(messages):
    out = [{"role": m.role, "content": m.content[:8000]} for m in messages if m.role in ("user", "assistant") and m.content.strip()]
    return out[-12:]


def detect_lang(text):
    return "el" if re.search(r"[Ͱ-Ͽ]", text or "") else "en"


def route(history, current, doc=None):
    ctx = f"Current agent: {current or 'none'}\n\nConversation:\n" + "\n".join(
        f"{m['role'].upper()}: {m['content'][:400]}" for m in history[-4:])
    if doc:
        ctx += (f"\n\nThe user attached a file named '{doc.name}' ({doc.mediaType}). Legal papers (contracts, court "
                "documents, notices) go to nomikos; lab results, prescriptions or medical documents go to asklepios.")
    try:
        r = claude().messages.create(model=MODEL, max_tokens=8, system=ROUTER_SYSTEM, messages=[{"role": "user", "content": ctx}])
        word = "".join(b.text for b in r.content if b.type == "text").strip().lower()
        word = re.sub(r"[^a-z]", "", word)
    except Exception:
        word = ""
    if word in AGENTS:
        return word
    if word == "general":
        return current if current in AGENTS else "general"
    return current if current in AGENTS else "general"


def run_asklepios(history, lang, res, doc=None):
    system = ASKLEPIOS_SYSTEM
    block = triage_rules.alert_block(res)
    if block:
        system += "\n\n" + block
    messages = history
    if doc:
        kind = "document" if doc.mediaType == "application/pdf" else "image"
        part = {"type": kind, "source": {"type": "base64", "media_type": doc.mediaType, "data": doc.data}}
        last = history[-1]["content"]
        messages = history[:-1] + [{"role": "user", "content": [part, {"type": "text", "text": last}]}]
        system += ("\n\nThe user attached a file. If it is lab results, a prescription or a medical report, interpret it "
                   "clearly and cautiously: highlight values outside the reference range, explain what they may mean in plain "
                   "language, and say what to ask a doctor. Never diagnose. If it is a photo of a symptom, describe what you "
                   "see and give the same cautious triage advice.")
    try:
        r = claude().messages.create(model=MODEL, max_tokens=1200 if doc else 900, system=system, messages=messages)
        text = "".join(b.text for b in r.content if b.type == "text").strip()
        text = text.replace("[[READY]]", "").strip()
        return text or triage_rules.fallback_reply(res, lang)
    except Exception:
        return triage_rules.fallback_reply(res, lang)


def run_nomikos(history, lang, token, doc=None):
    payload = {"messages": history, "area": "unsure", "lang": lang, "mode": "quick"}
    if doc:
        payload["document"] = {"data": doc.data, "mediaType": doc.mediaType}
    try:
        r = httpx.post(
            f"{NOMIKOS_API_URL}/api/analyze",
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
            timeout=120,
        )
        r.raise_for_status()
        d = r.json()
    except Exception:
        return None
    text = d.get("analysis") or ""
    if d.get("sources"):
        text += "\n\n" + ("Πηγές: " if lang == "el" else "Sources: ") + "; ".join(d["sources"][:8])
    if d.get("disclaimer"):
        text += "\n\n_" + d["disclaimer"] + "_"
    return text


def link_reply(key, lang):
    a = AGENTS[key]
    if lang == "el":
        return f"{a['icon']} Αυτό ταιριάζει στον **{a['name']}**, που δεν είναι ακόμα μέσα σε αυτή τη συνομιλία. Άνοιξέ τον εδώ: {a['url']}"
    return f"{a['icon']} That's a job for **{a['name']}**, which isn't inside this chat yet. Open it here: {a['url']}"


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/api/config")
def config():
    return {"supabaseUrl": SUPABASE_URL, "supabaseAnonKey": SUPABASE_ANON_KEY, "agents": AGENTS}


@app.post("/api/chat")
def chat(body: ChatIn, user=Depends(require_user)):
    history = clean(body.messages)
    if not history or history[-1]["role"] != "user":
        raise HTTPException(400, "Provide a user message.")
    lang = body.lang if body.lang in ("el", "en") else detect_lang(history[-1]["content"])
    doc = body.document
    if doc:
        if doc.mediaType not in DOC_TYPES:
            raise HTTPException(400, "Unsupported file type. Use PDF, JPG, PNG, GIF or WEBP.")
        if len(doc.data) > MAX_DOC_B64:
            raise HTTPException(413, "File is too large (max about 10 MB).")

    # Safety floors first, on every user turn, whatever agent is active.
    user_texts = [m["content"] for m in history if m["role"] == "user"]
    species = at.detect_species(user_texts)
    pet_ctx = species is not None or body.agent == "pets"
    res = triage_rules.evaluate(user_texts)
    pet_res = at.pets_floor(species, user_texts) if pet_ctx else None
    if pet_res and vet_rules.act_now_hits(pet_res):
        reply = vet_rules.act_now_reply(vet_rules.act_now_hits(pet_res), lang)
        if res.hits and triage_rules.act_now_hits(res):
            reply += "\n\n" + ("Αν εσύ ο ίδιος νιώθεις άσχημα: κάλεσε 166 ή 112." if lang == "el" else "If you yourself feel unwell: call 166 or 112.")
        return {"agent": "pets", "reply": reply, "emergency": True}
    if res.hits and triage_rules.act_now_hits(res) and not pet_ctx:
        return {"agent": "asklepios", "reply": triage_rules.act_now_reply(res, lang), "emergency": True}

    agent = route(history, body.agent, doc)
    if agent == "asklepios":
        if res.hits and triage_rules.act_now_hits(res):  # first-person red flag even in a pet-related chat
            return {"agent": "asklepios", "reply": triage_rules.act_now_reply(res, lang), "emergency": True}
        reply = run_asklepios(history, lang, res, doc)
    elif agent == "nomikos":
        reply = run_nomikos(history, lang, user["token"], doc)
        if reply is None:
            reply = ("⚠️ Ο Νομικός Αρωγός δεν απαντά αυτή τη στιγμή. Δοκίμασε ξανά σε λίγο." if lang == "el"
                     else "⚠️ The legal agent isn't responding right now. Please try again shortly.")
    elif agent in ("pets", "ev", "money", "astro"):
        try:
            if agent == "pets":
                reply = at.run_pets(history, lang, pet_res or at.pets_floor(species, user_texts), doc)
            elif doc:
                reply = ("Αυτός ο βοηθός δεν διαβάζει αρχεία ακόμα. Γράψε μου την ερώτησή σου με λόγια." if lang == "el"
                         else "This agent can't read attachments yet. Please describe your question in words.")
            elif agent == "ev":
                reply = at.run_ev(history, lang)
            elif agent == "money":
                reply = at.run_money(history, lang)
            else:
                reply = at.run_astro(history, lang)
        except Exception:
            reply = ("⚠️ Ο βοηθός δεν απαντά αυτή τη στιγμή. Δοκίμασε ξανά σε λίγο." if lang == "el"
                     else "⚠️ That agent isn't responding right now. Please try again shortly.")
            if agent == "pets":
                reply += ("\n\nΑν το κατοικίδιό σου έχει δύσκολη αναπνοή, κατάρρευση ή δηλητηρίαση, πήγαινε αμέσως σε κτηνίατρο."
                          if lang == "el" else
                          "\n\nIf your pet has trouble breathing, has collapsed or may have eaten a toxin, go to a vet now.")
    elif agent in AGENTS:
        reply = link_reply(agent, lang)
    else:
        reply = ("Πες μου τι συμβαίνει και θα σε πάω στον κατάλληλο βοηθό: υγεία, νομικό θέμα, κατοικίδιο, φόρτιση EV, χρήματα ή αστρολογία."
                 if lang == "el" else
                 "Tell me what's going on and I'll bring in the right agent: health, a legal issue, a pet, EV charging, money or astrology.")
    return {"agent": agent, "reply": reply, "emergency": False}


@app.get("/")
def index():
    return FileResponse("public/index.html")


app.mount("/static", StaticFiles(directory="public"), name="static")
