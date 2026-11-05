"""Tool-using agents for the Advisor: PetsAIHealth, EV Charger, Smart Money, Astro Forecast.

Each agent is Claude plus a short system prompt. EV, Smart Money and Astro call the live project apps through
their public HTTP endpoints, so the numbers come from the real engines, not from the model's memory.
"""
import base64
import io
import json
import math
import os
import re
import time

import anthropic
import httpx

import kb_vet
import vet_rules

MODEL = os.environ.get("ADVISOR_MODEL", "claude-sonnet-4-6")
EV_API = os.environ.get("EV_API_URL", "https://evchargers.up.railway.app").rstrip("/")
ASTRO_API = os.environ.get("ASTRO_API_URL", "https://astro-forecast.up.railway.app").rstrip("/")
SM_API = os.environ.get("SMARTMONEY_API_URL", "https://smartmoneywatcher.up.railway.app").rstrip("/")
SM_AUTH = (os.environ["SMARTMONEY_USER"], os.environ.get("SMARTMONEY_PASSWORD", "")) if os.environ.get("SMARTMONEY_USER") else None

_client = None


def client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def _text(r):
    return "".join(b.text for b in r.content if b.type == "text").strip()


DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
TEXT_TYPES = {"text/plain", "text/csv", "text/markdown", DOCX}


def extract_text(doc):
    """Text of a .txt/.csv/.md/.docx attachment, capped so one file cannot flood the context."""
    raw = base64.b64decode(doc.data)
    if doc.mediaType == DOCX:
        from docx import Document
        d = Document(io.BytesIO(raw))
        parts = [p.text for p in d.paragraphs if p.text.strip()]
        for t in d.tables:
            for row in t.rows:
                parts.append(" | ".join(c.text.strip() for c in row.cells))
        text = "\n".join(parts)
    else:
        text = raw.decode("utf-8", errors="replace")
    return text[:60000]


def doc_blocks(docs):
    """Claude content blocks for attachments: PDFs and images natively, text-like files inlined."""
    blocks = []
    for d in docs or []:
        if d.mediaType == "application/pdf":
            blocks.append({"type": "document", "source": {"type": "base64", "media_type": d.mediaType, "data": d.data}})
        elif d.mediaType.startswith("image/"):
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": d.mediaType, "data": d.data}})
        else:
            blocks.append({"type": "text", "text": f"[Attached file: {d.name}]\n{extract_text(d)}"})
    return blocks


def run_tools(system, history, tools, handlers, max_tokens=1100, max_iters=5):
    """Claude tool loop. Tool errors are returned to the model as text so it can explain them."""
    messages = [dict(m) for m in history]
    last_text = ""
    for _ in range(max_iters):
        r = client().messages.create(model=MODEL, max_tokens=max_tokens, system=system, tools=tools, messages=messages)
        last_text = _text(r) or last_text
        if r.stop_reason != "tool_use":
            return last_text
        messages.append({"role": "assistant", "content": r.content})
        results = []
        for b in r.content:
            if b.type != "tool_use":
                continue
            try:
                out = handlers[b.name](**b.input)
                out = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
            except Exception as e:  # noqa: BLE001
                out = f"Tool error: {str(e)[:300]}"
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": out[:7000]})
        messages.append({"role": "user", "content": results})
    return last_text


# ───────────────────────────── PetsAIHealth ─────────────────────────────
PETS_SYSTEM = """You are PetsAIHealth, an AI veterinary nurse for pets in Greece. Clinically accurate, direct, supportive.
Reply in the language the owner writes in (Greek or English). Keep answers short and easy to read on a phone.

Role: symptom triage for the owner's pet (ask ONE short question at a time), toxin warnings, home-care advice and when to see a vet.
Always recommend a vet for diagnosis and treatment. You do not diagnose or prescribe.

Three triage levels. Choose from the facts, not from fear:
- EMERGENCY: life-threatening within hours (laboured or open-mouth breathing, blue or very pale gums, collapse, seizure over 5 minutes
  or repeated, bloated hard abdomen with unproductive retching, cannot urinate despite straining, a toxin eaten, heavy bleeding).
  Say "🚨 GO TO A VET CLINIC IMMEDIATELY", then one short sentence of why, and what to do on the way (no food or water).
- URGENT: a vet should see the pet within 24 hours but it is not "rush now" (vomiting or diarrhoea over 24 h, limping with pain, red eye,
  not eating over 24 h, new lump, suspected urinary infection while urine is passing).
- SELF_CARE: mild, stable, bright pet that eats and drinks. Give home-monitoring advice and say when to contact a vet.
Before a final URGENT or SELF_CARE, make sure you know: species and age, how long it has lasted, appetite and water, vomiting, diarrhoea or
urination, any toxin or foreign body, and how bright the pet is. Skip what the owner already said. For a clear emergency, do not wait.
This is a plain text chat: never mention buttons, tools or screens that do not exist here. Owners can attach a photo or PDF with the paperclip.
Reminders, a vet finder, insurance and full reports live in the PetsAIHealth app (https://app.petsaihealth.com).

OPENING: never start with a greeting or a welcome ("Hello", "Glad you came", "Χαίρομαι που...", "Happy to help"). The owner is worried about a sick pet:
open with ONE short clause of concern that names the pet (e.g. "Λυπάμαι που ο Σπίθας δεν είναι καλά." / "Sorry Rex is unwell."), then go straight to your
advice or your ONE question. No cheerful words and no emoji in the first line.
GROUNDING: a block "RETRIEVED REFERENCE FACTS" may follow. Base thresholds, drugs, time windows and numbers only on those facts or the SAFETY RULE ALERTS;
if something is not covered, say you cannot verify it and send the owner to the vet. Put the fact's tag, like [S2], right after the sentence that relies on it.
Name what the findings fit as "possible" causes (most likely first), never as a diagnosis."""

_SPECIES_PATTERNS = [
    ("dog", r"\bdogs?\b|\bpupp(y|ies)\b|\bpooch\b|σκυλ|κουταβ"),
    ("cat", r"\bcats?\b|\bkitten|\bkitty\b|γατ"),
    ("rabbit", r"\brabbits?\b|\bbunn(y|ies)\b|κουνελ"),
    ("bird", r"\bbirds?\b|\bparrot|\bbudgie|\bcanary\b|\bparakeet|παπαγαλ|καναρ|πουλι|πουλακι|πουλια"),
    ("reptile", r"\btortoise|\bturtle|\bsnake\b|\blizard|\bgecko|\breptile|χελων|φιδι|σαυρ|ερπετ"),
    ("other", r"\bhamster|\bguinea pig|\bferret|\brat\b|\bmouse\b|χαμστερ|ινδικο χοιρ"),
]


def detect_species(user_texts):
    """Return a vet_rules species key, or None when the text does not mention a pet."""
    text = vet_rules.norm(" ".join(user_texts or []))
    for key, pat in _SPECIES_PATTERNS:
        if re.search(vet_rules.norm(pat), text):
            return key
    return None


def pets_floor(species, user_texts):
    return vet_rules.evaluate(species or "other", user_texts)


def run_pets(history, lang, res, docs=None, species=None):
    system = PETS_SYSTEM
    hints = vet_rules.prompt_hints(res)
    if hints:
        system += "\n\n" + hints
    found = []
    try:  # retrieval over the curated vet guidance (kb_vet); never blocks the answer
        block, found = kb_vet.retrieve(species, [m["content"] for m in history if m["role"] == "user"], [h.id for h in res.hits])
        if block:
            system += "\n\n" + block
            print(json.dumps({"event": "kb_retrieve", "agent": "pets", "ids": kb_vet.ids(found)}), flush=True)
    except Exception:
        found = []
    messages = history
    if docs:
        messages = history[:-1] + [{"role": "user", "content": doc_blocks(docs) + [{"type": "text", "text": history[-1]["content"]}]}]
        system += ("\n\nThe owner attached one or more files. If one is a photo of a symptom, describe what you see and give cautious triage advice. "
                   "If one is lab results, a vet report or vaccination or medical records, explain it in plain language, highlight values outside "
                   "the range, note dates that matter (due vaccines, follow-ups) and say what to ask the vet. Name which file each point comes from.")
    r = client().messages.create(model=MODEL, max_tokens=1400 if docs else 900, system=system, messages=messages)
    text = _text(r)
    text = re.sub(r"\[(TRIAGE|TOOL):[^\]]*\]", "", text).strip()
    return kb_vet.cite(text, found, lang)


# ───────────────────────────── EV Charger ─────────────────────────────
EV_SYSTEM = """You are the EV Charger assistant for drivers of electric cars in Greece. You answer from the live public charger network
(Greek national registry, with real-time availability) using your tools. Reply in the user's language (Greek or English), short and
phone-friendly. Always call a tool before naming chargers; never invent stations, power or availability.
- For "where can I charge near X" use find_chargers_near. For a trip use plan_route_chargers.
- Power tiers: under 50 kW is slow/AC, 50 kW and above is fast DC, 120 kW and above is ultra-fast.
- Availability is a snapshot refreshed about every 10 minutes: say so, and suggest a backup stop on long trips.
- Give name, place, kW, how many are free now and the distance. Maximum 5 results unless asked for more.
- If the user does not say their car, assume 50 kW or more is good; ask the car model only if range matters.
For the interactive map open https://evchargers.up.railway.app"""

EV_TOOLS = [
    {"name": "find_chargers_near", "description": "Find public EV chargers near a place in Greece, nearest first.",
     "input_schema": {"type": "object", "properties": {
         "place": {"type": "string", "description": "Town, address or landmark, e.g. 'Ναύπλιο' or 'Syntagma, Athens'"},
         "radius_km": {"type": "number", "description": "Search radius, default 10"},
         "min_kw": {"type": "number", "description": "Minimum power in kW, default 0"},
         "available_only": {"type": "boolean", "description": "Only stations with a free connector now"},
         "limit": {"type": "integer", "description": "Max results, default 5"}}, "required": ["place"]}},
    {"name": "plan_route_chargers", "description": "List fast chargers along the driving route between two places in Greece, in travel order.",
     "input_schema": {"type": "object", "properties": {
         "from_place": {"type": "string"}, "to_place": {"type": "string"},
         "min_kw": {"type": "number", "description": "Minimum power in kW, default 50"},
         "corridor_km": {"type": "number", "description": "Max distance from the route, default 3"},
         "available_only": {"type": "boolean"}}, "required": ["from_place", "to_place"]}},
]

_ev_cache = {"t": 0, "data": []}


def _ev_chargers():
    if time.time() - _ev_cache["t"] > 120 or not _ev_cache["data"]:
        r = httpx.get(f"{EV_API}/api/chargers", timeout=40)
        r.raise_for_status()
        _ev_cache["data"] = r.json().get("chargers", [])
        _ev_cache["t"] = time.time()
    return _ev_cache["data"]


def _geocode(place):
    r = httpx.get(f"{EV_API}/api/geocode", params={"q": place + (", Ελλάδα" if "ελλ" not in place.lower() and "greece" not in place.lower() else "")}, timeout=20)
    if r.status_code == 404:
        raise ValueError(f"Could not find '{place}'.")
    r.raise_for_status()
    d = r.json()
    return float(d["lat"]), float(d.get("lng", d.get("lon"))), d.get("label", place)


def _km(lat1, lng1, lat2, lng2):
    p = math.pi / 180
    a = 0.5 - math.cos((lat2 - lat1) * p) / 2 + math.cos(lat1 * p) * math.cos(lat2 * p) * (1 - math.cos((lng2 - lng1) * p)) / 2
    return 12742 * math.asin(math.sqrt(a))


def _free(c):
    return sum(1 for x in c.get("connectors", []) if x.get("status") == "AVAILABLE")


def _brief(c, dist=None):
    out = {"name": c.get("name"), "address": c.get("address"), "city": c.get("city"), "operator": c.get("operator"),
           "max_kw": c.get("maxKW"), "free_connectors_now": _free(c), "connectors_total": len(c.get("connectors", [])),
           "best_free_kw": c.get("availableMaxKW"), "open_24_7": c.get("open247"), "lat": c.get("lat"), "lng": c.get("lng")}
    if dist is not None:
        out["distance_km"] = round(dist, 1)
    return out


def find_chargers_near(place, radius_km=10, min_kw=0, available_only=False, limit=5):
    lat, lng, label = _geocode(place)
    rows = []
    for c in _ev_chargers():
        if (c.get("maxKW") or 0) < min_kw:
            continue
        if available_only and _free(c) == 0:
            continue
        d = _km(lat, lng, c["lat"], c["lng"])
        if d <= radius_km:
            rows.append((d, c))
    rows.sort(key=lambda x: x[0])
    return {"searched_around": label, "found": len(rows), "results": [_brief(c, d) for d, c in rows[: max(1, min(int(limit), 10))]]}


def plan_route_chargers(from_place, to_place, min_kw=50, corridor_km=3, available_only=False):
    r = httpx.get(f"{EV_API}/api/route", params={"from": from_place, "to": to_place}, timeout=40)
    r.raise_for_status()
    rt = r.json()
    g = rt.get("geometry")
    pts = g.get("coordinates") if isinstance(g, dict) else g
    if not pts:
        raise ValueError("No route geometry returned.")
    pts = [(p[1], p[0]) for p in pts]  # GeoJSON is [lng, lat]
    step = max(1, len(pts) // 400)
    samples = pts[::step]
    found = {}
    for c in _ev_chargers():
        if (c.get("maxKW") or 0) < min_kw or (available_only and _free(c) == 0):
            continue
        best_i, best_d = None, corridor_km
        for i, (la, lo) in enumerate(samples):
            if abs(la - c["lat"]) > 0.1 or abs(lo - c["lng"]) > 0.15:
                continue
            d = _km(la, lo, c["lat"], c["lng"])
            if d <= best_d:
                best_i, best_d = i, d
        if best_i is not None:
            found[c["id"]] = (best_i, best_d, c)
    ordered = sorted(found.values(), key=lambda x: x[0])
    n = len(samples) or 1
    out = []
    for i, d, c in ordered[:40]:
        b = _brief(c, d)
        b["approx_km_from_start"] = round(rt.get("distance", 0) / 1000 * i / n, 0)
        b["off_route_km"] = b.pop("distance_km")
        out.append(b)
    return {"route_km": round(rt.get("distance", 0) / 1000), "drive_minutes": round(rt.get("duration", 0) / 60),
            "chargers_along_route": len(ordered), "results_in_travel_order": out}


EV_HANDLERS = {"find_chargers_near": find_chargers_near, "plan_route_chargers": plan_route_chargers}


def run_ev(history, lang):
    return run_tools(EV_SYSTEM, history, EV_TOOLS, EV_HANDLERS)


# ───────────────────────────── Smart Money ─────────────────────────────
SM_SYSTEM = """You are the Smart Money assistant: a market-data and investing-education helper. Reply in the user's language (Greek or English),
short and phone-friendly. For prices, use get_quotes (live delayed quotes: US stocks and ETFs by ticker such as AAPL, SPY, TQQQ).
- You explain what the numbers mean: move size, range, volume, how to read options-flow and smart-money concepts. You do not give personal
  financial advice, price targets or buy/sell instructions. If asked "should I buy X", set out what to check and the main risks, and say the
  decision is theirs.
- The Barchart options-flow scores, big-move scanner and watchlists live in the Smart Money app (https://smartmoneywatcher.up.railway.app);
  they need CSV uploads, so they are not available in this chat.
- Always say when a quote was last updated. Never invent prices."""

SM_TOOLS = [
    {"name": "get_quotes", "description": "Get the latest quotes for up to 8 ticker symbols (US stocks/ETFs).",
     "input_schema": {"type": "object", "properties": {"symbols": {"type": "array", "items": {"type": "string"}, "description": "e.g. ['AAPL','SPY']"}},
                      "required": ["symbols"]}},
    {"name": "get_market_regime", "description": "Get the Smart Money app's current market-regime reading, if available.",
     "input_schema": {"type": "object", "properties": {}}},
]


def get_quotes(symbols):
    r = httpx.post(f"{SM_API}/api/quotes", json={"symbols": symbols[:8]}, auth=SM_AUTH, timeout=40)
    if r.status_code == 401:
        raise ValueError("Smart Money app is private: set SMARTMONEY_USER and SMARTMONEY_PASSWORD on the advisor service.")
    r.raise_for_status()
    return r.json()


def get_market_regime():
    r = httpx.get(f"{SM_API}/api/strategy/market-regime", auth=SM_AUTH, timeout=30)
    if r.status_code == 401:
        raise ValueError("Smart Money app is private: set SMARTMONEY_USER and SMARTMONEY_PASSWORD on the advisor service.")
    r.raise_for_status()
    return r.json()


SM_HANDLERS = {"get_quotes": get_quotes, "get_market_regime": get_market_regime}


def run_money(history, lang):
    return run_tools(SM_SYSTEM, history, SM_TOOLS, SM_HANDLERS)


# ───────────────────────────── Astro Forecast ─────────────────────────────
ASTRO_SYSTEM = """You are the Astro Forecast assistant. You read charts computed by the Astro Forecast engine (sidereal natal chart, Vimshottari dasha
periods and monthly transit activation). Reply in the user's language (Greek or English), warm but short and phone-friendly.
- To read a chart you need the birth date, birth time (to the minute if known), birth place and its UTC offset on that date (Greece is +2 in
  winter, +3 in summer). If something is missing, ask for it, one short question at a time. Work out latitude and longitude of the birth place
  yourself and say which place you used. Then call astro_chart.
- Be honest: monthly scores are "pressure clustering", not predictions. Astrology is a reflective tool, not science. Never make medical, legal,
  financial or life-or-death claims, and never say something will definitely happen.
- Tarot, compatibility, career and the PDF reports live in the Astro app (https://astro-forecast.up.railway.app)."""

ASTRO_TOOLS = [
    {"name": "astro_chart", "description": "Compute natal chart, current dasha and a monthly forecast for a birth moment.",
     "input_schema": {"type": "object", "properties": {
         "year": {"type": "integer"}, "month": {"type": "integer"}, "day": {"type": "integer"},
         "hour": {"type": "integer", "description": "Birth hour, 24h local time"}, "minute": {"type": "integer"},
         "utc_offset": {"type": "number", "description": "Hours from UTC at the birth place and date, e.g. 2 or 3"},
         "latitude": {"type": "number"}, "longitude": {"type": "number"},
         "months": {"type": "integer", "description": "Forecast months ahead, default 12"}},
         "required": ["year", "month", "day", "hour", "minute", "utc_offset", "latitude", "longitude"]}},
]


def _trim(obj, depth=0):
    """Keep tool output small: round floats, cap long lists."""
    if isinstance(obj, float):
        return round(obj, 2)
    if isinstance(obj, dict):
        return {k: _trim(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_trim(v, depth + 1) for v in obj[:24]]
    return obj


def astro_chart(year, month, day, hour, minute, utc_offset, latitude, longitude, months=12):
    q = {"year": year, "month": month, "day": day, "hour": hour, "minute": minute,
         "utc_offset": utc_offset, "latitude": latitude, "longitude": longitude}
    natal = httpx.get(f"{ASTRO_API}/api/natal", params=q, timeout=60)
    natal.raise_for_status()
    dasha = httpx.get(f"{ASTRO_API}/api/dasha", params=q, timeout=60)
    dasha.raise_for_status()
    fc = httpx.get(f"{ASTRO_API}/api/forecast", params={**q, "months": max(1, min(int(months), 24))}, timeout=90)
    fc.raise_for_status()
    n, d, f = natal.json(), dasha.json(), fc.json()
    d.pop("periods", None)
    f.pop("house_change_calendar", None)
    return {"natal": _trim(n), "dasha_now": d, "forecast": _trim(f)}


ASTRO_HANDLERS = {"astro_chart": astro_chart}


def run_astro(history, lang):
    return run_tools(ASTRO_SYSTEM, history, ASTRO_TOOLS, ASTRO_HANDLERS, max_tokens=1300)
