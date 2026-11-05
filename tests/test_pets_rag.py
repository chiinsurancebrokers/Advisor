"""Pets agent: retrieval, no-greeting rule and citations (no network, Anthropic faked).  python3 -I tests/test_pets_rag.py"""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import agents_tools as at

fails = 0
def check(name, cond):
    global fails; fails += (not cond); print(("OK   " if cond else "FAIL ") + name)

SEEN = {}
class _Msgs:
    def create(self, **kw):
        SEEN.update(kw)
        return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text=SEEN["reply"])], stop_reason="end_turn")
at._client = types.SimpleNamespace(messages=_Msgs())

def ask(texts, reply, species="dog", lang="el"):
    SEEN.clear(); SEEN["reply"] = reply
    hist = [{"role": "user", "content": t} for t in texts]
    res = at.pets_floor(species, texts)
    return at.run_pets(hist, lang, res, None, species=species)

out = ask(["Ο σπίθας έκανε 3 εμετούς, 1 διάρροια, αρνείται φαγητό, είναι κουρασμένος"], "Λυπάμαι που ο Σπίθας δεν είναι καλά. Εμετός και διάρροια πάνω από μέρα θέλουν κτηνίατρο σύντομα [S2].")
sysmsg = SEEN["system"]
check("system prompt has the retrieved, tagged facts", "RETRIEVED REFERENCE FACTS" in sysmsg and "[S1]" in sysmsg)
check("system prompt forbids a greeting for a sick pet", "OPENING: never start with a greeting" in sysmsg)
check("reply: tag removed and the cited source listed", "[S2]" not in out and "Πηγές:" in out)
check("reply: opening is concern, not a welcome", out.startswith("Λυπάμαι") and "Χαίρομαι" not in out)
out = ask(["Ο σπίθας κουτσαίνει λίγο"], "Παρακολούθησέ τον [S9].")
check("invented tag is dropped, no source line", "[S9]" not in out and "Πηγές" not in out)
out = ask(["ποιος είναι ο πρωθυπουργός"], "Μπορώ να βοηθήσω μόνο για κατοικίδια.")
check("no retrieval block for an unrelated message", "[S1]" not in SEEN["system"])
ask(["Έφαγε σοκολάτα"], "ok")
check("keyword topics still reach the prompt", "theobromine" in SEEN["system"])
print(f"\n{fails} failed"); sys.exit(1 if fails else 0)
