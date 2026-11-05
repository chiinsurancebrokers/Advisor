# Advisor

One chat that routes to the right project agent (Asklepios, Nomikosarogos; others link out for now).

- `app.py` FastAPI backend: router, safety floor, agent adapters
- `triage_rules.py` copied unchanged from the Kira repo (deterministic emergency rules)
- `public/index.html` chat UI with Supabase email-code login (Nomikosarogos Supabase project)

Env: `ANTHROPIC_API_KEY`, `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_JWT_SECRET`, optional `NOMIKOS_API_URL`, `ADVISOR_MODEL`.
- `kb.py`, `kb_vet.py`, `vet_knowledge.py`: copied unchanged from the PetsAIHealth repo (kiraaipet). Retrieval (RAG) over the curated vet guidance: the pets agent gets tagged facts in its prompt and a "Sources" line for what it cited. Change them in kiraaipet first, then copy. Env `KB_RAG=off` turns the BM25 part off.
- `kb_med.py`, `lifestyle_info.py` (+ `triage_rules.py`): copied from the Kira repo. Same retrieval for the Asklepios agent.
- `tests/test_pets_rag.py`: `python3 -I tests/test_pets_rag.py` (no network, Anthropic faked).
- Login: "Send code" sends one request per tap and then waits 60 s (a double tap used to send two codes; the second invalidates the first). The email still comes from the shared Supabase project's template: give the Advisor its own sender/template in Supabase Auth if it should not say "Νομικός Αρωγός".
