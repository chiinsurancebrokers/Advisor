# Advisor

One chat that routes to the right project agent (Asklepios, Nomikosarogos; others link out for now).

- `app.py` FastAPI backend: router, safety floor, agent adapters
- `triage_rules.py` copied unchanged from the Kira repo (deterministic emergency rules)
- `public/index.html` chat UI with Supabase email-code login (Nomikosarogos Supabase project)

Env: `ANTHROPIC_API_KEY`, `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_JWT_SECRET`, optional `NOMIKOS_API_URL`, `ADVISOR_MODEL`.
