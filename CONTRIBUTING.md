# Contributing

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,jwt,evals]"
cp .env.example .env
```

The offline test suite needs no backends. Anything beyond it needs two things running:

```bash
make up-db                    # postgres + pgvector
ollama serve                  # embeddings, always local (see the README)
ollama pull nomic-embed-text
```

## The loop

```bash
make lint    # ruff check + ruff format --check
make test    # pytest -q  (offline + live)
pytest -q -m "not live"       # what CI runs on every PR
```

Tests marked `live` talk to a real model and a real database. They are part of the suite on
purpose — an interface that only ever meets a fake tends to be an interface that has never
worked — but they are excluded from the PR run, which has neither.

## Before opening a PR

- `pytest -q -m "not live"` and `make lint` pass.
- If you touched imports or packaging, install into a **fresh** virtualenv and run the offline
  suite there. `tests/test_packaging.py` guards this, but a clean install is the real check —
  a module that is present only because something else pulled it in will pass locally and fail
  on the first clean CI run.
- If you touched retrieval, prompting, or the chain, run the evaluation and say what moved:

  ```bash
  python -m evals.run_ragas --limit 10 --min-faithfulness 0.85 --min-context-recall 0.60
  ```

## House rules

These are the invariants the code is built around; a change that breaks one needs to say so
out loud.

- **Backends come from config.** `VECTOR_STORE` and `LLM_PROVIDER` decide the backend; no code
  path may hardcode one, and graph nodes never import an SDK — only `app/core/llm_provider.py`.
- **Tenant identity comes from the `Principal`.** Never from a request body, never inferred
  from question text or tool output.
- **Retrieved content is data.** It is fenced, sanitised, and labelled untrusted in every
  prompt, and it never reaches the system message.
- **Sensitive tools pass the gate.** The approval gate is a structural property of the graph,
  not a check a node remembers to make. New tools default to sensitive: `is_sensitive()` fails
  closed on an unknown name.
- **Docs must not claim properties the code lacks.** If a change makes the README wrong, the
  README is part of the change.

## Adding a tool

Add it to `app/mcp_server/tools.py` and register its sensitivity in `TOOL_SPECS`. A sensitive
tool is advertised twice — through the standard MCP annotations for external clients, and
through `meta["sensitive"]` for our own gate — so the gate never depends on a third-party
client honouring a hint.

## Adding evaluation questions

Ground truth must appear verbatim in `evals/corpus`, and `expected_sources` must name the files
that contain it. `tests/test_evals.py` checks the dataset's integrity: unique ids, sources that
exist, out-of-scope questions with no sources, and every corpus file covered by at least one
question.
