# Multi-Agent RAG Support Bot

A self-hostable, domain-agnostic support agent: point it at your own documents, and it answers questions, routes complex cases to specialist sub-agents, and asks for human approval before taking any risky action.

Built as a reference implementation of a modern production RAG/agent stack — not a single-file tutorial, but the pieces wired together the way they'd actually ship.

> Swap the documents, swap the domain. The same stack runs a legal-docs assistant, an internal company wiki bot, or a personal knowledge base — nothing about the architecture is niche-specific.

> Swap the model, too. A single `LLMProvider` interface sits between the graph and the LLM — Claude by default, or a fully local Ollama model when the documents shouldn't leave the machine at all.

---

## Why this exists

Most public RAG demos stop at "embed some text, ask a question." This project goes one step further, into the parts that actually separate a prototype from something a team could run in production:

- Retrieval that's actually measured (RAGAS), not just eyeballed
- An agent that can escalate to specialists instead of one monolithic prompt
- A hard stop before anything irreversible happens (human-in-the-loop)
- Full tracing of every decision the agent makes, self-hosted so no conversation data leaves your own infrastructure

## Features

- 📄 **Bring your own documents** — drop PDFs/Markdown/text into a folder, they're chunked, embedded, and indexed automatically
- 🔀 **Smart routing** — simple questions get a fast single-shot RAG answer; complex or ambiguous ones go through a supervisor agent that delegates to specialist nodes
- 🧠 **Pluggable vector store** — pgvector by default (zero extra infrastructure); a Qdrant adapter ships behind the same interface for heavier filtering / multi-tenant setups, opt-in and not bundled
- 🛠️ **Tools via MCP** — the agent's tools are exposed through a standalone MCP server, so the same tool server also works from Claude Desktop or Claude Code
- ✋ **Human-in-the-loop** — any action tagged as sensitive (sending an email, writing to an external system) pauses the graph and waits for explicit approval before executing
- 📊 **Measured, not vibes-based** — a golden dataset of 50 questions scored with RAGAS (faithfulness, context recall, retrieval accuracy, refusal rate) runs in CI on every PR
- 🔍 **Full observability** — every trace (LLM calls, tool calls, retrieval) logged to a self-hosted Langfuse instance
- 🔒 **Injection-aware by design** — retrieved content is explicitly delimited and marked as untrusted data in every prompt
- 🖥️ **Cloud or fully local LLM** — swap Claude for a local Ollama model with one config change, for sensitive documents that shouldn't leave the machine

## Architecture

```
                          ┌─────────────┐
        user question ───►│  FastAPI    │  streaming responses, auth-gated
                          │  (entry)    │
                          └──────┬──────┘
                                 ▼
                          ┌─────────────┐
                          │  LangGraph  │  routes: simple RAG vs. supervisor
                          │  entry node │
                          └──────┬──────┘
                     ┌───────────┴────────────┐
                     ▼                        ▼
              ┌─────────────┐         ┌────────────────┐
              │ simple RAG  │         │  supervisor    │
              │ (LCEL chain)│         │  (multi-agent) │
              └──────┬──────┘         └───────┬────────┘
                     │              ┌──────────┼──────────┐
                     │              ▼          ▼          ▼
                     │        researcher   action-taker  reviewer
                     │              │          │
                     ▼              ▼          ▼ (via MCP tools)
              ┌─────────────┐  ┌───────────────────────┐
              │  pgvector   │  │  human-in-the-loop    │
              │ (or Qdrant) │  │  gate before execution│
              └─────────────┘  └───────────────────────┘

        every node traced end-to-end → self-hosted Langfuse
        every deploy gated on RAGAS scores → GitHub Actions
```

## Tech stack

| Layer | Tool |
|---|---|
| API | FastAPI, streaming via SSE |
| Orchestration | LangGraph (routing, loops, human-in-the-loop) |
| Chains | LangChain (LCEL) |
| LLM | Anthropic Claude API, or local via Ollama (swappable) |
| Vector store | pgvector (default) or Qdrant |
| Tools | Custom MCP server |
| Evaluation | RAGAS |
| Observability | Langfuse (self-hosted) |
| Database | PostgreSQL |
| Infra | Docker Compose |

## Quickstart

**Requirements:** Docker, Python 3.11+, and [Ollama](https://ollama.com) — embeddings always run
locally (see [Embeddings](#embeddings) below), even when answers come from Claude.

```bash
git clone https://github.com/<your-username>/multi-agent-rag-support-bot
cd multi-agent-rag-support-bot
cp .env.example .env          # add your ANTHROPIC_API_KEY (or set LLM_PROVIDER=ollama)

python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

ollama pull nomic-embed-text  # embeddings; ~275 MB
ollama serve &

docker compose up -d          # postgres + pgvector, langfuse (web, worker, clickhouse, redis, minio)
make wait                     # block until postgres is healthy

# Try it on the bundled sample corpus, or drop your own files in ./data/documents:
python -m scripts.seed_documents
python -m scripts.index_documents

uvicorn app.main:app --reload
```

```bash
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"question": "How long do I have to request a refund?"}'
```

Open `http://localhost:8000/docs` for the API and `http://localhost:3000` for the Langfuse
dashboard (Compose provisions the project and keys from `.env`, so tracing works on first boot).

`make help` lists the shortcuts: `up`, `up-db` (postgres only — enough for tests and indexing),
`down`, `clean`, `health`, `test`, `lint`, `evals`.

### Embeddings

Anthropic exposes no embeddings endpoint, so embeddings are always produced by Ollama
(`EMBEDDING_MODEL`, default `nomic-embed-text`) regardless of `LLM_PROVIDER`. With
`LLM_PROVIDER=claude` you still need a reachable Ollama for indexing and retrieval; only the
answers come from Claude. `EMBEDDING_DIM` sizes the `vector(...)` column and is read by the
Postgres init script, so changing it means `docker compose down -v` and a re-index.

### Auth

There is no login and no user table. `get_principal()` resolves the caller's
`Principal(user_id, tenant_id, scopes)`, and **the tenant always comes from there — never from
the request body**:

```bash
# in .env
AUTH_MODE=local         # a fixed principal from LOCAL_USER_ID / LOCAL_TENANT_ID (default)
# AUTH_MODE=jwt         # verify a bearer token onto the same Principal
```

JWT mode needs the extra and a secret:

```bash
pip install -e ".[jwt]"
JWT_SECRET=... python -m scripts.make_token --tenant acme --scopes "chat approvals:write"
```

### Switching vector store

```bash
# in .env
VECTOR_STORE=pgvector   # or: qdrant
```

pgvector is the default and the only backend the project runs, tests, and evaluates against.
The Qdrant adapter is written to the same `Retriever` interface and ships in the repo, but it
is **not bundled and not verified**: it needs the extra and the opt-in Compose overlay.

```bash
pip install -e ".[qdrant]"
docker compose -f docker-compose.yml -f docker-compose.qdrant.yml up -d
```

Selecting `qdrant` without the extra fails at startup with the install command, rather than at
the first query.

### Switching LLM provider

```bash
# in .env
LLM_PROVIDER=claude     # or: ollama

# if using ollama — run it locally first:
#   ollama pull gemma4:12b-mlx
#   ollama serve
OLLAMA_MODEL=gemma4:12b-mlx
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_THINK=true       # reasoning models: see the note below
```

Model choice is part of the security posture, not only an answer-quality knob. A small local
model will follow instructions found *inside* a retrieved document even though the prompt fences
it correctly — measured, with the fence holding and the model complying anyway. Prefer a model
that resists it; `gemma4:12b-mlx` and `gpt-oss:20b` did in testing, `lfm2.5:8b` did not.

Reasoning models spend part of their token budget on `message.thinking`, and a budget consumed
entirely by thinking returns an **empty answer**. `OLLAMA_THINK=false` disables it where the
model only has to fill in a schema (the evaluation judge does this by default: 121s and an empty
reply became 4s and correct JSON).

Every graph node calls a single `LLMProvider` interface (`app/core/llm_provider.py`) — the nodes never know which backend is actually answering. This also makes it possible to run the same golden dataset through both providers and compare RAGAS scores side by side.

## Project structure

```
app/
├── api/                   # FastAPI routes
│   ├── chat.py            #   POST /chat, POST /chat/stream (SSE)
│   ├── approvals.py       #   list / inspect / decide pending actions
│   └── approvals_store.py #   index of paused runs, per tenant
├── graph/                 # LangGraph definition
│   ├── state.py           #   shared state schema
│   ├── nodes.py           #   router + simple-RAG branch
│   ├── supervisor.py      #   supervisor, researcher, action-taker, reviewer
│   ├── approval.py        #   the human-in-the-loop gate (interrupt / resume)
│   └── build.py           #   graph assembly, conditional edges
├── rag/
│   ├── chunking.py        #   PDF / Markdown / text loaders, structure-aware splits
│   ├── retrievers/        #   base.py, pgvector.py, qdrant.py
│   └── chain.py           #   LCEL RAG chain
├── mcp_server/            # standalone MCP server exposing agent tools
└── core/
    ├── llm_provider.py    #   Claude / Ollama abstraction, swappable via .env
    ├── observability.py   #   Langfuse tracing (callback handler + wrappers)
    ├── config.py
    └── auth.py

evals/
├── corpus/                # fixture documents the golden dataset is written against
├── golden_dataset.json    # 50 questions: 45 answerable, 5 out-of-scope
├── judge.py               # RAGAS judge wired to LLMProvider (never OpenAI)
└── run_ragas.py           # run in CI, fails the build below threshold

scripts/
├── seed_documents.py      # copy the sample corpus into ./data/documents
├── index_documents.py     # chunk, embed, upsert; idempotent, --prune, --dry-run
└── make_token.py          # mint a dev JWT when AUTH_MODE=jwt

.github/workflows/
├── ci.yml                 # ruff + the offline test suite, on every PR
└── evals.yml              # RAGAS thresholds, on every PR
```

## Evaluation

```bash
python -m evals.run_ragas --min-faithfulness 0.85 --min-context-recall 0.60
```

Runs the golden dataset through the live pipeline and fails CI if retrieval or faithfulness
regresses — the same discipline as a test suite, applied to a system where
`assert answer == expected` doesn't work. Four numbers come out of a run:

| metric | what it catches |
|---|---|
| faithfulness | claims in the answer that the retrieved context does not support |
| context recall | retrieval that missed what the reference answer needs |
| source accuracy | the right answer from the wrong file (deterministic, no model involved) |
| refusal rate | confident answers to questions the corpus cannot answer |

Out-of-scope questions are scored by refusal, not folded into the faithfulness mean. A run
writes a JSON report with every question's score, and any question that errors outright fails
the run: "nothing scored" must never read as "passed".

**The judge is your provider, not OpenAI.** RAGAS defaults to OpenAI for judging and pulls
`openai` in transitively; `evals/judge.py` implements the judge interface on top of
`LLMProvider` instead, so evaluation traffic — retrieved documents included — stays on the
backend you configured. A test asserts no OpenAI client is ever constructed.

```bash
# the same dataset through either backend, for a side-by-side comparison
python -m evals.run_ragas --provider claude --judge-provider claude
python -m evals.run_ragas --provider ollama --judge-provider ollama
```

Measured on the bundled corpus: `claude-opus-5` scored 0.987 faithfulness / 1.000 context
recall in 100 seconds; `gemma4:12b-mlx` scored 0.974 / 1.000 in 49 minutes. Read a local
faithfulness score as a floor — a small judge marks some correct, grounded statements as
unsupported, which is why the threshold sits at 0.85 rather than 0.95.

## Observability

Every request becomes one Langfuse trace: the root span, the LangGraph run, one span per node,
retrieval with the citations it returned, every model call as a generation (model, tokens,
time-to-first-token), and every MCP tool call with the approval state behind it. `tenant_id`,
`user_id` and the thread id are attached to the whole trace, and the `trace_id` comes back in
the API response so an answer can be correlated with its trace.

Tracing is best-effort by construction: with no keys configured there is no client and every
helper short-circuits, and an unreachable Langfuse costs a log line rather than a failed chat.
`GET /health` reports which backends are configured and reachable.

## Security notes

- Retrieved content is wrapped in explicit delimiters and marked as untrusted reference
  material, never treated as instructions. Chunk text is sanitised so a document cannot forge
  or close a delimiter, and it is never interpolated into the system prompt. This removes the
  trivial breakouts; it cannot make a weak model obey the rule (see
  [Switching LLM provider](#switching-llm-provider)).
- Tools are least-privilege: read-only by default, write/send actions require the
  human-in-the-loop gate. The gate is **structural** — no edge in the graph reaches a dispatch
  without passing through it — and it fails closed: any resume value that is not an explicit
  approval is a rejection. A human may correct an action's arguments, never its tool name.
- Tenant/user filters are applied server-side from the caller's `Principal`, never from
  client-supplied values. `ChatRequest` has no `tenant_id` field and forbids extras, so sending
  one is a 422 before the graph runs — not a silently ignored value. Another tenant's approval
  thread returns 404, not 403; a 403 would confirm it exists.
- `send_email` writes to a local outbox file and has no SMTP credentials, so the pipeline is
  demonstrable without granting the process the ability to actually mail anyone.

## Development

```bash
pip install -e ".[dev,jwt,evals]"

make up-db        # postgres alone is enough for the offline suite
make test         # pytest -q
make lint         # ruff check + ruff format --check
```

Tests marked `live` talk to a real Ollama/Claude and a real Postgres; the default CI run is
`pytest -m "not live"`. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Roadmap

- [ ] Hybrid search (BM25 + vector) with reciprocal rank fusion
- [ ] Reranker stage before context is sent to the LLM
- [ ] Prompt-caching for the system prompt / tool definitions (Claude only)
- [ ] Multi-tenant mode with per-tenant Qdrant collections
- [ ] RAGAS comparison report: Claude vs. local Ollama model on the same golden dataset

## License

MIT — see [LICENSE](LICENSE).
