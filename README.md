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
- 🧠 **Two vector store backends** — pgvector (default, zero extra infrastructure) and Qdrant (for heavier filtering / multi-tenant setups), switchable via config
- 🛠️ **Tools via MCP** — the agent's tools are exposed through a standalone MCP server, so the same tool server also works from Claude Desktop or Claude Code
- ✋ **Human-in-the-loop** — any action tagged as sensitive (sending an email, writing to an external system) pauses the graph and waits for explicit approval before executing
- 📊 **Measured, not vibes-based** — a golden dataset of 30–50 questions scored with RAGAS (faithfulness, context recall) runs in CI on every PR
- 🔍 **Full observability** — every trace (LLM calls, tool calls, retrieval) logged to a self-hosted Langfuse instance
- 🔒 **Injection-aware by design** — retrieved content is explicitly delimited and marked as untrusted data in every prompt
- 🖥️ **Cloud or fully local LLM** — swap Claude for a local Ollama model with one config change, for sensitive documents that shouldn't leave the machine

## Architecture

```
                          ┌─────────────┐
        user question ───►│  FastAPI    │  streaming responses, JWT auth
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
              ┌─────────────┐  ┌──────────────────────┐
              │ pgvector /  │  │  human-in-the-loop    │
              │ Qdrant      │  │  gate before execution│
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

```bash
git clone https://github.com/<your-username>/multi-agent-rag-support-bot
cd multi-agent-rag-support-bot
cp .env.example .env          # add your ANTHROPIC_API_KEY

docker compose up -d          # postgres + pgvector, qdrant, langfuse

# drop your own documents in ./data/documents, then:
python -m scripts.index_documents

uvicorn app.main:app --reload
```

Open `http://localhost:8000/docs` for the API, `http://localhost:3000` for the Langfuse dashboard.

### Switching vector store

```bash
# in .env
VECTOR_STORE=pgvector   # or: qdrant
```

### Switching LLM provider

```bash
# in .env
LLM_PROVIDER=claude     # or: ollama

# if using ollama — run it locally first:
#   ollama pull llama3.1
#   ollama serve
OLLAMA_MODEL=llama3.1
OLLAMA_BASE_URL=http://localhost:11434
```

Every graph node calls a single `LLMProvider` interface (`app/core/llm_provider.py`) — the nodes never know which backend is actually answering. This also makes it possible to run the same golden dataset through both providers and compare RAGAS scores side by side.

## Project structure

```
app/
├── api/                  # FastAPI routes
├── graph/                # LangGraph definition
│   ├── state.py          #   shared state schema
│   ├── nodes.py          #   supervisor, researcher, action-taker, reviewer
│   └── build.py          #   graph assembly, conditional edges
├── rag/
│   ├── chunking.py
│   ├── retrievers/        #   pgvector.py, qdrant.py
│   └── chain.py           #   LCEL RAG chain
├── mcp_server/            # standalone MCP server exposing agent tools
└── core/
    ├── llm_provider.py     #   Claude / Ollama abstraction, swappable via .env
    ├── config.py
    └── auth.py

evals/
├── golden_dataset.json
└── run_ragas.py           # run in CI, fails build below threshold

.github/workflows/
└── evals.yml
```

## Evaluation

```bash
python -m evals.run_ragas --min-faithfulness 0.85 --min-context-recall 0.60
```

Runs the golden dataset through the live pipeline and fails CI if retrieval or faithfulness regresses — the same discipline as a test suite, applied to a system where `assert answer == expected` doesn't work.

## Security notes

- Retrieved content is wrapped in explicit delimiters and marked as untrusted reference material, never treated as instructions
- Tools are least-privilege: read-only by default, write/send actions require the human-in-the-loop gate
- Tenant/user filters are applied server-side from the auth token, never from client-supplied values

## Roadmap

- [ ] Hybrid search (BM25 + vector) with reciprocal rank fusion
- [ ] Reranker stage before context is sent to the LLM
- [ ] Prompt-caching for the system prompt / tool definitions (Claude only)
- [ ] Multi-tenant mode with per-tenant Qdrant collections
- [ ] RAGAS comparison report: Claude vs. local Ollama model on the same golden dataset

## License

MIT
