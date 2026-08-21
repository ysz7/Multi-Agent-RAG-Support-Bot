"""Score the golden dataset with RAGAS.

    python -m evals.run_ragas --min-faithfulness 0.85 --min-context-recall 0.60

Runs every question through the real retrieval-and-answer pipeline (the Phase 6
LCEL chain, against a real vector store and a real model), then scores the
results with RAGAS metrics judged by our own provider. Exits non-zero when a
mean score falls below a threshold, so CI fails on a regression the same way it
fails on a broken test.

Three numbers come out of a run:

* **faithfulness** — is every claim in the answer supported by the retrieved
  context? This is the anti-hallucination measure, and it is judged per claim.
* **context recall** — did retrieval actually fetch what the reference answer
  needs? A high faithfulness with a low context recall means the model is
  honest about a context that was never good enough.
* **source accuracy** — did the expected file appear in the retrieved chunks?
  Cheap, deterministic, and no model is involved, so it isolates retrieval from
  judgement.

Out-of-scope questions are scored separately: the corpus cannot answer them, so
the only correct behaviour is declining, and faithfulness against an empty
context is meaningless. They are reported as a refusal rate, not folded into the
means.

Both the pipeline provider and the judge provider are selected from `.env` and
can be overridden per run, which is what makes a Claude-vs-Ollama comparison on
the same dataset a matter of two invocations.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.core.llm_provider import get_llm_provider
from app.core.observability import (
    configure_observability,
    instrument_llm,
    request_trace,
)
from app.core.observability import (
    shutdown as shutdown_observability,
)
from app.rag.chain import NO_CONTEXT_ANSWER, RagChain, build_rag_chain
from evals.judge import ProviderJudge

logger = logging.getLogger("evals")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "evals" / "golden_dataset.json"
DEFAULT_CORPUS = ROOT / "evals" / "corpus"
DEFAULT_TENANT = "evals"

# Phrases that mark a decline. Matched against a *normalised* answer: models
# write "don't contain" as often as "does not contain", and an early version of
# this list only had the expanded forms — which measured our regexes rather than
# the model, and scored three correct refusals as failures.
_DECLINE_MARKERS = (
    NO_CONTEXT_ANSWER[:40].lower(),
    "do not contain",
    "does not contain",
    "do not cover",
    "does not cover",
    "do not address",
    "does not address",
    "do not mention",
    "does not mention",
    "do not provide",
    "does not provide",
    "do not specify",
    "does not specify",
    "do not say",
    "does not say",
    "do not include",
    "does not include",
    "no information",
    "not specified",
    "not stated",
    "cannot answer",
    "can not answer",
    "cannot be answered",
    "i do not have",
    "unable to answer",
    "nothing about",
)

# Contractions, expanded before matching. Apostrophes arrive as both ' and ’.
_CONTRACTIONS = {
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "can't": "cannot",
    "won't": "will not",
    "isn't": "is not",
    "aren't": "are not",
    "i'm": "i am",
}


@dataclass
class QuestionResult:
    id: str
    kind: str
    question: str
    answer: str
    reference: str
    expected_sources: list[str]
    retrieved_sources: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)
    faithfulness: float | None = None
    context_recall: float | None = None
    source_hit: bool | None = None
    declined: bool | None = None
    error: str | None = None
    unscored: list[str] = field(default_factory=list)
    seconds: float = 0.0


def load_dataset(path: Path, *, ids: list[str] | None = None, limit: int | None = None) -> dict:
    data = json.loads(path.read_text())
    questions = data["questions"]
    if ids:
        wanted = set(ids)
        questions = [q for q in questions if q["id"] in wanted]
        missing = wanted - {q["id"] for q in questions}
        if missing:
            raise SystemExit(f"unknown question ids: {', '.join(sorted(missing))}")
    if limit:
        questions = questions[:limit]
    data["questions"] = questions
    return data


def normalise_answer(answer: str) -> str:
    """Lowercase and expand contractions so marker matching is not spelling-sensitive."""
    text = answer.lower().replace("\u2019", "'")
    for contraction, expansion in _CONTRACTIONS.items():
        text = text.replace(contraction, expansion)
    return text


def looks_declined(answer: str) -> bool:
    """Whether an answer is a refusal rather than a claim."""
    text = normalise_answer(answer)
    return any(marker in text for marker in _DECLINE_MARKERS)


def _clean(value: float | None) -> float | None:
    """RAGAS returns NaN when it cannot score a row; keep that out of the means."""
    if value is None:
        return None
    return None if isinstance(value, float) and math.isnan(value) else float(value)


async def answer_question(chain: RagChain, question: dict, *, tenant_id: str) -> QuestionResult:
    """Run one question through the pipeline and collect what scoring needs."""
    result = QuestionResult(
        id=question["id"],
        kind=question.get("kind", "answerable"),
        question=question["question"],
        answer="",
        reference=question.get("reference", ""),
        expected_sources=list(question.get("expected_sources") or []),
    )
    started = time.monotonic()
    with request_trace(
        name=f"eval.{question['id']}",
        question=question["question"],
        user_id="evals",
        session_id=f"eval-{question['id']}",
        tenant_id=tenant_id,
        tags=["eval", result.kind],
    ):
        try:
            answer = await chain.ainvoke(question["question"], tenant_id=tenant_id)
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            result.seconds = time.monotonic() - started
            return result

    result.seconds = time.monotonic() - started
    result.answer = answer.text
    result.contexts = [chunk.content for chunk in answer.chunks]
    result.retrieved_sources = sorted({Path(c.source_path).name for c in answer.chunks})
    result.citations = [citation.citation for citation in answer.citations]
    result.declined = looks_declined(answer.text)
    if result.expected_sources:
        result.source_hit = all(
            source in result.retrieved_sources for source in result.expected_sources
        )
    return result


async def score_question(result: QuestionResult, *, faithfulness, context_recall) -> None:
    """Attach RAGAS scores. Out-of-scope rows are judged by refusal, not by score."""
    if result.error or result.kind == "out_of_scope":
        return
    if not result.contexts or not result.answer:
        result.error = result.error or "no context retrieved"
        return

    # A judge failure is a missing score, never a failed run: one slow or
    # malformed judgement must not throw away the other 49 questions.
    try:
        score = await faithfulness.ascore(
            user_input=result.question,
            response=result.answer,
            retrieved_contexts=result.contexts,
        )
        result.faithfulness = _clean(score.value)
    except Exception as exc:
        result.unscored.append(f"faithfulness: {type(exc).__name__}: {exc}")
        logger.warning("faithfulness judge failed for %s: %s", result.id, exc)

    try:
        score = await context_recall.ascore(
            user_input=result.question,
            retrieved_contexts=result.contexts,
            reference=result.reference,
        )
        result.context_recall = _clean(score.value)
    except Exception as exc:
        result.unscored.append(f"context_recall: {type(exc).__name__}: {exc}")
        logger.warning("context recall judge failed for %s: %s", result.id, exc)


def summarise(results: list[QuestionResult]) -> dict[str, Any]:
    answerable = [r for r in results if r.kind == "answerable"]
    out_of_scope = [r for r in results if r.kind == "out_of_scope"]

    def mean(values: list[float]) -> float | None:
        return round(statistics.fmean(values), 4) if values else None

    faithfulness = [r.faithfulness for r in answerable if r.faithfulness is not None]
    recall = [r.context_recall for r in answerable if r.context_recall is not None]
    hits = [r.source_hit for r in answerable if r.source_hit is not None]
    declines = [r.declined for r in out_of_scope if r.declined is not None]

    return {
        "questions": len(results),
        "answerable": len(answerable),
        "out_of_scope": len(out_of_scope),
        "errors": sum(1 for r in results if r.error),
        "faithfulness": mean(faithfulness),
        "faithfulness_scored": len(faithfulness),
        "context_recall": mean(recall),
        "context_recall_scored": len(recall),
        "source_accuracy": mean([float(h) for h in hits]),
        "refusal_rate": mean([float(d) for d in declines]),
        "mean_seconds": mean([r.seconds for r in results]),
    }


def check_thresholds(summary: dict, args: argparse.Namespace) -> list[str]:
    """Threshold failures, as human-readable lines. Empty means the run passed.

    Pipeline errors fail the run by default. A run where every question crashed
    scores nothing at all, and "nothing scored" must never read as "passed".
    """
    failures: list[str] = []
    max_errors = getattr(args, "max_errors", 0)
    if max_errors is not None and summary.get("errors", 0) > max_errors:
        failures.append(f"errors: {summary['errors']} > {max_errors} allowed")
    checks = (
        ("faithfulness", args.min_faithfulness),
        ("context_recall", args.min_context_recall),
        ("source_accuracy", args.min_source_accuracy),
        ("refusal_rate", args.min_refusal_rate),
    )
    for name, minimum in checks:
        if minimum is None:
            continue
        value = summary.get(name)
        if value is None:
            failures.append(f"{name}: nothing scored, cannot meet the {minimum} threshold")
        elif value < minimum:
            failures.append(f"{name}: {value:.3f} < {minimum}")
    return failures


def render_report(report: dict) -> str:
    summary = report["summary"]
    lines = [
        "",
        f"RAGAS report — {report['dataset']['questions']} questions"
        f" · pipeline={report['config']['llm_provider']}:{report['config']['model']}"
        f" · judge={report['config']['judge_provider']}:{report['config']['judge_model']}",
        "",
    ]
    for name in (
        "faithfulness",
        "context_recall",
        "source_accuracy",
        "refusal_rate",
        "mean_seconds",
    ):
        value = summary.get(name)
        lines.append(f"  {name:<18} {'n/a' if value is None else f'{value:.3f}'}")
    lines.append(f"  {'errors':<18} {summary['errors']}")

    worst = sorted(
        (r for r in report["results"] if r["kind"] == "answerable"),
        key=lambda r: (
            r["faithfulness"] if r["faithfulness"] is not None else -1,
            r["context_recall"] if r["context_recall"] is not None else -1,
        ),
    )[:5]
    if worst:
        lines += ["", "  lowest scoring questions:"]
        for row in worst:
            lines.append(
                f"    {row['id']:<24} faithfulness={row['faithfulness']}"
                f" context_recall={row['context_recall']} sources={row['retrieved_sources']}"
            )
    return "\n".join(lines) + "\n"


async def run(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.dataset, ids=args.ids, limit=args.limit)
    questions = dataset["questions"]
    if not questions:
        raise SystemExit("dataset is empty")

    overrides: dict[str, Any] = {}
    if args.provider:
        overrides["llm_provider"] = args.provider
    settings = Settings(_env_file=ROOT / ".env", **overrides)
    configure_observability(settings)

    if not args.skip_index:
        from scripts.index_documents import index_documents

        totals = await index_documents(
            settings, tenant_id=args.tenant, root=args.corpus, prune=True, dry_run=False
        )
        logger.info("corpus indexed: %s", totals)

    chain = build_rag_chain(settings)

    judge_settings = settings
    if args.judge_provider or args.judge_model:
        judge_overrides = dict(overrides)
        if args.judge_provider:
            judge_overrides["llm_provider"] = args.judge_provider
        if args.judge_model:
            key = (
                "claude_model"
                if (args.judge_provider or settings.llm_provider) == "claude"
                else "ollama_model"
            )
            judge_overrides[key] = args.judge_model
        judge_settings = Settings(_env_file=ROOT / ".env", **judge_overrides)
    judge_overrides_late: dict[str, Any] = {}
    if args.judge_timeout:
        judge_overrides_late["ollama_timeout_s"] = args.judge_timeout
    if not args.judge_thinking:
        # Measured on gemma4:12b-mlx: a statement-extraction prompt takes 121s with
        # thinking on and returns an *empty* answer (the budget goes to thinking),
        # versus 4s and correct JSON with it off. The judge does not need to reason
        # out loud; it needs to fill in a schema.
        judge_overrides_late["ollama_think"] = False
    if judge_overrides_late:
        judge_settings = judge_settings.model_copy(update=judge_overrides_late)

    judge = ProviderJudge(
        instrument_llm(get_llm_provider(judge_settings), judge_settings),
        max_tokens=args.judge_max_tokens,
    )

    from ragas.metrics.collections import ContextRecall, Faithfulness

    faithfulness = Faithfulness(llm=judge)
    context_recall = ContextRecall(llm=judge)

    semaphore = asyncio.Semaphore(args.concurrency)
    done = 0

    async def one(question: dict) -> QuestionResult:
        nonlocal done
        async with semaphore:
            result = await answer_question(chain, question, tenant_id=args.tenant)
            await score_question(result, faithfulness=faithfulness, context_recall=context_recall)
        done += 1
        logger.info(
            "[%d/%d] %s faithfulness=%s context_recall=%s%s",
            done,
            len(questions),
            result.id,
            result.faithfulness,
            result.context_recall,
            f" ERROR {result.error}" if result.error else "",
        )
        return result

    started = time.monotonic()
    results = await asyncio.gather(*(one(question) for question in questions))
    elapsed = time.monotonic() - started

    summary = summarise(list(results))
    failures = check_thresholds(summary, args)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "config": {
            "llm_provider": settings.llm_provider,
            "model": (
                settings.claude_model
                if settings.llm_provider == "claude"
                else settings.ollama_model
            ),
            "judge_provider": judge_settings.llm_provider,
            "judge_model": judge.model,
            "judge_thinking": bool(args.judge_thinking),
            "embedding_model": settings.embedding_model,
            "vector_store": settings.vector_store,
            "retrieval_top_k": settings.retrieval_top_k,
            "tenant": args.tenant,
            "judge_calls": judge.calls,
            "judge_retries": judge.retries,
            "judge_empty_replies": judge.empty_replies,
        },
        "dataset": {
            "path": str(args.dataset),
            "version": dataset.get("version"),
            "questions": len(questions),
        },
        "thresholds": {
            "max_errors": args.max_errors,
            "faithfulness": args.min_faithfulness,
            "context_recall": args.min_context_recall,
            "source_accuracy": args.min_source_accuracy,
            "refusal_rate": args.min_refusal_rate,
        },
        "summary": summary,
        "failures": failures,
        "results": [asdict(r) for r in results],
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")

    print(render_report(report))
    print(f"report written to {args.report}")
    shutdown_observability()

    if failures:
        print("\nFAILED thresholds:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nall thresholds met")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score the golden dataset with RAGAS.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument(
        "--tenant", default=DEFAULT_TENANT, help="tenant to index and query (default: evals)"
    )
    parser.add_argument(
        "--provider", choices=("claude", "ollama"), help="override LLM_PROVIDER for the pipeline"
    )
    parser.add_argument(
        "--judge-provider", choices=("claude", "ollama"), help="provider for the RAGAS judge"
    )
    parser.add_argument("--judge-model", help="model for the RAGAS judge")
    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        default=2048,
        help="token budget for the first judge attempt; doubled on each retry",
    )
    parser.add_argument(
        "--judge-thinking",
        action="store_true",
        help="let the judge model think out loud (Ollama); much slower, rarely better",
    )
    parser.add_argument(
        "--judge-timeout",
        type=float,
        default=900.0,
        help="per-call timeout for the judge; a retry at a doubled budget is slow",
    )
    parser.add_argument("--limit", type=int, help="only run the first N questions")
    parser.add_argument("--ids", nargs="+", help="only run these question ids")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--skip-index", action="store_true", help="assume the corpus is indexed")
    parser.add_argument("--min-faithfulness", type=float)
    parser.add_argument("--min-context-recall", type=float)
    parser.add_argument("--min-source-accuracy", type=float)
    parser.add_argument("--min-refusal-rate", type=float)
    parser.add_argument(
        "--max-errors",
        type=int,
        default=0,
        help="how many questions may fail outright before the run fails (default: 0)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "evals" / "reports" / "ragas-latest.json",
        help="where to write the JSON report artifact",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
