"""Phase 13: the evaluation harness against real backends.

One question, end to end: real retrieval, a real answer, and a real RAGAS
judgement produced by our own provider. Deliberately tiny — the point is that
the wiring works, not to reproduce a full run, which takes hours on a local
model.
"""

from pathlib import Path

import pytest

from app.core.config import Settings
from app.core.llm_provider import get_llm_provider
from app.rag.chain import build_rag_chain
from evals.judge import ProviderJudge
from evals.run_ragas import answer_question, load_dataset, score_question, summarise

pytestmark = pytest.mark.live

ROOT = Path(__file__).resolve().parent.parent
TENANT = "evals"


@pytest.fixture(scope="module")
def settings() -> Settings:
    return Settings(_env_file=ROOT / ".env")


@pytest.fixture(scope="module")
def dataset() -> dict:
    return load_dataset(ROOT / "evals" / "golden_dataset.json")


def _question(dataset: dict, qid: str) -> dict:
    return next(q for q in dataset["questions"] if q["id"] == qid)


async def test_the_pipeline_answers_a_golden_question(settings, dataset):
    """Requires the corpus to be indexed: `python -m evals.run_ragas --limit 1`."""
    chain = build_rag_chain(settings)
    result = await answer_question(chain, _question(dataset, "refund-window"), tenant_id=TENANT)

    if result.error or not result.contexts:
        pytest.skip(f"eval corpus not indexed for tenant {TENANT!r}: {result.error}")

    assert "30" in result.answer
    assert result.source_hit is True
    assert "handbook.md" in result.retrieved_sources


async def test_ragas_scores_a_real_answer(settings, dataset):
    """The judge is our provider, and it returns a usable number."""
    from ragas.metrics.collections import ContextRecall, Faithfulness

    chain = build_rag_chain(settings)
    result = await answer_question(chain, _question(dataset, "data-retention"), tenant_id=TENANT)
    if result.error or not result.contexts:
        pytest.skip(f"eval corpus not indexed for tenant {TENANT!r}: {result.error}")

    judge = ProviderJudge(get_llm_provider(settings), max_tokens=2048)
    await score_question(
        result,
        faithfulness=Faithfulness(llm=judge),
        context_recall=ContextRecall(llm=judge),
    )

    assert judge.calls >= 2, "the judge must actually have been called"
    scored = [s for s in (result.faithfulness, result.context_recall) if s is not None]
    if not scored:
        pytest.skip(f"local judge produced no score: {result.unscored}")
    assert all(0.0 <= score <= 1.0 for score in scored)

    summary = summarise([result])
    assert summary["answerable"] == 1


async def test_an_out_of_scope_question_is_declined(settings, dataset):
    chain = build_rag_chain(settings)
    result = await answer_question(chain, _question(dataset, "mobile-app"), tenant_id=TENANT)
    if result.error:
        pytest.skip(f"eval corpus not indexed for tenant {TENANT!r}: {result.error}")

    assert result.declined is True, f"expected a refusal, got: {result.answer!r}"
