"""Phase 13: the evaluation harness itself.

Scoring a pipeline is only useful if the harness is trustworthy, so this covers
the parts that decide whether CI goes red: the golden dataset's integrity, the
judge bridge that keeps evaluation traffic off OpenAI, the refusal detector, and
the threshold arithmetic. No model and no database are involved.
"""

import pytest

from evals.compat import patch_langchain_community

# The harness lives behind the `[evals]` extra, which `pip install -e ".[dev]"`
# does not bring in — skip rather than fail collection; CI installs the extra.
# The shim has to run first: a bare `import ragas` fails on a *different*
# missing module (see evals/compat.py), which would look like "not installed".
patch_langchain_community()
pytest.importorskip("ragas", reason="install the [evals] extra to run the evaluation tests")

import json
from pathlib import Path

from pydantic import BaseModel

from evals.judge import JudgeError, ProviderJudge, extract_json
from evals.run_ragas import (
    QuestionResult,
    check_thresholds,
    load_dataset,
    looks_declined,
    parse_args,
    summarise,
)

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "evals" / "golden_dataset.json"
CORPUS = ROOT / "evals" / "corpus"


# ---------------------------------------------------------------------------
# The golden dataset
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dataset() -> dict:
    return json.loads(DATASET.read_text())


def test_dataset_size(dataset):
    """The README promises 30–50 questions; hold the file to it."""
    assert 30 <= len(dataset["questions"]) <= 50


def test_ids_are_unique(dataset):
    ids = [q["id"] for q in dataset["questions"]]
    assert len(ids) == len(set(ids))


def test_every_question_is_complete(dataset):
    for question in dataset["questions"]:
        assert question["question"].strip()
        assert question["reference"].strip()
        assert question["kind"] in {"answerable", "out_of_scope"}


def test_expected_sources_exist(dataset):
    """A ground-truth source that is not in the corpus would score as a miss forever."""
    available = {path.name for path in CORPUS.iterdir() if path.is_file()}
    for question in dataset["questions"]:
        for source in question["expected_sources"]:
            assert source in available, f"{question['id']} points at a missing file: {source}"


def test_out_of_scope_questions_have_no_sources(dataset):
    for question in dataset["questions"]:
        if question["kind"] == "out_of_scope":
            assert question["expected_sources"] == []


def test_answerable_questions_have_sources(dataset):
    for question in dataset["questions"]:
        if question["kind"] == "answerable":
            assert question["expected_sources"]


def test_the_corpus_is_covered(dataset):
    """Every corpus file is exercised, or it is dead weight in the fixture."""
    cited = {s for q in dataset["questions"] for s in q["expected_sources"]}
    available = {p.name for p in CORPUS.iterdir() if p.is_file()}
    assert available - cited == set()


def test_load_dataset_filters(tmp_path):
    assert len(load_dataset(DATASET, limit=3)["questions"]) == 3
    picked = load_dataset(DATASET, ids=["refund-window"])["questions"]
    assert [q["id"] for q in picked] == ["refund-window"]
    with pytest.raises(SystemExit):
        load_dataset(DATASET, ids=["no-such-question"])


# ---------------------------------------------------------------------------
# The judge bridge
# ---------------------------------------------------------------------------


class Verdict(BaseModel):
    verdict: int
    reason: str = ""


class FakeResult:
    def __init__(self, text):
        self.text = text
        self.model = "fake"
        self.provider = "fake"


class FakeProvider:
    """Returns a scripted reply per call, recording the token budget it was given."""

    name = "fake"
    model = "fake-model"

    def __init__(self, replies):
        self.replies = list(replies)
        self.budgets: list[int | None] = []

    async def chat(self, messages, *, system=None, max_tokens=None):
        self.budgets.append(max_tokens)
        return FakeResult(self.replies.pop(0) if self.replies else "")

    async def aclose(self):
        return None


@pytest.mark.parametrize(
    "text",
    [
        '{"verdict": 1, "reason": "ok"}',
        '```json\n{"verdict": 1, "reason": "ok"}\n```',
        'Sure! Here is the JSON:\n{"verdict": 1, "reason": "ok"}\nHope that helps.',
        '  \n{"verdict": 1, "reason": "ok"}\n\n',
    ],
)
def test_extract_json_handles_what_models_actually_emit(text):
    assert extract_json(text)["verdict"] == 1


def test_extract_json_rejects_prose():
    with pytest.raises(JudgeError):
        extract_json("I think the answer is probably fine.")
    with pytest.raises(JudgeError):
        extract_json("")


async def test_judge_parses_a_good_reply():
    provider = FakeProvider(['{"verdict": 1, "reason": "supported"}'])
    judge = ProviderJudge(provider)
    assert (await judge.agenerate("prompt", Verdict)).verdict == 1
    assert judge.calls == 1 and judge.retries == 0


async def test_judge_retries_malformed_output():
    provider = FakeProvider(["not json at all", '{"verdict": 0, "reason": "no"}'])
    judge = ProviderJudge(provider)
    assert (await judge.agenerate("prompt", Verdict)).verdict == 0
    assert judge.retries == 1


async def test_judge_doubles_the_budget_after_an_empty_reply():
    """The Phase 3 trap: thinking eats the budget and the answer comes back empty."""
    provider = FakeProvider(["", "", '{"verdict": 1}'])
    judge = ProviderJudge(provider, max_tokens=1000)
    assert (await judge.agenerate("prompt", Verdict)).verdict == 1
    assert provider.budgets == [1000, 2000, 4000]
    assert judge.empty_replies == 2


async def test_judge_gives_up_loudly():
    """Never silently score zero: an unusable judge raises."""
    provider = FakeProvider(["nope", "still nope", "nope again"])
    judge = ProviderJudge(provider)
    with pytest.raises(JudgeError):
        await judge.agenerate("prompt", Verdict)


async def test_judge_uses_no_openai_client(monkeypatch):
    """The whole point of the bridge: evaluation traffic stays on our provider."""
    import openai

    def explode(*args, **kwargs):
        raise AssertionError("an OpenAI client was constructed during evaluation")

    monkeypatch.setattr(openai, "OpenAI", explode)
    monkeypatch.setattr(openai, "AsyncOpenAI", explode)

    provider = FakeProvider(['{"verdict": 1}'])
    judge = ProviderJudge(provider)
    assert (await judge.agenerate("prompt", Verdict)).verdict == 1


# ---------------------------------------------------------------------------
# Refusal detection, aggregation, thresholds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "answer",
    [
        "The documents do not mention a mobile app.",
        # Contracted forms: the first version of the detector missed these and
        # scored three correct refusals as failures.
        "The documents don't contain any information about the CEO.",
        "The documents don\u2019t address self-hosting, so I can't answer that.",
        "That is not specified in the documents.",
        "I don't have any indexed documents that cover that. Try rephrasing.",
    ],
)
def test_declines_are_detected(answer):
    assert looks_declined(answer)


@pytest.mark.parametrize(
    "answer",
    [
        "Refunds are issued within 30 days of purchase [1].",
        "The Team plan allows 600 requests per minute [2].",
        "Support cannot reset two-factor authentication for a customer [3].",
    ],
)
def test_real_answers_are_not_declines(answer):
    assert not looks_declined(answer)


def _row(**kwargs) -> QuestionResult:
    base = dict(
        id="q",
        kind="answerable",
        question="q?",
        answer="a",
        reference="r",
        expected_sources=["handbook.md"],
    )
    base.update(kwargs)
    return QuestionResult(**base)


def test_summary_separates_out_of_scope_questions():
    results = [
        _row(id="a", faithfulness=1.0, context_recall=1.0, source_hit=True),
        _row(id="b", faithfulness=0.5, context_recall=0.0, source_hit=False),
        _row(id="c", kind="out_of_scope", expected_sources=[], declined=True),
        _row(id="d", kind="out_of_scope", expected_sources=[], declined=False),
    ]
    summary = summarise(results)

    assert summary["answerable"] == 2
    assert summary["out_of_scope"] == 2
    # Refusals are not folded into faithfulness: 0.75 is the mean of a and b only.
    assert summary["faithfulness"] == 0.75
    assert summary["context_recall"] == 0.5
    assert summary["source_accuracy"] == 0.5
    assert summary["refusal_rate"] == 0.5


def test_unscored_rows_do_not_drag_the_mean_down():
    """A judge that failed is a missing measurement, not a zero."""
    results = [
        _row(id="a", faithfulness=1.0),
        _row(id="b", faithfulness=None, unscored=["faithfulness: JudgeError"]),
    ]
    summary = summarise(results)
    assert summary["faithfulness"] == 1.0
    assert summary["faithfulness_scored"] == 1


def test_thresholds_fail_below_the_minimum():
    args = parse_args(["--min-faithfulness", "0.85", "--min-context-recall", "0.6"])
    summary = {"faithfulness": 0.80, "context_recall": 0.90}
    failures = check_thresholds(summary, args)
    assert len(failures) == 1
    assert "faithfulness" in failures[0]


def test_thresholds_pass_when_met():
    args = parse_args(["--min-faithfulness", "0.85"])
    assert check_thresholds({"faithfulness": 0.9}, args) == []


def test_unset_thresholds_are_not_checked():
    assert check_thresholds({"faithfulness": 0.1}, parse_args([])) == []


def test_nothing_scored_fails_the_threshold():
    """A run where every judgement failed must not pass by having no data."""
    args = parse_args(["--min-faithfulness", "0.85"])
    failures = check_thresholds({"faithfulness": None}, args)
    assert failures and "nothing scored" in failures[0]


def test_defaults_point_at_the_committed_dataset():
    args = parse_args([])
    assert args.dataset == DATASET
    assert args.corpus == CORPUS
    assert args.tenant == "evals"


def test_errors_fail_the_run_by_default():
    """A run where questions crashed must be red even if nothing was scored."""
    args = parse_args([])
    failures = check_thresholds({"errors": 3}, args)
    assert failures and "errors" in failures[0]

    assert check_thresholds({"errors": 3}, parse_args(["--max-errors", "3"])) == []
