"""Phase 15: the sample corpus, so a clean clone has something to ask about.

`data/documents` is git-ignored — it is where a user's own documents go — so the
demo corpus is seeded from `evals/corpus` rather than committed twice. That makes
the Quickstart and the golden dataset share one set of documents: a question the
evaluation scores well is a question the Quickstart can be tried with.
"""

from pathlib import Path

from scripts.seed_documents import SAMPLE_CORPUS, main, seed

ROOT = Path(__file__).resolve().parent.parent


def test_the_sample_corpus_is_the_evaluation_corpus():
    assert SAMPLE_CORPUS == ROOT / "evals" / "corpus"
    assert list(SAMPLE_CORPUS.glob("*.md")), "the sample corpus is empty"


def test_seed_copies_the_corpus(tmp_path):
    destination = tmp_path / "documents"
    result = seed(SAMPLE_CORPUS, destination)

    expected = {p.name for p in SAMPLE_CORPUS.iterdir() if p.is_file()}
    assert set(result["copied"]) == expected
    assert {p.name for p in destination.iterdir()} == expected


def test_seed_never_clobbers_someone_elses_documents(tmp_path):
    """Run in a directory a user has already filled, it must not overwrite."""
    destination = tmp_path / "documents"
    destination.mkdir()
    mine = destination / "handbook.md"
    mine.write_text("my own notes")

    result = seed(SAMPLE_CORPUS, destination)

    assert "handbook.md" in result["skipped"]
    assert mine.read_text() == "my own notes"

    seed(SAMPLE_CORPUS, destination, force=True)
    assert mine.read_text() != "my own notes"


def test_seed_is_idempotent(tmp_path):
    destination = tmp_path / "documents"
    seed(SAMPLE_CORPUS, destination)
    second = seed(SAMPLE_CORPUS, destination)
    assert second["copied"] == []


def test_missing_corpus_fails_loudly(tmp_path):
    import pytest

    with pytest.raises(SystemExit):
        seed(tmp_path / "nope", tmp_path / "out")


def test_cli_writes_into_the_given_path(tmp_path, capsys):
    assert main(["--path", str(tmp_path / "docs")]) == 0
    assert (tmp_path / "docs" / "handbook.md").exists()
    assert "index_documents" in capsys.readouterr().out
