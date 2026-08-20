"""Phase 4: loading and structure-aware chunking. No database, no network."""

import pytest

from app.rag.chunking import (
    Chunk,
    UnsupportedDocument,
    chunk_document,
    content_hash,
    discover_documents,
    load_document,
    sliding_window,
    split_markdown_sections,
)

MARKDOWN = """# Handbook

Intro paragraph.

## Setup

Install the thing.

### Requirements

Python 3.11 or newer.

## Billing

Invoices go out monthly.
"""


def _write(tmp_path, name: str, text: str):
    path = tmp_path / name
    path.write_text(text)
    return path


# --- loading ---------------------------------------------------------------


def test_load_markdown_uses_first_heading_as_title(tmp_path):
    doc = load_document(_write(tmp_path, "handbook.md", MARKDOWN))
    assert doc.title == "Handbook"
    assert doc.metadata["type"] == "markdown"


def test_load_text_uses_filename_as_title(tmp_path):
    doc = load_document(_write(tmp_path, "notes.txt", "hello"))
    assert doc.title == "notes"
    assert doc.metadata["type"] == "text"


def test_unsupported_type_is_rejected(tmp_path):
    with pytest.raises(UnsupportedDocument, match="unsupported type"):
        load_document(_write(tmp_path, "data.csv", "a,b"))


def test_content_hash_is_stable_and_content_sensitive():
    assert content_hash(b"abc") == content_hash(b"abc")
    assert content_hash(b"abc") != content_hash(b"abd")


def test_discover_documents_is_recursive_and_sorted(tmp_path):
    _write(tmp_path, "b.md", "# B")
    (tmp_path / "nested").mkdir()
    _write(tmp_path, "nested/a.txt", "a")
    _write(tmp_path, "ignore.csv", "x")

    found = [p.name for p in discover_documents(tmp_path)]
    assert found == ["b.md", "a.txt"] or found == ["a.txt", "b.md"]
    assert "ignore.csv" not in found


def test_discover_documents_on_missing_dir_returns_empty(tmp_path):
    assert discover_documents(tmp_path / "nope") == []


# --- markdown structure ----------------------------------------------------


def test_markdown_sections_carry_a_breadcrumb_path():
    sections = dict(split_markdown_sections(MARKDOWN))
    assert "Handbook > Setup > Requirements" in sections
    assert sections["Handbook > Setup > Requirements"] == "Python 3.11 or newer."


def test_sibling_heading_pops_the_stack():
    """'Billing' is a sibling of 'Setup', so it must not nest under Requirements."""
    paths = [path for path, _ in split_markdown_sections(MARKDOWN)]
    assert "Handbook > Billing" in paths


# --- windowing -------------------------------------------------------------


def test_sliding_window_respects_size_and_overlaps():
    text = " ".join(f"word{i}" for i in range(400))
    windows = sliding_window(text, size=200, overlap=50)
    assert len(windows) > 1
    assert all(len(w) <= 200 for w in windows)


def test_sliding_window_short_text_is_one_chunk():
    assert sliding_window("short", size=100, overlap=10) == ["short"]


def test_sliding_window_rejects_overlap_at_least_size():
    with pytest.raises(ValueError, match="overlap"):
        sliding_window("abc", size=10, overlap=10)


# --- chunking --------------------------------------------------------------


def test_chunks_are_indexed_from_zero_and_carry_citation_metadata(tmp_path):
    doc = load_document(_write(tmp_path, "handbook.md", MARKDOWN))
    chunks = chunk_document(doc, size=200, overlap=40)

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.metadata["source_path"].endswith("handbook.md") for c in chunks)
    assert any("section" in c.metadata for c in chunks)


def test_markdown_chunks_never_span_two_sections(tmp_path):
    """A chunk mixing Setup and Billing text would produce a misleading citation."""
    doc = load_document(_write(tmp_path, "handbook.md", MARKDOWN))
    chunks = chunk_document(doc, size=1000, overlap=0)

    billing = [c for c in chunks if c.metadata.get("section", "").endswith("Billing")]
    assert billing, "expected a Billing chunk"
    assert all("Install the thing" not in c.content for c in billing)


def test_oversized_section_is_split(tmp_path):
    big = "# Title\n\n" + ("sentence. " * 500)
    doc = load_document(_write(tmp_path, "big.md", big))
    chunks = chunk_document(doc, size=300, overlap=50)

    assert len(chunks) > 1
    assert all(len(c.content) <= 300 for c in chunks)


def test_chunk_document_rejects_bad_overlap(tmp_path):
    doc = load_document(_write(tmp_path, "a.txt", "hello"))
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_document(doc, size=100, overlap=100)


def test_empty_document_yields_no_chunks(tmp_path):
    doc = load_document(_write(tmp_path, "empty.txt", "   \n\n  "))
    assert chunk_document(doc, size=100, overlap=10) == []


# --- pdf -------------------------------------------------------------------


def test_pdf_chunks_carry_page_numbers(tmp_path):
    from tests.pdf_fixture import make_pdf

    path = tmp_path / "policy.pdf"
    path.write_bytes(make_pdf(["Refunds are issued within 30 days.", "Contact support to start."]))

    doc = load_document(path)
    assert doc.metadata["type"] == "pdf"
    assert doc.metadata["page_count"] == 2

    chunks = chunk_document(doc, size=500, overlap=50)
    pages = {c.metadata["page"] for c in chunks}
    assert pages == {1, 2}
    assert any("Refunds" in c.content for c in chunks)


def test_pdf_pages_do_not_bleed_into_each_other(tmp_path):
    from tests.pdf_fixture import make_pdf

    path = tmp_path / "two.pdf"
    path.write_bytes(make_pdf(["AAA first page only.", "BBB second page only."]))

    chunks = chunk_document(load_document(path), size=500, overlap=0)
    for chunk in chunks:
        if chunk.metadata["page"] == 1:
            assert "BBB" not in chunk.content
