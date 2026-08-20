"""Loading and chunking source documents.

Splitting is structure-aware rather than a blind character window: Markdown
splits on heading boundaries (and carries the heading path into metadata so a
citation can say *where* in the document it came from), PDFs split per page,
and plain text splits on blank lines. Only when a single section is still too
large does it fall back to a sliding window.

Metadata written here is what the retriever later returns as a citation, so it
has to survive the round trip: source path, section or page, chunk index.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".txt", ".text"}

_ATX_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")


class UnsupportedDocument(ValueError):
    """File type this pipeline does not know how to read."""


@dataclass(frozen=True, slots=True)
class LoadedDocument:
    source_path: str
    title: str
    text: str
    content_hash: str
    metadata: dict = field(default_factory=dict)
    # Present only for PDFs: (page_number, page_text), 1-indexed.
    pages: tuple[tuple[int, str], ...] = ()


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_index: int
    content: str
    metadata: dict


def content_hash(data: bytes) -> str:
    """Stable identity for a file's bytes; drives idempotent re-indexing."""
    return hashlib.sha256(data).hexdigest()


# --- loading ---------------------------------------------------------------


def load_document(path: Path) -> LoadedDocument:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedDocument(f"{path.name}: unsupported type {suffix!r}")

    raw = path.read_bytes()
    digest = content_hash(raw)

    if suffix == ".pdf":
        pages = _read_pdf(path)
        text = "\n\n".join(page_text for _, page_text in pages)
        return LoadedDocument(
            source_path=str(path),
            title=path.stem,
            text=text,
            content_hash=digest,
            metadata={"type": "pdf", "page_count": len(pages)},
            pages=tuple(pages),
        )

    text = raw.decode("utf-8", errors="replace")
    doc_type = "markdown" if suffix in {".md", ".markdown"} else "text"
    return LoadedDocument(
        source_path=str(path),
        title=_markdown_title(text) if doc_type == "markdown" else path.stem,
        text=text,
        content_hash=digest,
        metadata={"type": doc_type},
    )


def _read_pdf(path: Path) -> list[tuple[int, str]]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for number, page in enumerate(reader.pages, start=1):
        extracted = (page.extract_text() or "").strip()
        if extracted:
            pages.append((number, extracted))
    return pages


def _markdown_title(text: str) -> str:
    for line in text.splitlines():
        if match := _ATX_HEADING_RE.match(line.strip()):
            return match.group(2).strip()
    return ""


def discover_documents(root: Path) -> list[Path]:
    """Every supported file under `root`, sorted for deterministic runs."""
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )


# --- splitting -------------------------------------------------------------


def split_markdown_sections(text: str) -> list[tuple[str, str]]:
    """Split on ATX headings into (heading_path, body) pairs.

    The heading path is breadcrumb-style ("Guide > Setup > Install") so a chunk
    keeps its position in the document hierarchy.
    """
    sections: list[tuple[str, str]] = []
    stack: list[str] = []
    current: list[str] = []
    heading_path = ""

    def flush() -> None:
        body = "\n".join(current).strip()
        if body:
            sections.append((heading_path, body))

    for line in text.splitlines():
        match = _ATX_HEADING_RE.match(line.strip())
        if match:
            flush()
            current = []
            level = len(match.group(1))
            title = match.group(2).strip()
            stack = stack[: level - 1]
            stack.append(title)
            heading_path = " > ".join(stack)
        else:
            current.append(line)

    flush()
    return sections


def sliding_window(text: str, size: int, overlap: int) -> list[str]:
    """Last-resort split for a block that is still oversized.

    Prefers to break on a paragraph, then a sentence, then a word, so chunks end
    at a natural boundary instead of mid-token.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    text = text.strip()
    if len(text) <= size:
        return [text] if text else []

    windows: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            for separator in ("\n\n", ". ", "\n", " "):
                cut = window.rfind(separator)
                # Only honour a boundary in the last third, else we waste the window.
                if cut > size // 3:
                    end = start + cut + len(separator)
                    break
        chunk = text[start:end].strip()
        if chunk:
            windows.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return windows


def _pack_blocks(blocks: list[str], size: int, overlap: int) -> list[str]:
    """Greedily merge small blocks up to `size`, splitting any oversized one."""
    packed: list[str] = []
    buffer = ""
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if len(block) > size:
            if buffer:
                packed.append(buffer)
                buffer = ""
            packed.extend(sliding_window(block, size, overlap))
            continue
        candidate = f"{buffer}\n\n{block}" if buffer else block
        if len(candidate) <= size:
            buffer = candidate
        else:
            packed.append(buffer)
            buffer = block
    if buffer:
        packed.append(buffer)
    return packed


def chunk_document(document: LoadedDocument, *, size: int, overlap: int) -> list[Chunk]:
    """Split a loaded document, attaching citation metadata to every chunk."""
    if overlap >= size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    doc_type = document.metadata.get("type")
    pieces: list[tuple[str, dict]] = []

    if doc_type == "pdf":
        for page_number, page_text in document.pages:
            for body in _pack_blocks(_PARAGRAPH_SPLIT_RE.split(page_text), size, overlap):
                pieces.append((body, {"page": page_number}))
    elif doc_type == "markdown":
        for heading_path, body in split_markdown_sections(document.text):
            for part in _pack_blocks(_PARAGRAPH_SPLIT_RE.split(body), size, overlap):
                pieces.append((part, {"section": heading_path} if heading_path else {}))
    else:
        for body in _pack_blocks(_PARAGRAPH_SPLIT_RE.split(document.text), size, overlap):
            pieces.append((body, {}))

    return [
        Chunk(
            chunk_index=index,
            content=content,
            metadata={
                "source_path": document.source_path,
                "title": document.title,
                "type": doc_type,
                **extra,
            },
        )
        for index, (content, extra) in enumerate(pieces)
    ]
