"""Index ./data/documents into the vector store.

    python -m scripts.index_documents [--tenant TENANT] [--prune] [--dry-run]

Idempotent: re-running over unchanged files does nothing. Edited files are
replaced. `--prune` also removes documents whose source file has been deleted.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from app.core.config import Settings, get_settings
from app.core.llm_provider import LLMError, get_embedding_provider
from app.rag.chunking import (
    UnsupportedDocument,
    chunk_document,
    discover_documents,
    load_document,
)
from app.rag.retrievers.pgvector import PgVectorStore

logger = logging.getLogger("index")

EMBED_BATCH_SIZE = 32


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant", help="tenant id to index under (default: LOCAL_TENANT_ID)")
    parser.add_argument("--path", type=Path, help="override DOCUMENTS_DIR")
    parser.add_argument(
        "--prune", action="store_true", help="delete indexed docs whose source file is gone"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="report what would change, write nothing"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


async def _embed_in_batches(provider, texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        vectors.extend(await provider.embed(texts[start : start + EMBED_BATCH_SIZE]))
    return vectors


async def index_documents(
    settings: Settings, *, tenant_id: str, root: Path, prune: bool, dry_run: bool
) -> dict[str, int]:
    paths = discover_documents(root)
    if not paths:
        logger.warning("no supported documents found under %s", root)
        return {"files": 0, "inserted": 0, "replaced": 0, "unchanged": 0, "chunks": 0, "pruned": 0}

    store = PgVectorStore(settings)
    embedder = get_embedding_provider(settings)
    totals = dict.fromkeys(("files", "inserted", "replaced", "unchanged", "chunks", "pruned"), 0)

    connection = await store.connect()
    try:
        for path in paths:
            try:
                document = load_document(path)
            except UnsupportedDocument as exc:
                logger.warning("skipping %s", exc)
                continue

            chunks = chunk_document(
                document, size=settings.chunk_size, overlap=settings.chunk_overlap
            )
            totals["files"] += 1

            if dry_run:
                logger.info("would index %s (%d chunks)", path.name, len(chunks))
                continue

            # Embedding is the expensive step, so skip it when the bytes are
            # already indexed under this tenant.
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "SELECT 1 FROM documents WHERE tenant_id = %s AND content_hash = %s",
                    (tenant_id, document.content_hash),
                )
                already_indexed = await cursor.fetchone() is not None

            if already_indexed:
                totals["unchanged"] += 1
                logger.info("unchanged  %s", path.name)
                continue

            embeddings = await _embed_in_batches(embedder, [c.content for c in chunks])
            result = await store.upsert_document(
                connection,
                tenant_id=tenant_id,
                document=document,
                chunks=chunks,
                embeddings=embeddings,
            )
            await connection.commit()

            totals[result.outcome.value] = totals.get(result.outcome.value, 0) + 1
            totals["chunks"] += result.chunk_count
            logger.info("%-10s %s (%d chunks)", result.outcome.value, path.name, result.chunk_count)

        if prune and not dry_run:
            removed = await store.delete_missing(
                connection, tenant_id=tenant_id, keep_paths={str(p) for p in paths}
            )
            await connection.commit()
            totals["pruned"] = removed
            if removed:
                logger.info("pruned %d document(s) with no source file", removed)
    finally:
        await connection.close()
        await embedder.aclose()

    return totals


async def _main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    settings = get_settings()
    tenant_id = args.tenant or settings.local_tenant_id
    root = args.path or settings.documents_dir

    try:
        totals = await index_documents(
            settings, tenant_id=tenant_id, root=root, prune=args.prune, dry_run=args.dry_run
        )
    except LLMError as exc:
        logger.error("embedding failed: %s", exc)
        return 1
    except Exception as exc:  # surface the cause without a traceback wall
        logger.error("indexing failed: %s: %s", type(exc).__name__, exc)
        return 1

    logger.info(
        "\ntenant=%s  files=%d  inserted=%d  replaced=%d  unchanged=%d  chunks=%d  pruned=%d",
        tenant_id,
        totals["files"],
        totals["inserted"],
        totals["replaced"],
        totals["unchanged"],
        totals["chunks"],
        totals["pruned"],
    )
    return 0


def main() -> None:
    sys.exit(asyncio.run(_main()))


if __name__ == "__main__":
    main()
