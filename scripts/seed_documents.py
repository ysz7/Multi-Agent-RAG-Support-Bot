"""Copy the sample corpus into ./data/documents so the demo works on a clean clone.

    python -m scripts.seed_documents [--force]

The sample corpus lives in `evals/corpus` and is the same set the golden dataset
is written against — one corpus, not two, so a question that scores well in the
evaluation is a question the Quickstart can actually be tried with.

`data/documents` is git-ignored: it is where *your* documents go. This only ever
adds files, and refuses to overwrite an existing one unless asked, so running it
in a directory someone has already filled is safe.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from app.core.config import get_settings

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CORPUS = ROOT / "evals" / "corpus"


def seed(source: Path, destination: Path, *, force: bool = False) -> dict[str, list[str]]:
    """Copy sample files into `destination`. Returns what was copied and skipped."""
    if not source.is_dir():
        raise SystemExit(f"sample corpus not found: {source}")

    destination.mkdir(parents=True, exist_ok=True)
    result: dict[str, list[str]] = {"copied": [], "skipped": []}

    for path in sorted(source.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        target = destination / path.name
        if target.exists() and not force:
            result["skipped"].append(path.name)
            continue
        shutil.copy2(path, target)
        result["copied"].append(path.name)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SAMPLE_CORPUS)
    parser.add_argument("--path", type=Path, help="override DOCUMENTS_DIR")
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    args = parser.parse_args(argv)

    destination = args.path or get_settings().documents_dir
    result = seed(args.source, destination, force=args.force)

    for name in result["copied"]:
        print(f"copied  {name}")
    for name in result["skipped"]:
        print(f"skipped {name} (already there; --force to overwrite)")
    print(f"\n{len(result['copied'])} file(s) in {destination}.")
    print("Next: python -m scripts.index_documents")
    return 0


if __name__ == "__main__":
    sys.exit(main())
