#!/usr/bin/env python3
"""
Validate /auto_examples/images references used by docs and example scripts.

Checks:
1. Source artifacts exist under docs/source/auto_examples/images/...
2. Built artifacts exist under docs/build/html/auto_examples/images/...
   (or a custom build directory if provided)
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

REF_PATTERN = re.compile(r"/auto_examples/images/(?:thumb/)?[^\s'\"`<>)]+")


@dataclass(frozen=True)
class RefOccurrence:
    file_path: Path
    line_number: int
    ref_path: str


def iter_examples(example_dir: Path) -> Iterable[Path]:
    return sorted(example_dir.glob("*.py"))


def iter_rst_files(docs_source_dir: Path) -> Iterable[Path]:
    return sorted(docs_source_dir.rglob("*.rst"))


def extract_refs(file_path: Path) -> List[RefOccurrence]:
    occurrences: List[RefOccurrence] = []
    text = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for idx, line in enumerate(text, start=1):
        for match in REF_PATTERN.finditer(line):
            occurrences.append(
                RefOccurrence(
                    file_path=file_path,
                    line_number=idx,
                    ref_path=match.group(0),
                )
            )
    return occurrences


def collect_occurrences(repo_root: Path) -> List[RefOccurrence]:
    examples_dir = repo_root / "examples"
    docs_source_dir = repo_root / "docs" / "source"

    files: List[Path] = []
    files.extend(iter_examples(examples_dir))
    files.extend(iter_rst_files(docs_source_dir))

    occurrences: List[RefOccurrence] = []
    for file_path in files:
        occurrences.extend(extract_refs(file_path))
    return occurrences


def normalize_ref(ref_path: str) -> Path:
    return Path(ref_path.lstrip("/"))


def exists_case_sensitive(path: Path) -> bool:
    """Check file existence with exact case matching for each path component."""
    abs_path = path if path.is_absolute() else (Path.cwd() / path)
    if not abs_path.exists():
        return False

    if not abs_path.anchor:
        return False

    current = Path(abs_path.anchor)
    for part in abs_path.relative_to(abs_path.anchor).parts:
        if not current.is_dir():
            return False
        children = {child.name for child in current.iterdir()}
        if part not in children:
            return False
        current = current / part

    return current.is_file()


def validate_source_paths(
    occurrences: Sequence[RefOccurrence],
    docs_source_dir: Path,
) -> List[RefOccurrence]:
    missing: List[RefOccurrence] = []
    for occ in occurrences:
        expected = docs_source_dir / normalize_ref(occ.ref_path)
        if not exists_case_sensitive(expected):
            missing.append(occ)
    return missing


def validate_built_paths(
    occurrences: Sequence[RefOccurrence],
    build_html_dir: Path,
) -> List[RefOccurrence]:
    missing: List[RefOccurrence] = []
    for occ in occurrences:
        expected = build_html_dir / normalize_ref(occ.ref_path)
        if not exists_case_sensitive(expected):
            missing.append(occ)
    return missing


def print_missing(
    title: str,
    missing: Sequence[RefOccurrence],
    repo_root: Path,
) -> None:
    if not missing:
        return
    print(title)
    print("-" * len(title))
    for occ in missing:
        rel_file = occ.file_path.relative_to(repo_root)
        print(f"{rel_file}:{occ.line_number} -> {occ.ref_path}")
    print()


def unique_occurrences(occurrences: Sequence[RefOccurrence]) -> List[RefOccurrence]:
    seen = set()
    unique: List[RefOccurrence] = []
    for occ in occurrences:
        key = (occ.file_path, occ.line_number, occ.ref_path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(occ)
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check docs/example image references.")
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repository root (defaults to parent of this script directory).",
    )
    parser.add_argument(
        "--build-dir",
        default="docs/build/html",
        help="Built HTML directory used for build artifact checks.",
    )
    parser.add_argument(
        "--require-build",
        action="store_true",
        help="Fail if build directory does not exist before checking built artifacts.",
    )
    parser.add_argument(
        "--skip-build-check",
        action="store_true",
        help="Only check source artifacts under docs/source.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    repo_root = (
        Path(args.repo_root).resolve()
        if args.repo_root
        else script_dir.parent.resolve()
    )
    docs_source_dir = repo_root / "docs" / "source"
    build_html_dir = (repo_root / args.build_dir).resolve()

    if not docs_source_dir.is_dir():
        print(f"ERROR: docs/source directory not found: {docs_source_dir}")
        return 1

    occurrences = unique_occurrences(collect_occurrences(repo_root))
    print(f"Found {len(occurrences)} /auto_examples/images references.")

    source_missing = validate_source_paths(occurrences, docs_source_dir)
    print_missing("Missing source artifacts", source_missing, repo_root)

    build_missing: List[RefOccurrence] = []
    if not args.skip_build_check:
        if not build_html_dir.is_dir():
            if args.require_build:
                print(f"ERROR: Build directory does not exist: {build_html_dir}")
                return 1
            print(f"Build directory not found; skipping build artifact checks: {build_html_dir}")
        else:
            build_missing = validate_built_paths(occurrences, build_html_dir)
            print_missing("Missing built artifacts", build_missing, repo_root)

    if source_missing or build_missing:
        print(
            f"FAIL: missing source={len(source_missing)}, "
            f"missing build={len(build_missing)}"
        )
        return 1

    print("PASS: all referenced source/build image artifacts were found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
