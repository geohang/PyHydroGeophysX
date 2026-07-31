#!/usr/bin/env python3
"""
Validate /auto_examples/images references used by docs and example scripts.

Checks:
1. Source artifacts exist under docs/source/auto_examples/images/...
2. Every Sphinx-Gallery example includes at least one output-image directive.
3. Built artifacts exist under docs/build/html/auto_examples/images/...
   (or a custom build directory if provided)
4. Each built gallery page embeds at least one of its declared output images.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

REF_PATTERN = re.compile(r"/auto_examples/images/(?:thumb/)?[^\s'\"`<>)]+")
GALLERY_EXAMPLE_PATTERN = re.compile(r"^(?:Ex|EX)_.*\.py$")


@dataclass(frozen=True)
class RefOccurrence:
    file_path: Path
    line_number: int
    ref_path: str


def iter_examples(example_dir: Path) -> Iterable[Path]:
    return sorted(example_dir.glob("*.py"))


def iter_gallery_examples(example_dir: Path) -> Iterable[Path]:
    return sorted(
        path
        for path in example_dir.glob("*.py")
        if GALLERY_EXAMPLE_PATTERN.match(path.name)
    )


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


def gallery_examples_without_images(example_dir: Path) -> List[Path]:
    """Return gallery examples that declare no output images."""
    return [
        path
        for path in iter_gallery_examples(example_dir)
        if not extract_refs(path)
    ]


def gallery_pages_without_embedded_images(
    example_dir: Path,
    build_html_dir: Path,
) -> List[Tuple[Path, Path]]:
    """Return built gallery pages that omit their declared output images."""
    missing: List[Tuple[Path, Path]] = []
    gallery_dir = build_html_dir / "auto_examples"

    for example_path in iter_gallery_examples(example_dir):
        html_path = gallery_dir / f"{example_path.stem}.html"
        if not html_path.is_file():
            missing.append((example_path, html_path))
            continue

        declared_names = {
            normalize_ref(occurrence.ref_path).name
            for occurrence in extract_refs(example_path)
        }
        html = html_path.read_text(encoding="utf-8", errors="ignore")
        if not any(name in html for name in declared_names):
            missing.append((example_path, html_path))

    return missing


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
    alternate_build_dirs: Sequence[Path] = (),
) -> Tuple[List[RefOccurrence], List[Tuple[RefOccurrence, Path]]]:
    missing: List[RefOccurrence] = []
    resolved_in_alternate: List[Tuple[RefOccurrence, Path]] = []
    for occ in occurrences:
        normalized = normalize_ref(occ.ref_path)
        candidates_primary = [
            build_html_dir / normalized,
            build_html_dir / "_images" / normalized.name,
        ]
        if any(exists_case_sensitive(candidate) for candidate in candidates_primary):
            continue

        resolved = False
        for alt_dir in alternate_build_dirs:
            candidates_alt = [
                alt_dir / normalized,
                alt_dir / "_images" / normalized.name,
            ]
            if any(exists_case_sensitive(candidate) for candidate in candidates_alt):
                resolved_in_alternate.append((occ, alt_dir))
                resolved = True
                break

        if not resolved:
            missing.append(occ)
    return missing, resolved_in_alternate


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
    examples_dir = repo_root / "examples"
    build_html_dir = (repo_root / args.build_dir).resolve()

    if not docs_source_dir.is_dir():
        print(f"ERROR: docs/source directory not found: {docs_source_dir}")
        return 1

    occurrences = unique_occurrences(collect_occurrences(repo_root))
    print(f"Found {len(occurrences)} /auto_examples/images references.")

    gallery_missing_refs = gallery_examples_without_images(examples_dir)
    if gallery_missing_refs:
        print("Gallery examples without output images")
        print("--------------------------------------")
        for path in gallery_missing_refs:
            print(path.relative_to(repo_root))
        print()
    else:
        gallery_count = len(list(iter_gallery_examples(examples_dir)))
        print(f"PASS: all {gallery_count} gallery examples declare output images.")

    source_missing = validate_source_paths(occurrences, docs_source_dir)
    print_missing("Missing source artifacts", source_missing, repo_root)

    build_missing: List[RefOccurrence] = []
    resolved_in_alt: List[Tuple[RefOccurrence, Path]] = []
    gallery_pages_missing_images: List[Tuple[Path, Path]] = []
    if not args.skip_build_check:
        alt_build_dirs = [
            (repo_root / "docs" / "_build" / "html").resolve(),
            (repo_root / "docs" / "build" / "html").resolve(),
        ]
        alt_build_dirs = [
            path for path in alt_build_dirs
            if path != build_html_dir and path.is_dir()
        ]

        if not build_html_dir.is_dir():
            if args.require_build:
                print(f"ERROR: Build directory does not exist: {build_html_dir}")
                return 1
            print(f"Build directory not found; skipping build artifact checks: {build_html_dir}")
        else:
            build_missing, resolved_in_alt = validate_built_paths(
                occurrences,
                build_html_dir,
                alternate_build_dirs=alt_build_dirs,
            )
            print_missing("Missing built artifacts", build_missing, repo_root)

            gallery_pages_missing_images = gallery_pages_without_embedded_images(
                examples_dir,
                build_html_dir,
            )
            if gallery_pages_missing_images:
                print("Built gallery pages without declared output images")
                print("--------------------------------------------------")
                for example_path, html_path in gallery_pages_missing_images:
                    print(
                        f"{example_path.relative_to(repo_root)} -> "
                        f"{html_path.relative_to(repo_root)}"
                    )
                print()
            else:
                print("PASS: every built gallery page embeds a declared output image.")

            if resolved_in_alt:
                alt_dirs_used = sorted({path for _, path in resolved_in_alt})
                alt_dirs_display = ", ".join(str(path) for path in alt_dirs_used)
                print(
                    "Warning: some built artifacts were resolved from alternate build "
                    f"directory/directories: {alt_dirs_display}"
                )
                print(
                    "Hint: align your sphinx-build output path with --build-dir for "
                    "deterministic CI checks."
                )

    if (
        gallery_missing_refs
        or source_missing
        or build_missing
        or gallery_pages_missing_images
    ):
        print(
            f"FAIL: gallery without images={len(gallery_missing_refs)}, "
            f"missing source={len(source_missing)}, "
            f"missing build={len(build_missing)}, "
            f"built pages without images={len(gallery_pages_missing_images)}"
        )
        return 1

    print("PASS: all referenced source/build image artifacts were found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
