"""Small, stable documentation-policy and local-link checker."""

from __future__ import annotations

import re
from pathlib import Path


ROOT_FILES = {"AGENTS.md", "README.md", "STATUS.md"}
DOC_FILES = {
    "docs/project.md",
    "docs/architecture.md",
    "docs/decisions.md",
    "docs/development.md",
    "docs/data-assets.md",
    "docs/productization-roadmap.md",
    "docs/contracts/kinematic-v2.md",
    "docs/contracts/imu-hdf5-v3.3.md",
}
FORBIDDEN_ROOT_FILES = {"TODO.md"}
LINK = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")


def check(root: Path) -> dict[str, object]:
    root = root.resolve()
    errors: list[str] = []
    expected = ROOT_FILES | DOC_FILES
    for relative in sorted(expected):
        if not (root / relative).is_file():
            errors.append(f"missing required document: {relative}")
    for relative in sorted(FORBIDDEN_ROOT_FILES):
        if (root / relative).exists():
            errors.append(f"retired root document still present: {relative}")
    actual_docs = {
        str(path.relative_to(root))
        for path in (root / "docs").rglob("*.md")
        if path.is_file()
    }
    if actual_docs != DOC_FILES:
        for relative in sorted(actual_docs - DOC_FILES):
            errors.append(f"unexpected active document: {relative}")

    total_lines = 0
    for relative in sorted(expected):
        path = root / relative
        if not path.is_file():
            continue
        content = path.read_text()
        lines = content.count("\n") + (not content.endswith("\n"))
        total_lines += lines
        limit = 220 if relative in ROOT_FILES else 320
        if lines > limit:
            errors.append(f"document exceeds {limit} lines: {relative} ({lines})")
        if re.search(r"^#{1,3}\s+D\d{2}\b", content, flags=re.MULTILINE):
            errors.append(f"chronological phase heading in active document: {relative}")
        for raw_target in LINK.findall(content):
            target = raw_target.strip().split("#", 1)[0]
            if not target or "://" in target or target.startswith(("mailto:", "urn:")):
                continue
            target_path = Path(target)
            if target_path.is_absolute():
                continue
            if not (path.parent / target_path).resolve().exists():
                errors.append(f"broken local link in {relative}: {raw_target}")
    if total_lines > 2200:
        errors.append(f"active documentation exceeds 2200 lines: {total_lines}")
    return {
        "passed": not errors,
        "documents": len(expected),
        "total_lines": total_lines,
        "errors": errors,
    }
