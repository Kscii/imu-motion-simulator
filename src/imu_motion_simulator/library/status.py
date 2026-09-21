"""Read acquisition receipts and partial-download state without changing files."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

from .acquire import hashes


def receipt_hash(record: dict[str, object]) -> str | None:
    value = record.get("sha256") or record.get("observed_sha256")
    return value if isinstance(value, str) else None


def receipt_bytes(record: dict[str, object]) -> int | None:
    value = record.get("bytes") or record.get("observed_bytes")
    return value if isinstance(value, int) else None


def library_path(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("invalid library path")
    logical = PurePosixPath(value)
    if logical.is_absolute() or ".." in logical.parts or str(logical) != value:
        raise ValueError("library path escapes root")
    return root.joinpath(*logical.parts)


def inspect_collection(root: Path, job: dict[str, object], verify_content: bool) -> dict[str, object]:
    target = library_path(root, job["path"])
    receipt = library_path(root, job["receipt"])
    if not target.is_dir():
        return {"id": job["id"], "state": "missing"}
    if not receipt.is_file():
        return {"id": job["id"], "state": "unreceipted_final"}
    recorded = json.loads(receipt.read_text())
    files = recorded.get("files")
    if not isinstance(files, list):
        return {"id": job["id"], "state": "unreceipted_final"}
    total = 0
    for item in files:
        path = library_path(target, item["path"])
        if not path.is_file():
            return {"id": job["id"], "state": "missing", "missing_path": item["path"]}
        size = path.stat().st_size
        if size != item.get("bytes"):
            return {"id": job["id"], "state": "size_mismatch", "path": item["path"],
                    "bytes": size, "expected_bytes": item.get("bytes")}
        total += size
        if verify_content and hashes(path)["sha256"] != item.get("sha256"):
            return {"id": job["id"], "state": "hash_mismatch", "path": item["path"]}
    expected_total = job.get("bytes") or recorded.get("total_bytes")
    if expected_total is not None and total != expected_total:
        return {"id": job["id"], "state": "size_mismatch", "bytes": total,
                "expected_bytes": expected_total}
    return {"id": job["id"], "state": "verified", "bytes": total,
            "files": len(files), "check": "content_hashes" if verify_content else "receipt_and_size"}


def inspect_library(root: Path, manifest: Path | None = None, inventory: Path | None = None,
                    verify_content: bool = False) -> dict[str, object]:
    root = root.resolve()
    if manifest is not None and inventory is not None:
        raise ValueError("manifest and inventory are mutually exclusive")
    source = inventory or manifest
    payload = {} if source is None else json.loads(source.read_text())
    jobs = payload.get("items", []) if inventory is not None else payload.get("files", [])
    if not isinstance(jobs, list):
        raise ValueError("library status input must contain a list of files or items")
    states = []
    for job in jobs:
        if job.get("kind") == "collection":
            states.append(inspect_collection(root, job, verify_content))
            continue
        target = library_path(root, job["path"])
        receipt = library_path(root, job["receipt"]) if job.get("receipt") else target.with_name(target.name + ".receipt.json")
        partial = target.with_name(target.name + ".partial")
        if target.is_file() and receipt.is_file():
            recorded = json.loads(receipt.read_text())
            size = target.stat().st_size
            expected_size = job.get("bytes") or receipt_bytes(recorded)
            expected_hash = job.get("sha256") or receipt_hash(recorded)
            if expected_size is not None and size != expected_size:
                states.append({"id": job["id"], "state": "size_mismatch", "bytes": size,
                               "expected_bytes": expected_size})
            elif verify_content or manifest is not None:
                actual = hashes(target)
                state = "verified" if expected_hash == actual["sha256"] else "hash_mismatch"
                states.append({"id": job["id"], "state": state, "bytes": size, **actual})
            else:
                states.append({"id": job["id"], "state": "verified", "bytes": size,
                               "sha256": expected_hash, "check": "receipt_and_size"})
        elif target.exists():
            states.append({"id": job["id"], "state": "unreceipted_final"})
        elif partial.exists():
            states.append({"id": job["id"], "state": "partial", "bytes": partial.stat().st_size})
        else:
            states.append({"id": job["id"], "state": "missing"})
    event_status = root / "logs/acquisition-status.json"
    result = {
        "root": str(root),
        "manifest": str(manifest.resolve()) if manifest else None,
        "inventory": str(inventory.resolve()) if inventory else None,
        "content_hashes_checked": bool(verify_content or manifest is not None),
        "entries": states,
        "counts": {state: sum(row["state"] == state for row in states) for state in sorted({row["state"] for row in states})},
        "last_acquisition_status": json.loads(event_status.read_text()) if event_status.is_file() else None,
    }
    return result
