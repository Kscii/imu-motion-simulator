"""Create a local preset-labelled full core H5 after independent work ends.

This command reads the formally audited default H5 and writes another HDF5 3.3
under ``--output-root``. It does not access GCS or change the platform rules.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import h5py

from imu_motion_simulator.contracts.common import get_json
from imu_motion_simulator.local_relabel import relabel_local_core
from imu_motion_simulator.provisional import validate_rules
from imu_motion_simulator.publication import _json_bytes


def _save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _source_h5(source_dir: Path, audit_sha: str, expected: int) -> Path | None:
    matched = []
    for path in source_dir.glob("*/synthetic-provisional-*.h5"):
        with h5py.File(path, "r") as handle:
            metadata = get_json(handle, "provenance/metadata")
            if (metadata.get("prod_audit_sha256") == audit_sha
                    and len(handle["candidate_index"]) == expected):
                matched.append(path)
    if len(matched) > 1:
        raise RuntimeError("Multiple full audited source H5 files match")
    return matched[0] if matched else None


def _gate(audit: Path, benchmark: Path, source_dir: Path, expected: int) \
        -> tuple[str, Path | None]:
    report = json.loads(audit.read_text())
    if report.get("phase") == "failed":
        raise RuntimeError("Formal prod cloud audit failed")
    if report.get("phase") != "complete":
        return "audit " + str(report.get("phase")), None
    state = json.loads(benchmark.read_text())
    if state.get("phase") == "failed":
        raise RuntimeError("Isolated production benchmark failed")
    if state.get("phase") != "complete":
        return "benchmark " + str(state.get("phase")), None
    audit_sha = hashlib.sha256(audit.read_bytes()).hexdigest()
    source = _source_h5(source_dir, audit_sha, expected)
    return ("ready", source) if source is not None else ("full source H5 pending", None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--benchmark-status", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--wait-hours", type=float, default=24)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.wait_hours <= 0:
        parser.error("A positive wait window is required")
    rules = validate_rules(json.loads(args.rules.read_text()))
    preview = json.loads(args.preview.read_text())
    if (rules["revision"] < 2 or preview["candidate_count"] != 15569
            or hashlib.sha256(_json_bytes(rules)).hexdigest()
               != preview["rules_sha256"]):
        raise ValueError("Frozen local preset preview does not match all candidates")
    reason, source = _gate(args.audit, args.benchmark_status, args.source_dir,
                           preview["candidate_count"])
    if args.check_only:
        print(json.dumps({"gate": reason, "source_h5": str(source) if source else None,
                          "candidate_count": preview["candidate_count"],
                          "weak_count": preview["weak_count"]}))
        return
    status_path = args.output_root / "status.json"
    deadline = time.monotonic() + args.wait_hours * 3600
    try:
        while reason != "ready":
            _save(status_path, {"phase": "waiting", "reason": reason,
                                "updated_at_utc": datetime.now(UTC).isoformat()})
            if time.monotonic() >= deadline:
                raise TimeoutError("Full audited H5 or benchmark did not finish: " + reason)
            time.sleep(60)
            reason, source = _gate(args.audit, args.benchmark_status,
                                   args.source_dir, preview["candidate_count"])
        assert source is not None
        _save(status_path, {"phase": "relabeling", "source_h5": str(source),
                            "rules_sha256": preview["rules_sha256"]})
        manifest = relabel_local_core(source, rules, args.output_root,
                                      audit_report=args.audit,
                                      expected_candidates=preview["candidate_count"])
        if (manifest["weak_count"] != preview["weak_count"]
                or manifest["unresolved_count"] != preview["unresolved_count"]):
            raise RuntimeError("Local H5 labels differ from frozen full-corpus preview")
        _save(status_path, {"phase": "complete",
                            "export_id": manifest["export_id"],
                            "h5_path": manifest["h5"]["path"],
                            "h5_sha256": manifest["h5"]["sha256"],
                            "candidate_count": manifest["candidate_count"],
                            "weak_count": manifest["weak_count"],
                            "unresolved_count": manifest["unresolved_count"],
                            "completed_at_utc": datetime.now(UTC).isoformat()})
        print(json.dumps(manifest, ensure_ascii=False))
    except Exception as error:
        _save(status_path, {"phase": "failed", "error": str(error),
                            "failed_at_utc": datetime.now(UTC).isoformat()})
        raise


if __name__ == "__main__":
    main()
