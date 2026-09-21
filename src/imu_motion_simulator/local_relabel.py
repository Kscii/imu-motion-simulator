"""Re-label an existing provisional core H5 without cloud publication.

The original IMU samples and candidate identities are copied unchanged. Only
the separate weak-label index and its frozen provenance change. This remains
unreviewed synthetic data, not a formal training snapshot.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import h5py

from .contracts.common import atomic_h5, get_json, put_json, sha256_file, text
from .contracts.core import logical_content_sha256
from .contracts.delivery import WEAK_LABEL_INDEX, table, validate_delivery
from .provisional import validate_rules, weak_label
from .publication import _json_bytes

SCHEMA = "imu_motion_simulator.local_provisional_relabel.v1"


def relabel_local_core(source_h5: Path, rules: dict, output_root: Path, *,
                       audit_report: Path | None = None,
                       expected_candidates: int | None = None) -> dict:
    """Write a new immutable local HDF5 3.3 from an existing provisional H5."""
    source_h5, output_root = Path(source_h5), Path(output_root)
    rules = validate_rules(rules)
    if rules["revision"] < 2:
        raise ValueError("Local relabel requires opt-in preset revision 2 or newer")
    source_report = validate_delivery(source_h5)
    if source_report["profile"] != "imu_dataset_provisional":
        raise ValueError("Source is not a provisional core H5")
    source_sha = sha256_file(source_h5)
    rules_sha = hashlib.sha256(_json_bytes(rules)).hexdigest()
    audit_sha = hashlib.sha256(Path(audit_report).read_bytes()).hexdigest() \
        if audit_report is not None else None
    if audit_report is not None:
        audit = json.loads(Path(audit_report).read_text())
        if audit.get("phase") != "complete":
            raise ValueError("Formal prod audit has not completed")
    identity = {"source_h5_sha256": source_sha, "rules_sha256": rules_sha,
                "audit_sha256": audit_sha, "algorithm": SCHEMA}
    export_id = hashlib.sha256(_json_bytes(identity)).hexdigest()[:32]
    dataset_id = "synthetic-provisional-local-" + export_id
    output = output_root / export_id / (dataset_id + ".h5")
    manifest_path = output.with_suffix(".manifest.json")
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if (previous.get("source_h5_sha256") != source_sha
                or previous.get("rules_sha256") != rules_sha
                or previous.get("h5", {}).get("sha256") != sha256_file(output)):
            raise ValueError("Existing local relabel manifest differs")
        return previous
    with h5py.File(source_h5, "r") as source:
        sampling_rate_hz = float(source.attrs["sampling_rate_hz"])
        metadata = get_json(source, "provenance/metadata")
        if audit_sha is not None and metadata.get("prod_audit_sha256") != audit_sha:
            raise ValueError("Source H5 does not bind this formal prod audit")
        index = source["candidate_index"][:]
        if expected_candidates is not None and len(index) != expected_candidates:
            raise ValueError("Source H5 does not cover the expected candidate count")
        raw = get_json(source, "weak_labels/source_candidates")
        raw_by_identity = {(row["candidate_id"], row["version_id"]): row
                           for row in raw}
        if len(raw_by_identity) != len(raw):
            raise ValueError("Source H5 has duplicate source candidates")
        if not raw or len(index) % len(raw):
            raise ValueError("Source H5 has inconsistent sensor mount coverage")
        mounts_per_candidate = len(index) // len(raw)
        sequences = source["sequences"][:]
        weak_rows = []
        for ordinal, (candidate, sequence) in enumerate(zip(index, sequences)):
            candidate_id, version_id = (text(candidate[name])
                                        for name in ("candidate_id", "version_id"))
            original = raw_by_identity[(candidate_id, version_id)]
            commit = {"candidate_id": candidate_id, "version_id": version_id,
                      "source_dataset": text(candidate["source_dataset"]),
                      "source_member": text(sequence["source_file"]),
                      "label_candidates": original["label_candidates"]}
            state, code, rule_id = weak_label(commit, rules)
            weak_rows.append((ordinal, state, code, rule_id))
        if not output.exists():
            with atomic_h5(output, validate_delivery) as result:
                for key in ("samples", "sequences", "annotations", "candidate_index"):
                    source.copy(key, result, name=key)
                labels = result.create_group("weak_labels")
                source.copy("weak_labels/source_candidates", labels,
                            name="source_candidates")
                labels.create_dataset("index", data=table(weak_rows, WEAK_LABEL_INDEX))
                for key, value in source.attrs.items():
                    result.attrs[key] = value
                result.attrs["dataset_id"] = dataset_id
                result.attrs["artifact_id"] = str(uuid5(NAMESPACE_URL, dataset_id))
                result.attrs["logical_content_sha256"] = logical_content_sha256(
                    result["samples"], result["sequences"][:],
                    result["annotations"][:], dataset_id=dataset_id)
                provenance = result.create_group("provenance")
                put_json(provenance, "metadata", {
                    **metadata, "rules": rules, "rules_sha256": rules_sha,
                    "local_relabel": {"schema": SCHEMA,
                                      "source_h5_sha256": source_sha,
                                      "source_dataset_id": text(source.attrs["dataset_id"]),
                                      "audit_sha256": audit_sha}})
    validated = validate_delivery(output)
    states = Counter(row[1] for row in weak_rows)
    result = {"schema": SCHEMA, "export_id": export_id,
              "dataset_id": dataset_id, "hdf5_version": "3.3.0",
              "artifact_profile": "imu_dataset_provisional",
              "evaluation_role": "unverified_synthetic",
              "coverage": "complete" if audit_sha is not None else "local-preview",
              "cloud_published": False,
              "candidate_count": len(raw), "sequence_count": len(index),
              "sample_count": validated["samples"],
              "duration_s": validated["samples"] /
                            (sampling_rate_hz * mounts_per_candidate),
              "weak_count": states["weak"],
              "unresolved_count": states["unresolved"],
              "source_h5_sha256": source_sha, "rules_sha256": rules_sha,
              "prod_audit_sha256": audit_sha,
              "h5": {"path": str(output), "filename": output.name,
                     "byte_length": output.stat().st_size,
                     "sha256": sha256_file(output)},
              "limitations": ["Machine QA only; no human quality or label review",
                              "Weak activity labels may be wrong or unresolved",
                              "No verified fall labels or replay in this H5"]}
    payload = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=".local-core-manifest-",
                                     suffix=".partial", dir=manifest_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, manifest_path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return result
