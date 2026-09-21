"""Immutable, unreviewed synthetic IMU export. Never a formal training snapshot."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence
from uuid import NAMESPACE_URL, uuid5

import h5py
import numpy as np

from .contracts.common import atomic_h5, json_dump, put_json, sha256_file, text
from .contracts.core import (
    ANNOTATIONS,
    COLUMNS,
    SEQUENCES,
    UNITS,
    logical_content_sha256,
)
from .contracts.delivery import (
    PROVISIONAL_INDEX,
    WEAK_LABEL_INDEX,
    table,
    validate_delivery,
)
from .contracts.internal import read_internal
from .motion.selection import selection_slice
from .production.config import config_digest
from .production.runner import _store
from .publication import _json_bytes, publication_prefix
from .sensors.layout import load_layout

RULE_SCHEMA = "imu_motion_simulator.provisional_rules.v1"
MANIFEST_SCHEMA = "imu_motion_simulator.provisional_export.v1"
CODE = re.compile(r"^[a-z][a-z0-9_]*$")


def default_rules() -> dict:
    """Conservative exact mappings. Every other source label stays unresolved."""
    mapping = {
        "walk": "walking", "run": "running", "jog": "jogging",
        "stand": "standing", "sit": "sitting", "jump": "jumping",
        "turn": "turning", "dance": "dancing", "throw": "throwing",
        "kick": "kicking", "wave": "waving", "stretch": "stretching",
    }
    return {"schema": RULE_SCHEMA, "revision": 1, "rules": [
        {"rule_id": f"babel-{source}-v1", "origin": "babel-1.0",
         "source_value": source, "source_dataset": None,
         "target_code": target, "is_fall": False}
        for source, target in sorted(mapping.items())]}


def validate_rules(rules: dict) -> dict:
    if not isinstance(rules, dict) or set(rules) != {"schema", "revision", "rules"} \
            or rules["schema"] != RULE_SCHEMA or type(rules["revision"]) is not int \
            or rules["revision"] < 1 or not isinstance(rules["rules"], list):
        raise ValueError("Invalid provisional ruleset")
    seen = set()
    for rule in rules["rules"]:
        if not isinstance(rule, dict) or set(rule) != {"rule_id", "origin", "source_value",
                                                   "source_dataset", "target_code", "is_fall"} \
                or not isinstance(rule["rule_id"], str) or not rule["rule_id"] \
                or rule["origin"] not in ({"babel-1.0", "stageii-source-member"}
                    | ({"source-member-token"} if rules["revision"] >= 2 else set())) \
                or not isinstance(rule["source_value"], str) or not rule["source_value"] \
                or rule["source_value"] != rule["source_value"].strip().lower() \
                or not CODE.fullmatch(str(rule["target_code"])) \
                or rule["is_fall"] is not False \
                or (rule["source_dataset"] is not None and
                    (not isinstance(rule["source_dataset"], str) or not rule["source_dataset"])):
            raise ValueError("Invalid or fall-implying provisional rule")
        identity = (rule["origin"], rule["source_value"], rule["source_dataset"])
        if rules["revision"] >= 2:
            value = rule["source_value"]
            if rule["origin"] == "source-member-token" and (
                    not re.fullmatch(r"[a-z]+", value)
                    or rule["source_dataset"] is None):
                raise ValueError("Invalid source-member token rule")
            if rule["origin"] == "babel-1.0" and value.startswith("catset:"):
                recording, separator, temporal = value.partition(";temporal:")
                parts = recording.removeprefix("catset:").split("|")
                times = temporal.split("|") if separator else []
                if (not all(parts) or len(parts) != len(set(parts))
                        or parts != sorted(parts)
                        or (separator and (not all(times)
                            or len(times) != len(set(times))
                            or times != sorted(times)))):
                    raise ValueError("Invalid exact BABEL category set")
            elif rule["origin"] == "babel-1.0" and value.startswith("code:") \
                    and not value[5:]:
                raise ValueError("Empty exact BABEL code")
        if rule["rule_id"] in seen or identity in seen:
            raise ValueError("Duplicate provisional rule")
        seen.update((rule["rule_id"], identity))
    return rules


def weak_label(commit: dict, rules: dict) -> tuple[str, str, str]:
    """Assign one recording-level weak code only when source evidence agrees."""
    labels = commit.get("label_candidates") or []
    # Revision 2 is an opt-in, locally compiled preset. These terms may describe
    # a fall or loss of balance; absence of a verified fall label is not ADL.
    if rules["revision"] >= 2 and any(
            risk in str(value).lower()
            for item in labels
            for value in (item.get("code", ""), *(item.get("categories") or []))
            for risk in ("fall", "stumble", "trip", "collapse")):
        return "unresolved", "", ""
    recording = [item for item in labels if item.get("kind") == "recording-candidate"
                 and item.get("code")]
    temporal = [item for item in labels if item.get("kind") == "temporal-candidate"
                and item.get("code")]
    matches = []
    for rule in rules["rules"]:
        if rule["source_dataset"] not in (None, commit["source_dataset"]):
            continue
        origin = rule["origin"]
        own = [item for item in recording if item.get("origin") == origin]
        if origin == "babel-1.0":
            values = {str(category).lower() for item in own
                      for category in item.get("categories") or []}
            temporal_values = {str(category).lower() for item in temporal
                               if item.get("origin") == origin
                               for category in item.get("categories") or []}
            source_value = rule["source_value"]
            if source_value.startswith("catset:"):
                recording_value, _, temporal_value = source_value.partition(";temporal:")
                expected = set(recording_value.removeprefix("catset:").split("|"))
                matches_categories = values == expected
                matches_temporal = (temporal_values == set(temporal_value.split("|"))
                                    if temporal_value else
                                    not temporal_values or temporal_values == values)
                if matches_categories and matches_temporal:
                    matches.append(rule)
            elif source_value.startswith("code:"):
                codes = {str(item["code"]).lower() for item in own}
                if not values and not temporal_values and codes == {source_value[5:]}:
                    matches.append(rule)
            # Unknown temporal labels do not prove consistency; known conflicting
            # activities make a whole-recording weak label unsafe.
            elif values == {source_value} and (
                    not temporal_values or temporal_values == values):
                matches.append(rule)
        elif len(own) == 1 and str(own[0]["code"]).lower() == rule["source_value"]:
            matches.append(rule)
        elif origin == "source-member-token" and rules["revision"] >= 2:
            source_member = str(commit.get("source_member", ""))
            stem = Path(source_member).stem.lower().removesuffix("_poses")
            tokens = set(re.findall(r"[a-z]+", stem))
            babel_recording = [item for item in recording
                               if item.get("origin") == "babel-1.0"]
            categories = {str(value).lower() for item in babel_recording
                          for value in item.get("categories") or []}
            codes = {str(item["code"]).lower() for item in babel_recording}
            if (not categories and codes <= {"none"}
                    and not any(risk in stem for risk in (
                        "fall", "stumble", "trip", "collapse", "recovery", "push", "accident"))
                    and rule["source_value"] in tokens):
                matches.append(rule)
    specific = [rule for rule in matches if rule["source_dataset"] == commit["source_dataset"]]
    if specific:
        matches = specific
    if rules["revision"] >= 2 and matches and len({
            rule["target_code"] for rule in matches}) == 1:
        chosen = min(matches, key=lambda row: row["rule_id"])
        return "weak", chosen["target_code"], chosen["rule_id"]
    if len(matches) != 1:
        return "unresolved", "", ""
    return "weak", matches[0]["target_code"], matches[0]["rule_id"]


def _published(config: dict, store) -> list[tuple[dict, str]]:
    ledger = Path(config["output"]) / "production.sqlite3"
    if not ledger.is_file():
        raise FileNotFoundError("Production ledger is missing")
    prefix = publication_prefix(config["publication"]["target"],
                                config["publication"]["run_id"])
    with sqlite3.connect(f"file:{ledger}?mode=ro", uri=True) as db:
        frozen = db.execute("SELECT config_sha256 FROM job WHERE id=1").fetchone()
        if frozen is None or frozen[0] != config_digest(config):
            raise ValueError("Production ledger belongs to a different frozen job")
        rows = db.execute("SELECT clip_id, commit_key FROM clips WHERE status='published' "
                          "ORDER BY clip_id").fetchall()
    if not rows:
        raise ValueError("No fully published candidates are available")
    result = []
    for candidate_id, key in rows:
        if not key or not key.startswith(f"{prefix}/candidates/{candidate_id}/") \
                or not key.endswith(".json"):
            raise ValueError("Published ledger contains an invalid candidate key")
        commit = store.read_json(key)
        version = key.rsplit("/", 1)[-1][:-5]
        if commit.get("schema") != "imu_motion_simulator.candidate_commit.v1" \
                or commit.get("candidate_id") != candidate_id \
                or commit.get("version_id") != version \
                or not re.fullmatch(r"[0-9a-f]{64}", version):
            raise ValueError("Published candidate identity mismatch")
        manifest = next((item for item in commit.get("bundle_files", [])
                         if item.get("name") == "manifest.json"), None)
        if manifest is None or not manifest["key"].startswith(prefix + "/"):
            raise ValueError("Published candidate lacks machine QA evidence")
        payload = store.read_bytes(manifest["key"])
        qa = json.loads(payload)
        if len(payload) != manifest["byte_length"] \
                or hashlib.sha256(payload).hexdigest() != manifest["sha256"] \
                or qa.get("qa", {}).get("passed") is not True:
            raise ValueError("Published candidate is not machine QA-passed")
        result.append((commit, hashlib.sha256(_json_bytes(commit)).hexdigest()))
    return result


def _artifact(config: dict, store, commit: dict, role: str, temporary: Path) -> Path:
    item = next((row for row in commit["objects"] if row["role"] == role), None)
    if item is None:
        raise ValueError("Candidate lacks " + role)
    prefix = publication_prefix(config["publication"]["target"],
                                config["publication"]["run_id"])
    if not item["key"].startswith(prefix + "/"):
        raise ValueError("Candidate object escapes publication namespace")
    clip_id, source = commit["candidate_id"], commit["source_dataset"]
    run = Path(config["output"]) / "sources" / source / "run"
    local = ({"motion": list((run / "motions").glob(f"{clip_id}.motion.h5")),
              "selection": list((run / "selections").glob(f"{clip_id}.selection.json")),
              "sensors": list((run / "sensors").glob(f"{clip_id}-*.sensors.h5"))})[role]
    for path in local:
        if path.stat().st_size == item["byte_length"] and sha256_file(path) == item["sha256"]:
            return path
    path = temporary / (role + (".json" if role == "selection" else ".h5"))
    store.download_file(item["key"], path, item["sha256"], item["byte_length"])
    return path


def _layout(config: dict) -> dict:
    catalog = json.loads((Path(config["catalog"]) / "catalog.json").read_text())
    from .pipeline.plan import resolve_inside
    return load_layout(resolve_inside(config["checkout"], catalog["inputs"]["layout"]))


def _coverage(config: dict, frozen_count: int) -> tuple[str, str]:
    """Freeze completeness before deriving the immutable export identity."""
    ledger = Path(config["output"]) / "production.sqlite3"
    if not ledger.is_file():
        return "unknown", "partial"
    with sqlite3.connect(f"file:{ledger}?mode=ro", uri=True) as db:
        row = db.execute("SELECT status FROM job WHERE id=1").fetchone()
        current_published = db.execute(
            "SELECT COUNT(*) FROM clips WHERE status='published'").fetchone()[0]
    job_status = row[0] if row else "unknown"
    coverage = ("complete" if job_status == "complete"
                and current_published == frozen_count else "partial")
    return job_status, coverage


def _audit_complete(path: Path, configs: Sequence[dict], prefix: str,
                    candidate_count: int) -> str:
    """Bind a full prod export to the completed two-job cloud audit."""
    if len(configs) != 2:
        raise ValueError("Formal prod coverage requires both frozen jobs")
    payload = path.read_bytes()
    audit = json.loads(payload)
    if (audit.get("schema") != "imu_motion_simulator.prod_chain_audit.v1"
            or audit.get("prefix") != prefix or audit.get("phase") != "complete"):
        raise ValueError("Formal prod cloud audit is not complete")
    reported = [audit.get("native"), audit.get("stageii")]
    by_report = {str(Path(config["output"]) / "production-report.json"): config
                 for config in configs}
    if len(by_report) != 2:
        raise ValueError("Production jobs share an output ledger")
    total = 0
    for item in reported:
        if not isinstance(item, dict) or item.get("report_path") not in by_report:
            raise ValueError("Cloud audit does not bind both production jobs")
        config = by_report.pop(item["report_path"])
        summary = item.get("summary") or {}
        counts = summary.get("counts") or {}
        if (summary.get("status") != "complete"
                or summary.get("config_sha256") != config_digest(config)
                or counts.get("failed", 0) != 0
                or counts.get("published", 0) + counts.get("excluded", 0)
                   != item.get("planned_clips")):
            raise ValueError("Cloud audit job summary differs from frozen production")
        total += counts["published"]
    cloud = audit.get("all_cloud") or {}
    if (by_report or total != candidate_count
            or cloud.get("full_reference_audit") is not True
            or cloud.get("cloud_commits") != candidate_count
            or cloud.get("cloud_feeds") != candidate_count):
        raise ValueError("Cloud audit does not cover the frozen candidate union")
    return hashlib.sha256(payload).hexdigest()


def export_provisional(config: dict, *, rules: dict | None = None,
                       additional_configs: Sequence[dict] = (),
                       audit_report: Path | None = None) -> dict:
    """Freeze the currently committed machine-pass set and publish manifest last."""
    configs = (config, *additional_configs)
    publication = config["publication"]
    if publication["target"] == "local":
        raise ValueError("Provisional export requires a published dev or prod job")
    if any(other["publication"] != publication for other in configs[1:]):
        raise ValueError("Production jobs must share one publication target")
    store = _store(config)
    prefix = publication_prefix(publication["target"], publication["run_id"])
    rules_key = f"{prefix}/datasets/provisional/rules/current.json"
    if rules is None:
        try:
            rules = store.read_json(rules_key)
        except FileNotFoundError:
            rules = default_rules()
    rules = validate_rules(rules)
    rules_sha = hashlib.sha256(_json_bytes(rules)).hexdigest()
    layouts = [_layout(item) for item in configs]
    if any(item != layouts[0] for item in layouts[1:]):
        raise ValueError("Production jobs use different sensor layouts")
    if any(item.get("policy_sha256") != config.get("policy_sha256")
           for item in configs[1:]):
        raise ValueError("Production jobs use different machine QA policies")
    published = [(source_config, row, digest)
                 for source_config in configs
                 for row, digest in _published(source_config, store)]
    published.sort(key=lambda item: (item[1]["candidate_id"], item[1]["version_id"]))
    ids = [row["candidate_id"] for _, row, _ in published]
    if len(ids) != len(set(ids)):
        raise ValueError("Production jobs contain duplicate candidate IDs")
    by_source = {}
    for source_config, row, _ in published:
        source = row["source_dataset"]
        prior = by_source.setdefault(source, source_config)
        if prior is not source_config:
            raise ValueError("Production jobs overlap in source datasets")
    identities = [(row["candidate_id"], row["version_id"], digest)
                  for _, row, digest in published]
    statuses = [_coverage(item, sum(source_config is item for source_config, _, _
                                    in published)) for item in configs]
    all_complete = all(coverage == "complete" for _, coverage in statuses)
    job_status = (statuses[0][0] if len(statuses) == 1 else
                  "complete" if all_complete else "running")
    audit_sha = None
    if audit_report is not None:
        if publication["target"] != "prod" or not all_complete:
            raise ValueError("Formal prod export requires completed jobs")
        audit_sha = _audit_complete(Path(audit_report), configs, prefix,
                                    len(identities))
    coverage = ("complete" if all_complete and
                (publication["target"] != "prod" or audit_sha is not None)
                else "partial")
    identity = {"commits": identities, "rules_sha256": rules_sha,
                "coverage": coverage}
    if len(configs) > 1:
        identity["jobs"] = sorted(config_digest(item) for item in configs)
    if audit_sha is not None:
        identity["audit_sha256"] = audit_sha
    export_id = hashlib.sha256(_json_bytes(identity)).hexdigest()[:32]
    dataset_id = "synthetic-provisional-" + export_id
    base = f"{prefix}/datasets/provisional/exports/{export_id}"
    manifest_key = base + "/manifest.json"
    try:
        previous = store.read_json(manifest_key)
    except FileNotFoundError:
        previous = None
    if previous is not None:
        if previous.get("schema") != MANIFEST_SCHEMA or previous.get("rules_sha256") != rules_sha \
                or previous.get("candidate_count") != len(identities):
            raise ValueError("Existing provisional export differs")
        return previous
    output = Path(config["output"]) / "provisional" / export_id
    output.mkdir(parents=True, exist_ok=True)
    path = output / (dataset_id + ".h5")
    layout = layouts[0]
    if not path.exists():
        scratch = output / "scratch"
        sequences, index, weak, raw_labels = [], [], [], []
        try:
            with atomic_h5(path, validate_delivery) as handle:
                handle.attrs.update(imu_schema_version="3.3.0",
                                    artifact_profile="imu_dataset_provisional",
                                    artifact_id=str(uuid5(NAMESPACE_URL, base)),
                                    dataset_id=dataset_id,
                                    sampling_rate_hz=np.float64(25),
                                    axis_frame="sensor_local", hdf5_compatibility="1.14",
                                    evaluation_role="unverified_synthetic",
                                    feature_columns=json_dump(COLUMNS))
                samples = handle.create_dataset("samples", shape=(0, 6),
                                                maxshape=(None, 6), chunks=(4096, 6), dtype="<f4")
                samples.attrs.update(columns=json_dump(COLUMNS), units=json_dump(UNITS))
                for source_config, commit, digest in published:
                    scratch.mkdir(exist_ok=True)
                    try:
                        motion = _artifact(source_config, store, commit, "motion", scratch)
                        sensors = _artifact(source_config, store, commit, "sensors", scratch)
                        selection = _artifact(source_config, store, commit, "selection", scratch)
                        _, motion_meta, motion_arrays = read_internal(motion, "motion")
                        _, sensor_meta, sensor_arrays = read_internal(sensors, "sensors")
                        _, selection_value = selection_slice(selection, motion)
                        if sensor_meta["kind_metadata"]["selection_id"] != selection_value["selection_id"] \
                                or sensor_meta["kind_metadata"]["layout_id"] != layout["layout_id"] \
                                or selection_value["label_candidates"] != commit.get("label_candidates", []):
                            raise ValueError("Candidate sensor lineage differs")
                        info = motion_meta["kind_metadata"]
                        participant = "smplh-" + info["source_gender"] + "-" + hashlib.sha256(
                            motion_arrays["betas"].astype("<f8").tobytes()).hexdigest()[:12]
                        state, code, rule_id = weak_label(commit, rules)
                        raw_labels.append({"candidate_id": commit["candidate_id"],
                                           "version_id": commit["version_id"],
                                           "label_candidates": commit.get("label_candidates") or []})
                        force = sensor_arrays["specific_force_m_s2"]
                        angular = sensor_arrays["angular_velocity_rad_s"]
                        if force.shape != angular.shape or force.shape[1] != len(layout["mounts"]):
                            raise ValueError("Candidate sensor layout differs")
                        for mount_index, mount in enumerate(layout["mounts"]):
                            values = np.concatenate((force[:, mount_index], angular[:, mount_index]),
                                                    axis=1).astype("<f4")
                            start = len(samples)
                            samples.resize((start + len(values), 6))
                            samples[start:start + len(values)] = values
                            ordinal = len(sequences)
                            sequences.append((start, start + len(values), info["source_member"],
                                              participant, selection_value["selection_id"],
                                              mount["id"], "unverified", False, "recording",
                                              info["source_fps_hz"]))
                            index.append((ordinal, commit["candidate_id"], commit["version_id"],
                                          digest, commit["source_dataset"]))
                            weak.append((ordinal, state, code, rule_id))
                    finally:
                        shutil.rmtree(scratch, ignore_errors=True)
                sequence_array = table(sequences, SEQUENCES)
                annotations = table([], ANNOTATIONS)
                handle.create_dataset("sequences", data=sequence_array)
                handle.create_dataset("annotations", data=annotations)
                handle.create_dataset("candidate_index", data=table(index, PROVISIONAL_INDEX))
                handle.create_dataset("weak_labels/index", data=table(weak, WEAK_LABEL_INDEX))
                put_json(handle, "weak_labels/source_candidates", raw_labels)
                put_json(handle, "provenance/metadata", {
                    "schema": "imu_motion_simulator.provisional_provenance.v1",
                    "rules_sha256": rules_sha, "rules": rules,
                    "input_commits_sha256": hashlib.sha256(_json_bytes(identities)).hexdigest(),
                    "source_job_config_sha256": sorted(config_digest(item) for item in configs),
                    "prod_audit_sha256": audit_sha,
                    "limitations": ["machine QA only; no human quality or label review",
                                    "weak activity labels may be wrong or unresolved",
                                    "no verified fall labels, replay, model assets or video"]})
                handle.attrs.update(sequence_count=np.int64(len(sequences)),
                                    sample_count=np.int64(len(samples)),
                                    annotation_count=np.int64(0),
                                    logical_content_sha256=logical_content_sha256(
                                        samples, sequence_array, annotations,
                                        dataset_id=dataset_id))
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    report = validate_delivery(path)
    with h5py.File(path) as handle:
        states = [text(value) for value in handle["weak_labels/index"]["state"]]
        counts = {"weak": states.count("weak"),
                  "unresolved": states.count("unresolved")}
    h5_key = base + "/" + path.name
    digest = sha256_file(path)
    store.put_file(path, h5_key, digest)
    result = {"schema": MANIFEST_SCHEMA, "export_id": export_id,
              "dataset_id": dataset_id, "hdf5_version": "3.3.0",
              "artifact_profile": "imu_dataset_provisional",
              "evaluation_role": "unverified_synthetic",
              "candidate_count": len(identities),
              "sequence_count": report["sequences"], "sample_count": report["samples"],
              "duration_s": report["samples"] / (25 * len(layout["mounts"])),
              "weak_count": counts["weak"], "unresolved_count": counts["unresolved"],
              "source_job_status": job_status, "coverage": coverage,
              "source_job_config_sha256": sorted(config_digest(item) for item in configs),
              "prod_audit_sha256": audit_sha,
              "rules_sha256": rules_sha, "rules_revision": rules["revision"],
              "input_commits_sha256": hashlib.sha256(_json_bytes(identities)).hexdigest(),
              "h5": {"object_key": h5_key, "filename": path.name,
                     "sha256": digest, "byte_length": path.stat().st_size},
              "created_at_utc": datetime.now(UTC).isoformat(),
              "limitations": ["Machine QA only; no human review or verified labels",
                              "Use a formal reviewed snapshot for validated training"]}
    store.put_json(manifest_key, result)
    return result
