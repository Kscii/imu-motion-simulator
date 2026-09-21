"""Unattended, fail-closed handoff between the two frozen formal prod jobs.

This is an operations tool, not a second producer.  It waits for the native
job to exit, checks its local ledger against the cloud commit/feed index, and
only then starts the Stage-II job.  It never makes review decisions or creates
a training snapshot.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time

from imu_motion_simulator.production import JobState, load_job_config
from imu_motion_simulator.production.runner import _source_plan_rows
from imu_motion_simulator.publication import GcsPublicationStore, publication_prefix


FEED_NAME = re.compile(r"\d{8}T\d{12}Z-(.+)-([0-9a-f]{64})\.json\Z")


def _save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _service_state(unit: str) -> str:
    result = subprocess.run(
        ["systemctl", "--user", "show", unit, "--property=ActiveState", "--value"],
        check=True, text=True, capture_output=True)
    return result.stdout.strip()


def _wait_for_complete(config: dict, unit: str, *, poll_seconds: int) -> dict:
    ledger = Path(config["output"]) / "production.sqlite3"
    if not ledger.is_file():
        raise RuntimeError(f"Job ledger missing: {ledger}")
    state = JobState(config)
    while True:
        summary = state.summary()
        service = _service_state(unit)
        if summary["status"] == "complete" and service != "active":
            return summary
        if summary["status"] in {"failed", "partial", "paused", "cancelled"} \
                and service != "active":
            raise RuntimeError(f"{unit} stopped in {summary['status']}: {summary['counts']}")
        if summary["status"] in {"queued", "running"} and service != "active":
            raise RuntimeError(f"{unit} is {service} while ledger says {summary['status']}")
        time.sleep(poll_seconds)


def _only_transient_network_failures(config: dict) -> int:
    """Never retry source/quality failures under the network recovery policy."""
    ledger = Path(config["output"]) / "production.sqlite3"
    with sqlite3.connect(f"file:{ledger}?mode=ro", uri=True) as db:
        errors = [row[0] or "" for row in db.execute(
            "SELECT error FROM clips WHERE status='failed'")]
    domains = ("storage.googleapis.com", "oauth2.googleapis.com")
    symptoms = ("Timeout", "NameResolutionError", "ConnectionError")
    return (len(errors) if errors and all(
        any(domain in error for domain in domains)
        and any(symptom in error for symptom in symptoms)
        for error in errors) else 0)


def _wait_with_network_retry(config: dict, config_path: Path, unit: str,
                             cli: Path, *, poll_seconds: int,
                             retry_attempts: int, retry_delay_seconds: int) -> tuple[dict, list[dict]]:
    attempts = []
    while True:
        try:
            return _wait_for_complete(config, unit, poll_seconds=poll_seconds), attempts
        except RuntimeError:
            summary = JobState(config).summary()
            failed_count = _only_transient_network_failures(config)
            # A failed job indicates a fatal run error; clip errors alone cannot
            # prove that such an error was transient.
            if (summary["status"] != "partial"
                    or _service_state(unit) == "active"
                    or failed_count != summary["counts"].get("failed", 0)
                    or not failed_count or len(attempts) >= retry_attempts):
                raise
            time.sleep(retry_delay_seconds)
            result = subprocess.run([str(cli), "production", "retry", str(config_path)],
                                    check=False)
            attempts.append({"attempt": len(attempts) + 1,
                             "failed_before": failed_count,
                             "exit_code": result.returncode})


def _ledger(config: dict, *, expected_published: int,
            expected_excluded: int) -> tuple[dict, dict[str, tuple[str, str]]]:
    state = JobState(config)
    summary = state.summary()
    rows = _source_plan_rows(config)
    planned = sum(len(clips) for _, _, _, clips in rows)
    counts = summary["counts"]
    if (summary["status"] != "complete" or counts.get("published", 0) != expected_published
            or counts.get("excluded", 0) != expected_excluded
            or counts.get("failed", 0) != 0
            or sum(counts.values()) != planned):
        raise RuntimeError(f"Frozen job count mismatch: {summary}, planned={planned}")
    report_path = Path(config["output"]) / "production-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report["status"] != "complete" or report["planned_clips"] != planned
            or report["source_errors"]
            or any(value["status"] != "complete" for value in report["sources"].values())):
        raise RuntimeError(f"Source report incomplete: {report_path}")
    with sqlite3.connect(Path(config["output"]) / "production.sqlite3") as db:
        clip_rows = db.execute(
            "SELECT clip_id, source_dataset, status, commit_key FROM clips"
        ).fetchall()
    commits: dict[str, tuple[str, str]] = {}
    per_source = Counter()
    for clip_id, source, status, key in clip_rows:
        per_source[source] += 1
        if status == "published":
            if not key or key in commits:
                raise RuntimeError(f"Missing or duplicate commit key: {clip_id}")
            commits[key] = (clip_id, source)
        elif status != "excluded" or key:
            raise RuntimeError(f"Unexpected clip ledger state: {clip_id} {status}")
    if len(commits) != expected_published or len(clip_rows) != planned:
        raise RuntimeError("Ledger rows do not match frozen counts")
    for source, _, _, clips in rows:
        if per_source[source] != len(clips):
            raise RuntimeError(f"Incomplete source {source}: {per_source[source]}/{len(clips)}")
    return {"summary": summary, "planned_clips": planned,
            "sources": len(rows), "report_path": str(report_path)}, commits


def _feed_pair(key: str) -> tuple[str, str]:
    name = key.rsplit("/", 1)[-1]
    match = FEED_NAME.fullmatch(name)
    if match is None:
        raise RuntimeError(f"Unexpected feed key: {key}")
    return match.group(1), match.group(2)


def _audit_cloud(store: GcsPublicationStore, prefix: str,
                 commits: dict[str, tuple[str, str]], *, full: bool = False) -> dict:
    listed = ({blob.name: blob for blob in store.bucket.list_blobs(prefix=f"{prefix}/")}
              if full else None)
    cloud_commits = (
        {key: blob for key, blob in listed.items()
         if key.startswith(f"{prefix}/candidates/")}
        if listed is not None else
        {blob.name: blob for blob in store.bucket.list_blobs(prefix=f"{prefix}/candidates/")})
    if set(cloud_commits) != set(commits):
        missing = set(commits) - set(cloud_commits)
        extra = set(cloud_commits) - set(commits)
        raise RuntimeError(f"Cloud commit mismatch: missing={len(missing)}, extra={len(extra)}")
    expected_pairs = set()
    by_source: dict[str, str] = {}
    for key, (clip_id, source) in commits.items():
        version = Path(key).stem
        if key != f"{prefix}/candidates/{clip_id}/{version}.json":
            raise RuntimeError(f"Invalid ledger commit key: {key}")
        expected_pairs.add((clip_id, version))
        by_source.setdefault(source, key)
        blob = cloud_commits[key]
        if not blob.size or not (blob.metadata or {}).get("sha256"):
            raise RuntimeError(f"Cloud commit has no size/hash metadata: {key}")
    feeds = ([blob for key, blob in listed.items()
              if key.startswith(f"{prefix}/index-feed/")]
             if listed is not None else list(store.bucket.list_blobs(
                 prefix=f"{prefix}/index-feed/")))
    feed_pairs = [_feed_pair(blob.name) for blob in feeds]
    if len(feed_pairs) != len(commits) or set(feed_pairs) != expected_pairs:
        raise RuntimeError(f"Cloud feed mismatch: feeds={len(feed_pairs)}, commits={len(commits)}")
    full_artifact_checks = 0
    full_bytes_checked = 0
    if listed is not None:
        digests = {}
        for key, blob in cloud_commits.items():
            payload = blob.download_as_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            if len(payload) != blob.size or digest != (blob.metadata or {}).get("sha256"):
                raise RuntimeError(f"Cloud commit content hash mismatch: {key}")
            commit = json.loads(payload)
            clip_id, source = commits[key]
            if (commit.get("schema") != "imu_motion_simulator.candidate_commit.v1"
                    or commit.get("candidate_id") != clip_id
                    or commit.get("version_id") != Path(key).stem
                    or commit.get("source_dataset") != source):
                raise RuntimeError(f"Cloud commit identity mismatch: {key}")
            digests[key] = digest
            full_bytes_checked += len(payload)
            for artifact in commit["objects"] + commit["bundle_files"]:
                linked = listed.get(artifact["key"])
                if (linked is None or linked.size != artifact["byte_length"]
                        or (linked.metadata or {}).get("sha256") != artifact["sha256"]):
                    raise RuntimeError(f"Cloud artifact reference mismatch: {artifact['key']}")
                full_artifact_checks += 1
        for blob in feeds:
            payload = blob.download_as_bytes()
            if (len(payload) != blob.size
                    or hashlib.sha256(payload).hexdigest()
                    != (blob.metadata or {}).get("sha256")):
                raise RuntimeError(f"Cloud feed content hash mismatch: {blob.name}")
            feed = json.loads(payload)
            if (feed.get("schema") != "imu_motion_simulator.candidate_index_feed.v1"
                    or feed.get("feed_key") != blob.name
                    or feed.get("commit_sha256") != digests.get(feed.get("commit_key"))
                    or (feed.get("candidate_id"), feed.get("version_id"))
                    != _feed_pair(blob.name)):
                raise RuntimeError(f"Cloud feed binding mismatch: {blob.name}")
            full_bytes_checked += len(payload)
    artifact_checks = 0
    content_checks = 0
    for source, key in sorted(by_source.items()):
        commit = store.read_json(key)
        clip_id, recorded_source = commits[key]
        if (commit["candidate_id"] != clip_id or commit["source_dataset"] != recorded_source
                or commit["version_id"] != Path(key).stem
                or commit["schema"] != "imu_motion_simulator.candidate_commit.v1"):
            raise RuntimeError(f"Sample commit identity mismatch: {key}")
        for artifact in commit["objects"] + commit["bundle_files"]:
            blob = store.bucket.get_blob(artifact["key"])
            if (blob is None or blob.size != artifact["byte_length"]
                    or (blob.metadata or {}).get("sha256") != artifact["sha256"]):
                raise RuntimeError(f"Sample artifact mismatch: {artifact['key']}")
            artifact_checks += 1
            if (artifact.get("role") == "selection"
                    or artifact.get("name") == "manifest.json"):
                payload = blob.download_as_bytes()
                if (len(payload) != artifact["byte_length"]
                        or hashlib.sha256(payload).hexdigest() != artifact["sha256"]):
                    raise RuntimeError(f"Sample content hash mismatch: {artifact['key']}")
                content_checks += 1
    return {"cloud_commits": len(cloud_commits), "cloud_feeds": len(feed_pairs),
            "full_reference_audit": full,
            "referenced_artifacts_checked": full_artifact_checks,
            "commit_feed_bytes_checked": full_bytes_checked,
            "sampled_sources": sorted(by_source),
            "sampled_artifacts": artifact_checks,
            "sampled_content_hashes": content_checks}


def _start_stageii(cli: Path, config_path: Path, unit: str,
                   upload_profile: Path | None = None) -> None:
    ledger = Path(load_job_config(config_path)["output"]) / "production.sqlite3"
    if ledger.exists() or _service_state(unit) == "active":
        return
    command = ["systemd-run", "--user", f"--unit={unit}",
                    "--description=IMU synthetic prod Stage-II GRAB SOMA",
                    "--property=Restart=no", str(cli), "production", "run",
                    str(config_path), "--json"]
    if upload_profile is not None:
        command.extend(["--upload-profile", str(upload_profile)])
    subprocess.run(command, check=True)
    for _ in range(30):
        if ledger.is_file():
            return
        time.sleep(1)
    raise RuntimeError("Stage-II service started but did not create a ledger")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native_config", type=Path)
    parser.add_argument("stageii_config", type=Path)
    parser.add_argument("--native-unit", default="imu-motion-native.service")
    parser.add_argument("--stageii-unit", default="imu-motion-stageii.service")
    parser.add_argument("--sdk-cli", type=Path,
                        help="installed SDK used for retries and Stage-II launch")
    parser.add_argument("--stageii-upload-profile", type=Path,
                        help="operational upload profile bound on Stage-II first run")
    parser.add_argument("--native-published", type=int, required=True)
    parser.add_argument("--native-excluded", type=int, required=True)
    parser.add_argument("--stageii-published", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--job-retry-attempts", type=int, default=0)
    parser.add_argument("--retry-delay-seconds", type=int, default=300)
    args = parser.parse_args()
    if (args.poll_seconds < 1 or args.job_retry_attempts < 0
            or args.retry_delay_seconds < 0):
        parser.error("poll/retry parameters must be nonnegative; poll must be positive")
    native = load_job_config(args.native_config)
    stageii = load_job_config(args.stageii_config)
    for config in (native, stageii):
        pub = config["publication"]
        if pub["target"] != "prod" or pub["backend"] != "gcs" or pub["run_id"] is not None:
            raise RuntimeError("Both jobs must target the formal prod GCS prefix")
    if native["publication"] != stageii["publication"]:
        raise RuntimeError("Jobs do not share one publication target")
    publication = native["publication"]
    prefix = publication_prefix("prod", None)
    store = GcsPublicationStore(publication["bucket"], publication["project"])
    report = {"schema": "imu_motion_simulator.prod_chain_audit.v1",
              "prefix": prefix, "phase": "waiting-native"}
    _save(args.report, report)
    try:
        cli = args.sdk_cli or args.stageii_config.parent / ".venv" / "bin" / "imu-sim"
        _, report["native_retries"] = _wait_with_network_retry(
            native, args.native_config, args.native_unit, cli,
            poll_seconds=args.poll_seconds,
            retry_attempts=args.job_retry_attempts,
            retry_delay_seconds=args.retry_delay_seconds)
        native_result, native_commits = _ledger(
            native, expected_published=args.native_published,
            expected_excluded=args.native_excluded)
        report.update({"phase": "auditing-native", "native": native_result})
        _save(args.report, report)
        report["native_cloud"] = _audit_cloud(store, prefix, native_commits)
        report["phase"] = "starting-stageii"
        _save(args.report, report)
        _start_stageii(cli, args.stageii_config, args.stageii_unit,
                       args.stageii_upload_profile)
        report["phase"] = "waiting-stageii"
        _save(args.report, report)
        _, report["stageii_retries"] = _wait_with_network_retry(
            stageii, args.stageii_config, args.stageii_unit, cli,
            poll_seconds=args.poll_seconds,
            retry_attempts=args.job_retry_attempts,
            retry_delay_seconds=args.retry_delay_seconds)
        stageii_result, stageii_commits = _ledger(
            stageii, expected_published=args.stageii_published, expected_excluded=0)
        if set(native_commits) & set(stageii_commits):
            raise RuntimeError("Native and Stage-II commit keys overlap")
        report.update({"phase": "auditing-all", "stageii": stageii_result})
        _save(args.report, report)
        report["all_cloud"] = _audit_cloud(
            store, prefix, native_commits | stageii_commits, full=True)
        report["phase"] = "complete"
        report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        _save(args.report, report)
        return 0
    except Exception as error:
        report["phase"] = "failed"
        report["error"] = str(error)
        report["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
        _save(args.report, report)
        raise


if __name__ == "__main__":
    sys.exit(main())
