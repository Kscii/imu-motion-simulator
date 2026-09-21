"""Immutable, per-candidate publication to a local store or private GCS prefix.

The commit document is the only discovery surface. Objects and replay payloads
are uploaded first; an interrupted upload therefore never creates a candidate.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import time
from uuid import uuid4

from .contracts.common import sha256_file
from .pipeline.machine_review import validate_candidate_corpus
from .review.bundle import validate_bundle


ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SHARED_VIEWER = {"index.html", "app.js", "three.module.js", "three.core.min.js"}
MEDIA_TYPES = {".json": "application/json", ".html": "text/html",
               ".js": "text/javascript", ".h5": "application/x-hdf5",
               ".bin": "application/octet-stream"}


def _precondition_failed_exceptions():
    """Return the optional GCS exception without making cloud a core dependency."""
    try:
        from google.api_core.exceptions import PreconditionFailed
    except ModuleNotFoundError:
        return ()
    return (PreconditionFailed,)


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode()


def _safe_key(key):
    value = PurePosixPath(key)
    if value.is_absolute() or not value.parts or ".." in value.parts:
        raise ValueError("Unsafe object key")
    return str(value)


def dev_prefix(run_id):
    return publication_prefix("dev", run_id)


def publication_prefix(target, run_id=None):
    """Resolve a configured publication target without accepting arbitrary keys."""
    if target == "prod":
        if run_id is not None:
            raise ValueError("Production publication does not use a run ID")
        return "synthetic-motion/prod/v1"
    if target == "dev":
        if not isinstance(run_id, str) or not ID.fullmatch(run_id):
            raise ValueError("Dev run ID must use letters, digits, dots, dashes or underscores")
        return f"synthetic-motion/dev/{run_id}/v1"
    raise ValueError("Publication target must be dev or prod")


class LocalPublicationStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key):
        path = (self.root / _safe_key(key)).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Object key escapes store")
        return path

    def put_file(self, source, key, digest):
        source, target = Path(source), self._path(key)
        if sha256_file(source) != digest:
            raise ValueError("Local object changed before publication")
        if target.exists():
            if target.stat().st_size != source.stat().st_size or sha256_file(target) != digest:
                raise ValueError("Existing object has different content: " + key)
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + "." + uuid4().hex + ".partial")
        try:
            shutil.copyfile(source, temporary)
            if sha256_file(temporary) != digest:
                raise ValueError("Copied object hash differs")
            try:
                os.link(temporary, target)
            except FileExistsError:
                if target.stat().st_size != source.stat().st_size or sha256_file(target) != digest:
                    raise ValueError("Concurrent object has different content")
        finally:
            temporary.unlink(missing_ok=True)

    def put_json(self, key, value):
        target = self._path(key)
        payload = _json_bytes(value)
        if target.exists():
            if target.read_bytes() != payload:
                raise ValueError("Existing commit has different content: " + key)
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + "." + uuid4().hex + ".partial")
        try:
            temporary.write_bytes(payload)
            try:
                os.link(temporary, target)
            except FileExistsError:
                if target.read_bytes() != payload:
                    raise ValueError("Concurrent commit has different content")
        finally:
            temporary.unlink(missing_ok=True)

    def read_json(self, key):
        return json.loads(self._path(key).read_text())

    def read_bytes(self, key):
        return self._path(key).read_bytes()

    def list_keys(self, prefix):
        root = self._path(prefix)
        if not root.exists():
            return []
        return sorted(path.relative_to(self.root).as_posix()
                      for path in root.rglob("*") if path.is_file())

    def download_file(self, key, destination, digest, size):
        source = self._path(key)
        if source.stat().st_size != size or sha256_file(source) != digest:
            raise ValueError("Published object hash differs: " + key)
        destination = Path(destination)
        if destination.exists():
            if destination.stat().st_size == size and sha256_file(destination) == digest:
                return
            raise ValueError("Existing downloaded object differs")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if sha256_file(destination) != digest:
            destination.unlink(missing_ok=True)
            raise ValueError("Downloaded object hash differs")


class GcsPublicationStore:
    """Optional cloud adapter; installation uses the project's ``cloud`` group."""

    def __init__(self, bucket, project=None):
        try:
            from google.cloud import storage
        except ImportError as error:
            raise RuntimeError("Install the optional cloud dependency group") from error
        self.bucket = storage.Client(project=project).bucket(bucket)
        self._verified_viewer: dict[str, tuple[str, int, float]] = {}
        self.metrics = {"sdk_calls": 0, "sdk_seconds": 0.0,
                        "cache_hits": 0, "upload_errors": 0,
                        "precondition_exists": 0,
                        "attempted_upload_bytes": 0, "verified_bytes": 0}

    def _call(self, operation, *args, **kwargs):
        started = time.monotonic()
        self.metrics["sdk_calls"] += 1
        kind = getattr(operation, "__name__", "")
        if kind in {"upload_from_filename", "upload_from_string", "get_blob"}:
            calls = f"sdk_{kind}_calls"
            seconds = f"sdk_{kind}_seconds"
            self.metrics[calls] = self.metrics.get(calls, 0) + 1
        try:
            return operation(*args, **kwargs)
        except Exception:
            self.metrics["upload_errors"] += 1
            raise
        finally:
            elapsed = time.monotonic() - started
            self.metrics["sdk_seconds"] += elapsed
            if kind in {"upload_from_filename", "upload_from_string", "get_blob"}:
                self.metrics[seconds] = self.metrics.get(seconds, 0.0) + elapsed

    def metrics_snapshot(self):
        return dict(self.metrics)

    def put_file(self, source, key, digest):
        source = Path(source)
        if sha256_file(source) != digest:
            raise ValueError("Local object changed before publication")
        size = source.stat().st_size
        # Only immutable content-addressed viewer assets may use a short-lived
        # process-local verified cache. A restart always checks GCS again.
        parts = key.split("/")
        viewer = (len(parts) >= 3 and parts[-3] == "viewer"
                  and parts[-2] == digest and parts[-1] in SHARED_VIEWER)
        cached = self._verified_viewer.get(key)
        if (viewer and cached is not None and cached[0] == digest
                and cached[1] == size and cached[2] > time.monotonic()):
            self.metrics["cache_hits"] += 1
            return
        blob = self.bucket.blob(_safe_key(key))
        blob.metadata = {"sha256": digest}
        try:
            self.metrics["attempted_upload_bytes"] += size
            self._call(blob.upload_from_filename,
                str(source), content_type=MEDIA_TYPES.get(source.suffix, "application/octet-stream"),
                if_generation_match=0, checksum="auto")
        except _precondition_failed_exceptions():
            self.metrics["upload_errors"] -= 1
            self.metrics["precondition_exists"] += 1
        remote = self._call(self.bucket.get_blob, key)
        if remote is None or remote.size != size \
                or (remote.metadata or {}).get("sha256") != digest:
            raise ValueError("Remote object could not be verified: " + key)
        if viewer:
            self._verified_viewer[key] = (digest, size, time.monotonic() + 600)
        self.metrics["verified_bytes"] += size

    def put_json(self, key, value):
        payload = _json_bytes(value)
        blob = self.bucket.blob(_safe_key(key))
        blob.metadata = {"sha256": hashlib.sha256(payload).hexdigest()}
        try:
            self.metrics["attempted_upload_bytes"] += len(payload)
            self._call(blob.upload_from_string, payload, content_type="application/json",
                       if_generation_match=0, checksum="auto")
        except _precondition_failed_exceptions():
            self.metrics["upload_errors"] -= 1
            self.metrics["precondition_exists"] += 1
        remote = self._call(self.bucket.get_blob, key)
        if remote is None or remote.size != len(payload) \
                or (remote.metadata or {}).get("sha256") != hashlib.sha256(payload).hexdigest():
            raise ValueError("Remote commit could not be verified: " + key)
        self.metrics["verified_bytes"] += len(payload)

    def read_json(self, key):
        from google.api_core.exceptions import NotFound
        try:
            payload = self.bucket.blob(_safe_key(key)).download_as_bytes()
        except NotFound as error:
            raise FileNotFoundError(key) from error
        return json.loads(payload)

    def read_bytes(self, key):
        from google.api_core.exceptions import NotFound
        try:
            return self.bucket.blob(_safe_key(key)).download_as_bytes()
        except NotFound as error:
            raise FileNotFoundError(key) from error

    def list_keys(self, prefix):
        return sorted(blob.name for blob in self.bucket.list_blobs(prefix=_safe_key(prefix)))

    def download_file(self, key, destination, digest, size):
        destination = Path(destination)
        if destination.exists():
            if destination.stat().st_size == size and sha256_file(destination) == digest:
                return
            raise ValueError("Existing downloaded object differs")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + "." + uuid4().hex + ".partial")
        try:
            self.bucket.blob(_safe_key(key)).download_to_filename(str(temporary))
            if temporary.stat().st_size != size or sha256_file(temporary) != digest:
                raise ValueError("Downloaded object hash differs: " + key)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)


def choose_pilot(index, *, per_category=3):
    """Select repeatable examples from the existing four-stratum review index."""
    if index.get("schema") != "imu_motion_simulator.review_sample_index.v1":
        raise ValueError("Unexpected review sample index")
    chosen = []
    for category in ("ordinary", "warning", "label-ambiguous", "high-risk"):
        rows = [row for row in index["bundles"] if row["category"] == category]
        rows.sort(key=lambda row: (hashlib.sha256(row["candidate_id"].encode()).hexdigest(),
                                   row["candidate_id"]))
        if len(rows) < per_category:
            raise ValueError("Not enough samples in " + category)
        chosen.extend(rows[:per_category])
    if not any(row["source"] in {"GRAB", "SOMA"} for row in chosen):
        stageii = [row for row in index["bundles"]
                   if row["source"] in {"GRAB", "SOMA"} and row not in chosen]
        if not stageii:
            raise ValueError("Review index lacks Stage-II coverage")
        replacement = min(stageii, key=lambda row: (row["category"] != "high-risk",
                                                     row["candidate_id"]))
        same_category = [row for row in chosen if row["category"] == replacement["category"]]
        chosen.remove(same_category[-1])
        chosen.append(replacement)
    return sorted(chosen, key=lambda row: (row["category"], row["candidate_id"]))


def publish_candidate(corpus, candidate, bundle, store, *, run_id=None, outbox,
                      target="dev"):
    """Publish one fully validated candidate; safe to retry with the same inputs."""
    validate_candidate_corpus(corpus)
    candidate_id = candidate["candidate_id"]
    if not ID.fullmatch(candidate_id) or candidate not in corpus["candidates"]:
        raise ValueError("Candidate is not in the validated corpus")
    bundle = Path(bundle).resolve()
    validation = validate_bundle(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    if validation["decision"] != "unreviewed" or not manifest["qa"]["passed"]:
        raise ValueError("Only unreviewed machine-pass bundles may be published")
    risk_score = manifest["qa"].get("risk_score")
    if risk_score is not None and (
            isinstance(risk_score, bool) or not isinstance(risk_score, (int, float))
            or not math.isfinite(risk_score) or not 0 <= risk_score <= 1):
        raise ValueError("Invalid frozen QA risk score")
    risk_policy_id = "kinematic-qa-v1-high-0.7" if risk_score is not None else None
    for role, field in (("motion", "motion_sha256"), ("sensors", "sensor_sha256")):
        if manifest[field] != candidate["objects"][role]:
            raise ValueError("Review bundle input differs: " + role)
    if manifest["selection"] is None or not manifest["selection"].get("selection_id"):
        raise ValueError("Review bundle must bind a selection")
    descriptors = {row["sha256"]: row for row in corpus["objects"]}
    selection_path = Path(descriptors[candidate["objects"]["selection"]]["local_path"])
    selection_value = json.loads(selection_path.read_text())
    if selection_value.get("selection_id") != manifest["selection"]["selection_id"] \
            or selection_value.get("motion_sha256") != candidate["objects"]["motion"]:
        raise ValueError("Review bundle selection differs from candidate")
    prefix = publication_prefix(target, run_id)
    objects = []
    paths = []
    for role, digest in sorted(candidate["objects"].items()):
        descriptor = descriptors[digest]
        source = Path(descriptor["local_path"])
        key = prefix + descriptor["object_key"].removeprefix("synthetic-motion/v1")
        objects.append({"role": role, "key": key, "sha256": digest,
                        "byte_length": descriptor["byte_length"]})
        paths.append((source, key, digest))
    bundle_files = []
    bundle_paths = []
    for source in sorted(bundle.iterdir()):
        if not source.is_file() or source.name.endswith(".partial"):
            continue
        digest = sha256_file(source)
        bundle_files.append({"name": source.name, "sha256": digest,
                             "byte_length": source.stat().st_size})
    version_source = {"candidate_id": candidate_id,
                      "policy_sha256": corpus["policy"]["sha256"],
                      "objects": objects, "bundle_files": bundle_files}
    if risk_policy_id is not None:
        version_source["risk_policy_id"] = risk_policy_id
    version_id = hashlib.sha256(_json_bytes(version_source)).hexdigest()
    for item in bundle_files:
        name = item["name"]
        item["key"] = (f"{prefix}/viewer/{item['sha256']}/{name}" if name in SHARED_VIEWER
                       else f"{prefix}/previews/{candidate_id}/{version_id}/{name}")
        bundle_paths.append((bundle / name, item["key"], item["sha256"]))
    commit = {
        "schema": "imu_motion_simulator.candidate_commit.v1",
        "candidate_id": candidate_id, "version_id": version_id,
        "source_dataset": candidate["source_dataset"],
        "source_member": candidate["source_member"],
        "policy_sha256": corpus["policy"]["sha256"],
        "label_candidates": candidate["label_candidates"],
        "warning_flags": candidate["warning_flags"],
        "risk_score": risk_score,
        "risk_tier": ("high" if risk_score >= .7 else "ordinary")
        if risk_score is not None else "unknown",
        "risk_policy_id": risk_policy_id,
        "objects": objects, "bundle_files": bundle_files,
        "published_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    outbox = Path(outbox).resolve()
    legacy_receipt = outbox / (candidate_id + "-" + version_id + ".json")
    if not (target == "dev" and legacy_receipt.exists()):
        outbox = outbox / target / (run_id or "_production")
    outbox.mkdir(parents=True, exist_ok=True)
    receipt = outbox / (candidate_id + "-" + version_id + ".json")
    if receipt.exists():
        prior = json.loads(receipt.read_text())
        if prior["version_id"] != version_id or prior["candidate_id"] != candidate_id:
            raise ValueError("Outbox identity conflict")
        commit = prior
    else:
        try:
            with receipt.open("xb") as handle:
                handle.write(_json_bytes(commit))
        except FileExistsError:
            commit = json.loads(receipt.read_text())
            if commit["version_id"] != version_id or commit["candidate_id"] != candidate_id:
                raise ValueError("Outbox identity conflict")
    for source, key, digest in paths + bundle_paths:
        store.put_file(source, key, digest)
    commit_key = f"{prefix}/candidates/{candidate_id}/{version_id}.json"
    store.put_json(commit_key, commit)
    # The commit remains the sole eligibility authority. This small, append-only
    # receipt lets the annotation index discover new commits without relisting
    # every candidate on each poll. Keep its key in the local outbox for retries.
    feed_marker = receipt.with_name(receipt.stem + ".feed.json")

    def new_feed():
        posted_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        return {
            "schema": "imu_motion_simulator.candidate_index_feed.v1",
            "candidate_id": candidate_id,
            "version_id": version_id,
            "commit_key": commit_key,
            "commit_sha256": hashlib.sha256(_json_bytes(commit)).hexdigest(),
            "posted_at_utc": datetime.now(timezone.utc).isoformat(),
            "feed_key": f"{prefix}/index-feed/{posted_at[:8]}{posted_at[9:11]}/"
                        f"{posted_at}-{candidate_id}-{version_id}.json",
        }

    reused_marker = feed_marker.exists()
    if reused_marker:
        feed = json.loads(feed_marker.read_text())
    else:
        feed = new_feed()
        try:
            with feed_marker.open("xb") as handle:
                handle.write(_json_bytes(feed))
        except FileExistsError:
            feed = json.loads(feed_marker.read_text())
            reused_marker = True
    if feed["candidate_id"] != candidate_id or feed["version_id"] != version_id \
            or feed["commit_key"] != commit_key \
            or feed["commit_sha256"] != hashlib.sha256(_json_bytes(commit)).hexdigest():
        raise ValueError("Outbox index feed identity conflict")
    if reused_marker:
        try:
            remote_feed = store.read_json(feed["feed_key"])
        except FileNotFoundError:
            # A failed feed upload may be retried after later clips have already
            # advanced the consumer's lexicographic cursor. Repost under a new
            # key so the incremental consumer can still discover this commit.
            feed = new_feed()
            temporary = feed_marker.with_name(feed_marker.name + "." + uuid4().hex + ".partial")
            try:
                temporary.write_bytes(_json_bytes(feed))
                os.replace(temporary, feed_marker)
            finally:
                temporary.unlink(missing_ok=True)
        else:
            if remote_feed != feed:
                raise ValueError("Published index feed differs from local outbox")
    store.put_json(feed["feed_key"], feed)
    return {"candidate_id": candidate_id, "version_id": version_id,
            "commit_key": commit_key, "objects": len(paths),
            "bundle_files": len(bundle_paths)}
