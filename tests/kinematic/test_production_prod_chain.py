"""Production handoff must reject missing cloud receipts before Stage-II starts."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

_spec = importlib.util.spec_from_file_location(
    "production_prod_chain", Path(__file__).resolve().parents[2]
    / "tools" / "production_prod_chain.py")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
_audit_cloud = _module._audit_cloud
_feed_pair = _module._feed_pair


PREFIX = "synthetic-motion/prod/v1"
CLIP = "accad-walk-a"
VERSION = "a" * 64
COMMIT_KEY = f"{PREFIX}/candidates/{CLIP}/{VERSION}.json"
FEED_KEY = f"{PREFIX}/index-feed/2026091712/20260917T123456123456Z-{CLIP}-{VERSION}.json"


class Blob:
    def __init__(self, name, payload):
        self.name = name
        self.payload = payload
        self.size = len(payload)
        self.metadata = {"sha256": hashlib.sha256(payload).hexdigest()}

    def download_as_bytes(self):
        return self.payload


class Bucket:
    def __init__(self, blobs):
        self.blobs = {blob.name: blob for blob in blobs}

    def list_blobs(self, prefix):
        return (blob for key, blob in self.blobs.items() if key.startswith(prefix))

    def get_blob(self, key):
        return self.blobs.get(key)


class Store:
    def __init__(self, blobs, commit):
        self.bucket = Bucket(blobs)
        self.commit = commit

    def read_json(self, key):
        assert key == COMMIT_KEY
        return self.commit


def fixture():
    selection = b'{"frame_range":[0,10]}\n'
    manifest = b'{"schema":"review"}\n'
    artifacts = [
        {"role": "selection", "key": f"{PREFIX}/objects/selection.json",
         "byte_length": len(selection), "sha256": hashlib.sha256(selection).hexdigest()},
        {"name": "manifest.json", "key": f"{PREFIX}/previews/{CLIP}/{VERSION}/manifest.json",
         "byte_length": len(manifest), "sha256": hashlib.sha256(manifest).hexdigest()},
    ]
    commit = {"schema": "imu_motion_simulator.candidate_commit.v1",
              "candidate_id": CLIP, "version_id": VERSION, "source_dataset": "ACCAD",
              "objects": [artifacts[0]], "bundle_files": [artifacts[1]]}
    commit_bytes = json.dumps(commit).encode()
    feed = {"schema": "imu_motion_simulator.candidate_index_feed.v1",
            "feed_key": FEED_KEY, "commit_key": COMMIT_KEY,
            "commit_sha256": hashlib.sha256(commit_bytes).hexdigest(),
            "candidate_id": CLIP, "version_id": VERSION}
    blobs = [Blob(COMMIT_KEY, commit_bytes), Blob(FEED_KEY, json.dumps(feed).encode()),
             Blob(artifacts[0]["key"], selection), Blob(artifacts[1]["key"], manifest)]
    return Store(blobs, commit)


def test_cloud_audit_accepts_complete_commit_and_feed():
    result = _audit_cloud(fixture(), PREFIX, {COMMIT_KEY: (CLIP, "ACCAD")})
    assert result["cloud_commits"] == result["cloud_feeds"] == 1
    assert result["sampled_content_hashes"] == 2


def test_full_cloud_audit_checks_every_reference_and_feed_binding():
    result = _audit_cloud(fixture(), PREFIX, {COMMIT_KEY: (CLIP, "ACCAD")}, full=True)
    assert result["full_reference_audit"] is True
    assert result["referenced_artifacts_checked"] == 2
    store = fixture()
    feed = json.loads(store.bucket.blobs[FEED_KEY].payload)
    feed["commit_sha256"] = "0" * 64
    store.bucket.blobs[FEED_KEY] = Blob(FEED_KEY, json.dumps(feed).encode())
    with pytest.raises(RuntimeError, match="Cloud feed binding mismatch"):
        _audit_cloud(store, PREFIX, {COMMIT_KEY: (CLIP, "ACCAD")}, full=True)


def test_cloud_audit_blocks_missing_feed():
    store = fixture()
    del store.bucket.blobs[FEED_KEY]
    with pytest.raises(RuntimeError, match="Cloud feed mismatch"):
        _audit_cloud(store, PREFIX, {COMMIT_KEY: (CLIP, "ACCAD")})


def test_cloud_audit_blocks_damaged_artifact():
    store = fixture()
    key = store.commit["objects"][0]["key"]
    store.bucket.blobs[key].payload = b"damaged"
    with pytest.raises(RuntimeError, match="Sample content hash mismatch"):
        _audit_cloud(store, PREFIX, {COMMIT_KEY: (CLIP, "ACCAD")})


def test_feed_parser_rejects_unexpected_names():
    assert _feed_pair(FEED_KEY) == (CLIP, VERSION)
    with pytest.raises(RuntimeError, match="Unexpected feed key"):
        _feed_pair(f"{PREFIX}/index-feed/{CLIP}-{VERSION}.json")


def test_network_retry_gate_rejects_mixed_failures(tmp_path):
    ledger = tmp_path / "production.sqlite3"
    with sqlite3.connect(ledger) as db:
        db.execute("CREATE TABLE clips (status TEXT, error TEXT)")
        db.executemany("INSERT INTO clips VALUES ('failed', ?)", [
            ("storage.googleapis.com: Timeout",),
            ("oauth2.googleapis.com: NameResolutionError",),
        ])
    config = {"output": str(tmp_path)}
    assert _module._only_transient_network_failures(config) == 2
    with sqlite3.connect(ledger) as db:
        db.execute("INSERT INTO clips VALUES ('failed', 'Invalid source frame')")
    assert _module._only_transient_network_failures(config) == 0


def test_network_retry_waits_for_terminated_job_then_resumes_once(monkeypatch):
    statuses = iter(["partial", "complete"])
    calls = []

    def wait(_config, _unit, *, poll_seconds):
        assert poll_seconds == 1
        if next(statuses) == "partial":
            raise RuntimeError("partial job")
        return {"status": "complete"}

    class State:
        def __init__(self, _config):
            pass

        def summary(self):
            return {"status": "partial", "counts": {"failed": 2}}

    monkeypatch.setattr(_module, "_wait_for_complete", wait)
    monkeypatch.setattr(_module, "JobState", State)
    monkeypatch.setattr(_module, "_only_transient_network_failures", lambda _: 2)
    monkeypatch.setattr(_module, "_service_state", lambda _: "inactive")
    monkeypatch.setattr(_module.time, "sleep", lambda delay: calls.append(("sleep", delay)))
    monkeypatch.setattr(_module.subprocess, "run", lambda command, **_: (
        calls.append(("run", command)) or SimpleNamespace(returncode=0)))

    summary, retries = _module._wait_with_network_retry(
        {}, Path("native.yaml"), "native.service", Path("imu-sim"),
        poll_seconds=1, retry_attempts=3, retry_delay_seconds=5)
    assert summary["status"] == "complete"
    assert retries == [{"attempt": 1, "failed_before": 2, "exit_code": 0}]
    assert calls == [("sleep", 5),
                     ("run", ["imu-sim", "production", "retry", "native.yaml"])]


def test_network_retry_never_retries_source_failures(monkeypatch):
    class State:
        def __init__(self, _config):
            pass

        def summary(self):
            return {"status": "partial", "counts": {"failed": 2}}

    monkeypatch.setattr(_module, "_wait_for_complete", lambda *_args, **_kwargs: (
        (_ for _ in ()).throw(RuntimeError("partial job"))))
    monkeypatch.setattr(_module, "JobState", State)
    monkeypatch.setattr(_module, "_only_transient_network_failures", lambda _: 0)
    monkeypatch.setattr(_module, "_service_state", lambda _: "inactive")
    monkeypatch.setattr(_module.subprocess, "run", lambda *_args, **_kwargs: (
        pytest.fail("must not retry a source failure")))
    with pytest.raises(RuntimeError, match="partial job"):
        _module._wait_with_network_retry({}, Path("native.yaml"), "native.service",
                                         Path("imu-sim"), poll_seconds=1,
                                         retry_attempts=3, retry_delay_seconds=0)


def test_network_retry_does_not_mask_fatal_run_failure(monkeypatch):
    class State:
        def __init__(self, _config):
            pass

        def summary(self):
            return {"status": "failed", "counts": {"failed": 2}}

    def fail(*_args, **_kwargs):
        raise RuntimeError("fatal run failure")

    monkeypatch.setattr(_module, "_wait_for_complete", fail)
    monkeypatch.setattr(_module, "JobState", State)
    monkeypatch.setattr(_module, "_only_transient_network_failures", lambda _: 2)
    monkeypatch.setattr(_module, "_service_state", lambda _: "inactive")
    monkeypatch.setattr(_module.subprocess, "run", lambda *_args, **_kwargs: (
        pytest.fail("must not retry a failed job")))
    with pytest.raises(RuntimeError, match="fatal run failure"):
        _module._wait_with_network_retry({}, Path("native.yaml"), "native.service",
                                         Path("imu-sim"), poll_seconds=1,
                                         retry_attempts=3, retry_delay_seconds=0)
