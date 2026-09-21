import hashlib
import json
import time

import pytest

from imu_motion_simulator.contracts.common import sha256_file
from imu_motion_simulator.publication import (
    GcsPublicationStore, LocalPublicationStore, choose_pilot, publish_candidate,
)
from imu_motion_simulator.snapshot_worker import (pending_snapshot_ids,
                                                  retry_failed_snapshot, run_pending_once)


def test_local_store_create_only_and_verified_retry(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"good")
    digest = sha256_file(source)
    store = LocalPublicationStore(tmp_path / "store")
    store.put_file(source, "synthetic-motion/dev/run/v1/objects/file", digest)
    store.put_file(source, "synthetic-motion/dev/run/v1/objects/file", digest)
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        store.put_file(source, "synthetic-motion/dev/run/v1/objects/file", digest)
    assert (tmp_path / "store/synthetic-motion/dev/run/v1/objects/file").read_bytes() == b"good"
    with pytest.raises(ValueError, match="Unsafe"):
        store.put_json("../escape", {"key": "value"})


def test_gcs_viewer_cache_only_after_verified_content_addressed_upload(tmp_path):
    source = tmp_path / "index.html"
    source.write_bytes(b"viewer")
    digest = sha256_file(source)
    class Blob:
        def __init__(self):
            self.metadata = {}
        def upload_from_filename(self, *_args, **_kwargs):
            bucket.uploads += 1
    class Bucket:
        uploads = 0
        reads = 0
        def blob(self, _key):
            return Blob()
        def get_blob(self, _key):
            self.reads += 1
            return type("Remote", (), {"size": source.stat().st_size,
                                         "metadata": {"sha256": digest}})()
    bucket = Bucket()
    store = GcsPublicationStore.__new__(GcsPublicationStore)
    store.bucket = bucket
    store._verified_viewer = {}
    store.metrics = {"sdk_calls": 0, "sdk_seconds": 0.0,
                     "cache_hits": 0, "upload_errors": 0,
                     "precondition_exists": 0,
                     "attempted_upload_bytes": 0, "verified_bytes": 0}
    viewer_key = f"synthetic-motion/prod/v1/viewer/{digest}/index.html"
    store.put_file(source, viewer_key, digest)
    store.put_file(source, viewer_key, digest)
    assert (bucket.uploads, bucket.reads, store.metrics["cache_hits"]) == (1, 1, 1)
    ordinary_key = f"synthetic-motion/prod/v1/objects/{digest}/index.html"
    store.put_file(source, ordinary_key, digest)
    store.put_file(source, ordinary_key, digest)
    assert (bucket.uploads, bucket.reads) == (3, 3)
    metrics = store.metrics_snapshot()
    assert metrics["sdk_upload_from_filename_calls"] == 3
    assert metrics["sdk_get_blob_calls"] == 3
    assert metrics["sdk_upload_from_filename_seconds"] >= 0
    assert metrics["sdk_get_blob_seconds"] >= 0


def test_pilot_selection_has_four_strata_and_stageii():
    rows = [
        {"candidate_id": f"{category}-{index}", "category": category,
         "source": "GRAB" if category == "warning" and index == 4 else "ACCAD"}
        for category in ("ordinary", "warning", "label-ambiguous", "high-risk")
        for index in range(5)
    ]
    chosen = choose_pilot({"schema": "imu_motion_simulator.review_sample_index.v1",
                           "bundles": rows})
    assert len(chosen) == 12
    assert {row["category"] for row in chosen} == {
        "ordinary", "warning", "label-ambiguous", "high-risk"}
    assert any(row["source"] == "GRAB" for row in chosen)
    assert len({row["candidate_id"] for row in chosen}) == 12


def test_publish_commit_is_last_and_retry_is_idempotent(tmp_path, monkeypatch):
    import imu_motion_simulator.publication as publication

    monkeypatch.setattr(publication, "validate_candidate_corpus", lambda corpus: None)
    monkeypatch.setattr(publication, "validate_bundle", lambda bundle: {"decision": "unreviewed"})
    sources = {}
    for role in ("motion", "sensors", "selection"):
        path = tmp_path / role
        path.write_text(json.dumps({"selection_id": "selection-1", "motion_sha256": "M"})
                        if role == "selection" else role)
        sources[role] = path
    digests = {role: sha256_file(path) for role, path in sources.items()}
    # The selection's recorded parent identity is independent of this fake file hash.
    (tmp_path / "selection").write_text(json.dumps({
        "selection_id": "selection-1", "motion_sha256": digests["motion"]}))
    digests["selection"] = sha256_file(tmp_path / "selection")
    candidate = {
        "candidate_id": "clip-1", "source_dataset": "ACCAD",
        "source_member": "ACCAD/clip-1.npz", "label_candidates": [],
        "warning_flags": [], "objects": digests,
    }
    corpus = {"policy": {"sha256": "a" * 64}, "candidates": [candidate],
              "objects": [
                  {"sha256": digest, "byte_length": path.stat().st_size,
                   "local_path": str(path),
                   "object_key": f"synthetic-motion/v1/objects/{digest}/{role}"}
                  for role, path in sources.items() for digest in [digests[role]]
              ]}
    bundle = tmp_path / "bundle"; bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({
        "motion_sha256": digests["motion"], "sensor_sha256": digests["sensors"],
        "selection": {"selection_id": "selection-1"},
        "qa": {"passed": True, "risk_score": .8}}))
    (bundle / "index.html").write_text("<html>clip</html>")
    root = tmp_path / "remote"; outbox = tmp_path / "outbox"

    class InterruptedStore(LocalPublicationStore):
        def __init__(self, path):
            super().__init__(path)
            self.calls = 0

        def put_file(self, source, key, digest):
            self.calls += 1
            if self.calls == 2:
                raise OSError("simulated interruption")
            super().put_file(source, key, digest)

    with pytest.raises(OSError, match="interruption"):
        publish_candidate(corpus, candidate, bundle, InterruptedStore(root),
                          run_id="pilot", outbox=outbox)
    assert not list(root.rglob("candidates/*.json"))
    store = LocalPublicationStore(root)
    first = publish_candidate(corpus, candidate, bundle, store,
                              run_id="pilot", outbox=outbox)
    second = publish_candidate(corpus, candidate, bundle, store,
                               run_id="pilot", outbox=outbox)
    assert first == second
    commit = json.loads((root / first["commit_key"]).read_text())
    assert commit["risk_score"] == .8
    assert commit["risk_tier"] == "high"
    assert commit["risk_policy_id"] == "kinematic-qa-v1-high-0.7"
    assert commit["version_id"] == hashlib.sha256(publication._json_bytes({
        "candidate_id": "clip-1", "policy_sha256": "a" * 64,
        "risk_policy_id": "kinematic-qa-v1-high-0.7",
        "objects": commit["objects"],
        "bundle_files": [{key: item[key] for key in ("name", "sha256", "byte_length")}
                         for item in commit["bundle_files"]],
    })).hexdigest()
    feed_keys = store.list_keys("synthetic-motion/dev/pilot/v1/index-feed")
    assert len(feed_keys) == 1
    feed = store.read_json(feed_keys[0])
    assert feed["schema"] == "imu_motion_simulator.candidate_index_feed.v1"
    assert feed["feed_key"] == feed_keys[0]
    assert feed["commit_key"] == first["commit_key"]
    assert feed["commit_sha256"] == hashlib.sha256(
        publication._json_bytes(commit)).hexdigest()


def test_failed_feed_reposts_after_later_commit_for_incremental_discovery(
        tmp_path, monkeypatch):
    import imu_motion_simulator.publication as publication

    monkeypatch.setattr(publication, "validate_candidate_corpus", lambda corpus: None)
    monkeypatch.setattr(publication, "validate_bundle", lambda bundle: {"decision": "unreviewed"})
    motion = tmp_path / "motion.h5"
    sensors = tmp_path / "sensors.h5"
    selection = tmp_path / "selection.json"
    motion.write_bytes(b"motion")
    sensors.write_bytes(b"sensors")
    selection.write_text(json.dumps({"selection_id": "selection-1",
                                     "motion_sha256": sha256_file(motion)}))
    files = {"motion": motion, "sensors": sensors, "selection": selection}
    digests = {role: sha256_file(path) for role, path in files.items()}
    candidates = [{"candidate_id": name, "source_dataset": "ACCAD",
                   "source_member": name + ".npz", "label_candidates": [],
                   "warning_flags": [], "objects": digests}
                  for name in ("clip-a", "clip-b")]
    corpus = {"policy": {"sha256": "a" * 64}, "candidates": candidates,
              "objects": [{"sha256": digests[role],
                           "byte_length": path.stat().st_size,
                           "local_path": str(path),
                           "object_key": f"synthetic-motion/v1/objects/{digests[role]}/{role}"}
                          for role, path in files.items()]}
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({
        "motion_sha256": digests["motion"], "sensor_sha256": digests["sensors"],
        "selection": {"selection_id": "selection-1"}, "qa": {"passed": True}}))

    class FailFirstFeed(LocalPublicationStore):
        failed_key = None

        def put_json(self, key, value):
            if "/index-feed/" in key and self.failed_key is None:
                self.failed_key = key
                raise OSError("feed upload interrupted")
            return super().put_json(key, value)

    store = FailFirstFeed(tmp_path / "store")
    outbox = tmp_path / "outbox"
    with pytest.raises(OSError, match="feed upload interrupted"):
        publish_candidate(corpus, candidates[0], bundle, store,
                          run_id="pilot", outbox=outbox)
    assert len(store.list_keys("synthetic-motion/dev/pilot/v1/candidates/")) == 1
    time.sleep(.002)
    publish_candidate(corpus, candidates[1], bundle, store,
                      run_id="pilot", outbox=outbox)
    later_key = store.list_keys("synthetic-motion/dev/pilot/v1/index-feed/")[0]
    time.sleep(.002)
    publish_candidate(corpus, candidates[0], bundle, store,
                      run_id="pilot", outbox=outbox)
    keys = store.list_keys("synthetic-motion/dev/pilot/v1/index-feed/")
    assert len(keys) == 2
    assert store.failed_key not in keys
    assert keys[0] == later_key
    assert store.read_json(keys[1])["candidate_id"] == "clip-a"


def test_auto_worker_selects_only_unhandled_requests(tmp_path, monkeypatch):
    store = LocalPublicationStore(tmp_path / "store")
    prefix = "synthetic-motion/dev/pilot/v1/snapshots"
    queued = "synthetic-" + "a" * 32
    failed = "synthetic-" + "b" * 32
    complete = "synthetic-" + "c" * 32
    for snapshot_id in (queued, failed, complete):
        store.put_json(f"{prefix}/requests/{snapshot_id}.json", {"snapshot_id": snapshot_id})
    store.put_json(f"{prefix}/failures/{failed}/attempt.json", {"state": "failed"})
    store.put_json(f"{prefix}/results/{complete}.json", {"state": "complete"})
    assert pending_snapshot_ids(store, "pilot") == [queued]
    import imu_motion_simulator.snapshot_worker as worker
    seen = []

    def fake_run(_store, _run_id, snapshot_id, output, **_kwargs):
        seen.append((snapshot_id, output))
        return {"state": "complete"}

    monkeypatch.setattr(worker, "run_snapshot", fake_run)
    results = run_pending_once(store, "pilot", tmp_path / "out",
                               model_archive=tmp_path / "model", dmpl_archive=None,
                               layout=tmp_path / "layout")
    assert results == [{"snapshot_id": queued, "state": "complete"}]
    assert seen == [(queued, tmp_path / "out" / queued)]


def test_snapshot_worker_keeps_dev_and_prod_requests_separate(tmp_path):
    store = LocalPublicationStore(tmp_path / "store")
    dev_id = "synthetic-" + "d" * 32
    prod_id = "synthetic-" + "e" * 32
    store.put_json(
        f"synthetic-motion/dev/pilot/v1/snapshots/requests/{dev_id}.json",
        {"snapshot_id": dev_id})
    store.put_json(
        f"synthetic-motion/prod/v1/snapshots/requests/{prod_id}.json",
        {"snapshot_id": prod_id})
    assert pending_snapshot_ids(store, "pilot") == [dev_id]
    assert pending_snapshot_ids(store, None, target="prod") == [prod_id]


def test_snapshot_retry_requires_failed_request_and_keeps_frozen_id(tmp_path, monkeypatch):
    store = LocalPublicationStore(tmp_path / "store")
    prefix = "synthetic-motion/dev/pilot/v1/snapshots"
    snapshot_id = "synthetic-" + "f" * 32
    store.put_json(f"{prefix}/requests/{snapshot_id}.json", {"snapshot_id": snapshot_id})
    options = {"model_archive": tmp_path / "model", "dmpl_archive": None,
               "layout": tmp_path / "layout"}
    with pytest.raises(ValueError, match="no recorded failure"):
        retry_failed_snapshot(store, "pilot", snapshot_id, tmp_path / "out", **options)
    store.put_json(f"{prefix}/failures/{snapshot_id}/attempt.json", {"state": "failed"})
    import imu_motion_simulator.snapshot_worker as worker
    calls = []
    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return {"state": "complete"}
    monkeypatch.setattr(worker, "run_snapshot", fake_run)
    assert retry_failed_snapshot(store, "pilot", snapshot_id, tmp_path / "out",
                                 **options) == {"state": "complete"}
    assert calls[0][0][2] == snapshot_id
    store.put_json(f"{prefix}/results/{snapshot_id}.json", {"state": "complete"})
    with pytest.raises(ValueError, match="already complete"):
        retry_failed_snapshot(store, "pilot", snapshot_id, tmp_path / "out", **options)
