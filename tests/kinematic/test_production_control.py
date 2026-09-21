"""Production API boundaries and frozen configuration identity."""
from __future__ import annotations

import json
from pathlib import Path
from threading import Thread
import time

import pytest
import yaml
from fastapi.testclient import TestClient

from imu_motion_simulator.cli import main
from imu_motion_simulator.production import (JobState, load_job_config,
                                             run_pending_snapshots)
from imu_motion_simulator.production.control import create_app
from imu_motion_simulator.production.dashboard import create_app as create_dashboard
from imu_motion_simulator.production.runner import run_job
from imu_motion_simulator.production.upload_queue import (
    DurablePublisher, blocking_upload_error, transient_upload_error)
from imu_motion_simulator.production.upload_runtime import SCHEMA as UPLOAD_RUNTIME_SCHEMA


ROOT = Path(__file__).resolve().parents[2]


def _configuration(tmp_path):
    for name in ("library", "catalog", "checkout"):
        (tmp_path / name).mkdir()
    (tmp_path / "catalog/catalog.json").write_text(json.dumps({
        "schema": "imu_motion_simulator.amass_catalog.v1", "sources": []}))
    policy = tmp_path / "policy.json"
    policy.write_bytes((ROOT / "configs/quality/machine-review-v1.json").read_bytes())
    path = tmp_path / "job.yaml"
    path.write_text(yaml.safe_dump({
        "schema": "imu_motion_simulator.production_job.v1",
        "library_root": "library", "catalog": "catalog", "checkout": "checkout",
        "output": "output", "policy": "policy.json", "sensor_workers": 1,
        "sources": [], "clips_per_source": None,
        "publication": {"target": "dev", "backend": "local", "bucket": None,
                        "project": None, "root": "objects", "run_id": "job-test"},
    }))
    return path, policy


def test_control_api_validates_job_identity_and_actions(tmp_path):
    config_path, _ = _configuration(tmp_path)
    config = load_job_config(config_path)
    with TestClient(create_app(config)) as client:
        job = client.get("/api/v1/jobs").json()["jobs"][0]
        job_id = job["job_id"]
        assert job["status"] == "queued"
        assert client.get("/api/v1/jobs/unknown").status_code == 404
        assert client.post(f"/api/v1/jobs/{job_id}/pause").status_code == 409
        assert client.post(f"/api/v1/jobs/{job_id}/cancel").status_code == 200
        assert client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "cancelled"


def test_installed_production_cli_validates_outside_checkout(tmp_path, monkeypatch, capsys):
    config_path, _ = _configuration(tmp_path)
    monkeypatch.chdir(tmp_path / "library")
    assert main(["production", "validate", str(config_path)]) == 0
    assert json.loads(capsys.readouterr().out)["planned_clips"] == 0


def test_public_snapshot_api_uses_the_frozen_job(tmp_path):
    config_path, policy = _configuration(tmp_path)
    (tmp_path / "catalog/catalog.json").write_text(json.dumps({
        "schema": "imu_motion_simulator.amass_catalog.v1", "sources": [],
        "inputs": {"smplh_archive": "smplh.tar.xz",
                   "dmpl_archive": "dmpl.tar.xz", "layout": "layout.json"}}))
    (tmp_path / "library/smplh.tar.xz").write_bytes(b"model-placeholder")
    (tmp_path / "checkout/layout.json").write_text("{}")
    config = load_job_config(config_path)
    JobState(config)
    assert run_pending_snapshots(config) == []
    policy.write_text(policy.read_text().replace("0.8", "0.7", 1))
    with pytest.raises(ValueError, match="different frozen configuration"):
        run_pending_snapshots(load_job_config(config_path))


def test_job_rejects_policy_change_at_same_path(tmp_path):
    config_path, policy = _configuration(tmp_path)
    JobState(load_job_config(config_path))
    policy.write_text(policy.read_text().replace("0.8", "0.7", 1))
    with pytest.raises(ValueError, match="different frozen configuration"):
        JobState(load_job_config(config_path))


def test_job_rejects_unsafe_source_list(tmp_path):
    config_path, _ = _configuration(tmp_path)
    value = yaml.safe_load(config_path.read_text())
    value["sources"] = [["unhashable"]]
    config_path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="unique source names"):
        load_job_config(config_path)


def test_resume_rejects_changed_source_inputs(tmp_path):
    config_path, _ = _configuration(tmp_path)
    state = JobState(load_job_config(config_path))
    identity = {"plan_sha256": "a", "archive_sha256": "b", "model_sha256": "c"}
    state.bind_source("ACCAD", identity)
    state.bind_source("ACCAD", identity)
    with pytest.raises(ValueError, match="Frozen source inputs changed"):
        state.bind_source("ACCAD", {**identity, "archive_sha256": "new"})
    assert [event["kind"] for event in state.events_since(0)].count("source-bound") == 1


@pytest.mark.parametrize("terminal", ["complete", "cancelled"])
def test_terminal_job_cannot_restart(tmp_path, terminal):
    config_path, _ = _configuration(tmp_path)
    config = load_job_config(config_path)
    state = JobState(config)
    state.set_status(terminal)
    with pytest.raises(ValueError, match="cannot be restarted"):
        run_job(config)
    assert state.summary()["status"] == terminal


def test_durable_upload_intent_survives_restart_and_retries_without_recomputing(tmp_path):
    config_path, _ = _configuration(tmp_path)
    config = load_job_config(config_path)
    config["upload"] = {"mode": "durable", "workers": 1,
                        "max_pending_bytes": 100, "min_free_bytes": 0}
    state = JobState(config)
    payload = {"corpus": {"identity": "frozen"}, "candidate": {"candidate_id": "one"},
               "bundle": str(tmp_path / "bundle")}
    state.enqueue_upload("one", "ACCAD", payload, 60)
    assert state.summary()["counts"] == {"pending-publish": 1}
    assert state.upload_summary()["bytes"] == 60
    first_claim = state.claim_upload()
    assert first_claim[0] == "one"
    assert first_claim[3] >= 0
    # A killed uploader leaves a claim; the next owner requeues the same intent.
    resumed = JobState(config)
    resumed.reset_upload_claims()
    assert resumed.claim_upload()[1] == payload
    resumed.retry_upload("one", "temporary connection timeout", 0)
    assert resumed.claim_upload()[0] == "one"
    resumed.finish_upload("one", "commits/one.json", 60, 0.2)
    assert resumed.summary()["counts"] == {"published": 1}
    assert resumed.upload_summary()["count"] == 0
    assert resumed.phase_summary()[0]["phase"] == "upload"


def test_source_timing_accumulates_archive_work_across_resume(tmp_path):
    config_path, _ = _configuration(tmp_path)
    state = JobState(load_job_config(config_path))
    state.record_source_phase("GRAB", "archive-hash", 1.5)
    state.record_source_phase("GRAB", "archive-hash", 2.0)
    state.record_source_phase("GRAB", "archive-stream-next", 4.0)
    assert state.source_phase_summary() == [
        {"source_dataset": "GRAB", "phase": "archive-hash",
         "elapsed_s": 3.5, "samples": 2},
        {"source_dataset": "GRAB", "phase": "archive-stream-next",
         "elapsed_s": 4.0, "samples": 1},
    ]
    with pytest.raises(ValueError, match="Invalid production source phase"):
        state.record_source_phase("GRAB", "other", 1.0)


def test_operational_profile_preserves_frozen_job_identity_and_blocked_intent(tmp_path):
    config_path, _ = _configuration(tmp_path)
    config = load_job_config(config_path)
    state = JobState(config)
    digest = state.summary()["config_sha256"]
    profile = {"schema": UPLOAD_RUNTIME_SCHEMA, "mode": "durable", "workers": 4,
               "max_pending_bytes": 64 * 1024**3, "min_free_bytes": 100 * 1024**3}
    state.bind_upload_runtime(profile)
    state.bind_upload_runtime(profile)
    assert JobState(config).upload_runtime() == profile
    assert state.summary()["config_sha256"] == digest
    payload = {"candidate": {"candidate_id": "one"}}
    state.enqueue_upload("one", "ACCAD", payload, 100)
    assert state.claim_upload()[0] == "one"
    state.block_upload("one", "403 Forbidden")
    assert state.upload_summary()["blocked"] == 1
    state.reset_upload_claims()
    assert state.claim_upload() is None
    assert state.unblock_uploads() == 1
    assert state.claim_upload()[1] == payload
    state.finish_upload("one", "prod/one.json", 100, 0.1)
    assert state.summary()["counts"] == {"published": 1}
    state.set_status("running")
    with pytest.raises(ValueError, match="stopped"):
        state.bind_upload_runtime(profile)


def test_auth_and_remote_identity_conflicts_do_not_retry_forever():
    class Forbidden(Exception):
        code = 403
    assert blocking_upload_error(Forbidden("storage.googleapis.com"))
    assert not transient_upload_error(Forbidden("storage.googleapis.com"))
    assert blocking_upload_error(ValueError("Remote object could not be verified: key"))
    assert not transient_upload_error(ValueError("Remote commit could not be verified"))
    assert transient_upload_error(TimeoutError("storage.googleapis.com timed out"))
    assert not transient_upload_error(ValueError("storage.googleapis.com local hash changed"))


def test_frozen_job_resumes_and_drains_queued_clip_from_bound_runtime(tmp_path,
                                                                     monkeypatch):
    config_path, _ = _configuration(tmp_path)
    config = load_job_config(config_path)
    state = JobState(config)
    state.enqueue_upload("one", "ACCAD", {
        "corpus": {}, "candidate": {"candidate_id": "one"},
        "bundle": str(tmp_path / "frozen-bundle")}, 12)
    profile = {"schema": UPLOAD_RUNTIME_SCHEMA, "mode": "durable", "workers": 1,
               "max_pending_bytes": 1000, "min_free_bytes": 0}
    attempts = []
    class MetricStore:
        def __init__(self):
            self.calls = 0
        def metrics_snapshot(self):
            return {"sdk_calls": self.calls}
    monkeypatch.setattr("imu_motion_simulator.production.runner._store",
                        lambda _: MetricStore())
    def fake_publish(_corpus, candidate, _bundle, store, **_kwargs):
        store.calls += 1
        attempts.append(candidate["candidate_id"])
        if len(attempts) == 1:
            raise ConnectionError("offline")
        return {"commit_key": "dev/one.json"}
    monkeypatch.setattr("imu_motion_simulator.production.upload_queue.publish_candidate",
                        fake_publish)
    monkeypatch.setattr("imu_motion_simulator.production.upload_queue.random.uniform",
                        lambda *_: 0)
    report = run_job(config, upload_profile=profile)
    assert report["status"] == "complete"
    assert attempts == ["one", "one"]
    assert report["counts"] == {"published": 1}
    assert report["upload_runtime"] == profile
    assert JobState(config).summary()["upload_queue"]["count"] == 0
    phases = {row["phase"]: row for row in JobState(config).phase_summary()}
    assert phases["upload-queue-age-to-claim"]["count"] == 1
    with JobState(config)._connect() as db:
        metrics = [json.loads(row[0]) for row in db.execute(
            "SELECT metrics_json FROM upload_transport")]
    assert sum(row["sdk_calls"] for row in metrics) == 2
    assert sum(row["worker_claims"] for row in metrics) == 2


def test_auth_failure_halts_publisher_without_losing_upload_intent(tmp_path,
                                                                  monkeypatch):
    config_path, _ = _configuration(tmp_path)
    config = load_job_config(config_path)
    config["upload"] = {"mode": "durable", "workers": 1,
                        "max_pending_bytes": 1000, "min_free_bytes": 0}
    state = JobState(config)
    state.enqueue_upload("one", "ACCAD", {
        "corpus": {}, "candidate": {"candidate_id": "one"},
        "bundle": str(tmp_path / "bundle")}, 12)
    class Forbidden(Exception):
        code = 403
    monkeypatch.setattr("imu_motion_simulator.production.upload_queue.publish_candidate",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            Forbidden("403 Forbidden")))
    publisher = DurablePublisher(state, config, lambda _: object())
    publisher.start()
    try:
        assert publisher.stop_event.wait(2)
        with pytest.raises(RuntimeError, match="Upload worker failed"):
            publisher.drain()
        assert state.upload_summary()["blocked"] == 1
        assert state.summary()["counts"] == {"pending-publish": 1}
        assert state.unblock_uploads() == 1
    finally:
        publisher.stop()


def test_durable_backpressure_releases_after_upload(tmp_path):
    config_path, _ = _configuration(tmp_path)
    config = load_job_config(config_path)
    config["upload"] = {"mode": "durable", "workers": 1,
                        "max_pending_bytes": 50, "min_free_bytes": 0}
    state = JobState(config)
    state.enqueue_upload("one", "ACCAD", {"bundle": "frozen"}, 60)
    publisher = DurablePublisher(state, config, lambda _: None)
    result = []
    waiter = Thread(target=lambda: result.append(publisher.wait_for_capacity()))
    waiter.start()
    time.sleep(0.05)
    assert waiter.is_alive()
    state.claim_upload()
    state.finish_upload("one", "commits/one.json", 60, 0.1)
    waiter.join(timeout=2)
    assert result == [True]
    assert transient_upload_error(ConnectionError("offline"))
    assert not transient_upload_error(ValueError("Local object changed"))


def test_dashboard_reads_current_ledger_and_export_is_explicit(tmp_path, monkeypatch):
    config_path, _ = _configuration(tmp_path)
    state = JobState(load_job_config(config_path))
    state.record_clip("one", "ACCAD", "published", commit_key="dev/one.json",
                      published_bytes=125)
    state.record_phase("one", "prepare", 2.0)
    state.record_phase("one", "upload", 1.0)
    state.record_source_phase("ACCAD", "archive-hash", 0.5)
    state.record_upload_metrics("worker-1", {"worker_claims": 1,
                                              "worker_busy_s": 1.0,
                                              "sdk_get_blob_calls": 2})
    def fake_export(config, *, additional_configs=()):
        assert config["output"] == str(tmp_path / "output")
        assert not additional_configs
        return {"export_id": "frozen", "candidate_count": 1,
                "duration_s": 12.0}
    monkeypatch.setattr("imu_motion_simulator.provisional.export_provisional", fake_export)
    with TestClient(create_dashboard([config_path], units=())) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "合成 IMU 生产控制面板" in page.text
        overview = client.get("/api/v1/dashboard").json()
        assert overview["jobs"][0]["counts"]["published"] == 1
        assert overview["jobs"][0]["phases"][0]["phase"] == "prepare"
        assert overview["jobs"][0]["source_phases"][0]["elapsed_s"] == 0.5
        assert overview["jobs"][0]["upload_transport"]["sdk_get_blob_calls"] == 2
        assert overview["export"]["status"] == "idle"
        assert client.post("/api/v1/export-core").status_code == 403
        started = client.post("/api/v1/export-core", headers={
            "X-IMU-Dashboard": "export-core"})
        assert started.status_code == 200
        for _ in range(100):
            finished = client.get("/api/v1/dashboard").json()["export"]
            if finished["status"] != "running":
                break
            time.sleep(0.01)
        assert finished["result"]["export_id"] == "frozen"
        assert state.summary()["counts"] == {"published": 1}


def test_phase_summary_has_percentiles(tmp_path):
    config_path, _ = _configuration(tmp_path)
    state = JobState(load_job_config(config_path))
    for clip, seconds in zip("abcd", (1.0, 2.0, 3.0, 4.0)):
        state.record_phase(clip, "prepare", seconds)
    summary = state.phase_summary()[0]
    assert summary["count"] == 4
    assert summary["total_s"] == 10.0
    assert summary["p50_s"] == 2.5
    assert summary["p95_s"] == pytest.approx(3.85)


def test_durable_worker_retries_offline_store_and_drains_existing_queue(tmp_path,
                                                                         monkeypatch):
    config_path, _ = _configuration(tmp_path)
    config = load_job_config(config_path)
    config["upload"] = {"mode": "durable", "workers": 2,
                        "max_pending_bytes": 1000, "min_free_bytes": 0}
    state = JobState(config)
    attempts = []
    def store_factory(_):
        attempts.append("connect")
        if len(attempts) == 1:
            raise ConnectionError("offline")
        return object()
    def fake_publish(corpus, candidate, bundle, store, **kwargs):
        assert candidate["candidate_id"] == "one"
        return {"commit_key": "dev/one.json"}
    monkeypatch.setattr("imu_motion_simulator.production.upload_queue.publish_candidate",
                        fake_publish)
    # Keep the test fast while still exercising the real retry transition.
    monkeypatch.setattr("imu_motion_simulator.production.upload_queue.random.uniform",
                        lambda *_: 0.0)
    publisher = DurablePublisher(state, config, store_factory)
    published_events = []
    def faulty_progress():
        published_events.append("one")
        raise RuntimeError("display disconnected")
    publisher.on_published = faulty_progress
    publisher.start()
    try:
        state.enqueue_upload("one", "ACCAD", {
            "corpus": {}, "candidate": {"candidate_id": "one"},
            "bundle": str(tmp_path / "bundle")}, 60)
        deadline = time.monotonic() + 3
        while ((state.upload_summary()["count"] or not published_events
                or "progress-callback-error" not in [event["kind"]
                                                     for event in state.events_since(0)])
               and time.monotonic() < deadline):
            time.sleep(0.02)
        assert state.summary()["counts"] == {"published": 1}
        assert len(attempts) >= 2
        assert published_events == ["one"]
        assert "progress-callback-error" in [event["kind"]
                                             for event in state.events_since(0)]
        publisher.drain()
    finally:
        publisher.stop()
