"""Run frozen, isolated GCS-dev pipeline comparisons after prod finishes.

Each manifest supplies distinct outputs and run IDs. The script waits for the
formal prod audit, full export and manual dashboard export to finish before it
starts any measurement. It never changes a production job or publication key.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from threading import Event, Thread
from urllib.request import urlopen

from imu_motion_simulator.production.config import config_digest, load_job_config
from imu_motion_simulator.production.runner import _source_plan_rows
from imu_motion_simulator.production.upload_runtime import load_upload_profile
from imu_motion_simulator.publication import GcsPublicationStore, publication_prefix
from tools.production_prod_chain import _audit_cloud

def _save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _prod_gate(audit: Path, dashboard: str, units: tuple[str, ...] = ()) -> str:
    try:
        phase = json.loads(audit.read_text())["phase"]
    except (FileNotFoundError, ValueError, KeyError) as error:
        return f"audit unreadable: {error}"
    if phase == "failed":
        raise RuntimeError("Formal prod cloud audit failed; benchmark must not start")
    if phase != "complete":
        return f"audit {phase}"
    for unit in units:
        state = subprocess.run(["systemctl", "--user", "is-active", unit],
                               capture_output=True, text=True, timeout=5,
                               check=False).stdout.strip()
        if state == "active":
            return f"{unit} active"
    try:
        with urlopen(dashboard, timeout=5) as response:
            export = json.load(response).get("export", {})
    except (OSError, ValueError) as error:
        return f"dashboard unavailable: {error}"
    if export.get("status") == "running":
        return "manual core export running"
    return "ready"


def _expected_sources(manifest: dict) -> dict:
    if "sources" in manifest:
        return manifest["sources"]
    return {manifest["source"]: {
        "source_plan_sha256": manifest["source_plan_sha256"],
        "clip_count": manifest["clip_count"],
        "clip_ids_sha256": manifest["clip_ids_sha256"],
    }}


def _validate_workload(config: dict, expected: dict) -> None:
    publication = config["publication"]
    if publication["target"] != "dev" or publication["backend"] != "gcs":
        raise ValueError("Benchmark may publish only to isolated GCS dev")
    rows = _source_plan_rows(config)
    catalog = json.loads((Path(config["catalog"]) / "catalog.json").read_text())
    plan_hashes = {row["source_dataset"]: row["plan_sha256"]
                   for row in catalog["sources"]}
    if {source for source, _, _, _ in rows} != set(expected):
        raise ValueError("Benchmark sources differ from frozen manifest")
    for source, _, _, clips in rows:
        ids = [clip["id"] for clip in clips]
        digest = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
        identity = expected[source]
        if (len(ids) != identity["clip_count"]
                or digest != identity["clip_ids_sha256"]
                or plan_hashes[source] != identity["source_plan_sha256"]):
            raise ValueError(f"Frozen benchmark clips changed: {source}")


def _queue_samples(path: Path, output: Path, stop: Event) -> None:
    with path.open("w") as handle:
        while not stop.wait(2):
            ledger = output / "production.sqlite3"
            if not ledger.is_file():
                continue
            try:
                with sqlite3.connect(f"file:{ledger}?mode=ro", uri=True,
                                     timeout=2) as db:
                    queued = db.execute("SELECT COUNT(*), COALESCE(SUM(byte_length),0) "
                                        "FROM upload_queue").fetchone()
                sample = {"elapsed_monotonic_s": time.monotonic(),
                          "queue_count": queued[0], "queue_bytes": queued[1],
                          "free_bytes": shutil.disk_usage(output).free}
                handle.write(json.dumps(sample) + "\n")
                handle.flush()
            except (OSError, sqlite3.Error):
                continue


def _run_one(row: dict, expected: dict, profile: Path, result_dir: Path) -> dict:
    config_path = Path(row["config"])
    config = load_job_config(config_path)
    _validate_workload(config, expected)
    if config["publication"]["run_id"] != row["run_id"]:
        raise ValueError("Benchmark run ID differs from manifest")
    output = Path(config["output"])
    suffix = row["suffix"]
    if output.exists():
        raise FileExistsError(f"Benchmark output already exists: {output}")
    result_dir.mkdir(parents=True, exist_ok=True)
    log = result_dir / f"{suffix}.log"
    telemetry = []
    for name, command in (("cpu", ["sar", "-u", "1"]),
                          ("network", ["sar", "-n", "DEV", "1"]),
                          ("disk", ["iostat", "-xz", "1"])):
        handle = (result_dir / f"{suffix}-{name}.txt").open("w")
        telemetry.append((subprocess.Popen(command, stdout=handle,
                                           stderr=subprocess.STDOUT), handle))
    stop = Event()
    monitor = Thread(target=_queue_samples,
                     args=(result_dir / f"{suffix}-queue.jsonl", output, stop),
                     daemon=True)
    monitor.start()
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    command = [sys.executable, "-m", "imu_motion_simulator.cli", "production",
               "run", str(config_path), "--upload-profile", str(profile)]
    started = time.monotonic()
    try:
        with log.open("w") as handle:
            run_process = subprocess.run(command, env=env, stdout=handle,
                                         stderr=subprocess.STDOUT, check=False)
    finally:
        wall_s = time.monotonic() - started
        stop.set()
        monitor.join(timeout=5)
        for telemetry_process, handle in telemetry:
            if telemetry_process.poll() is None:
                telemetry_process.terminate()
            telemetry_process.wait(timeout=5)
            handle.close()
    if run_process.returncode:
        raise RuntimeError(f"Benchmark {suffix} exited {run_process.returncode}; see {log}")
    report = json.loads((output / "production-report.json").read_text())
    with sqlite3.connect(output / "production.sqlite3") as db:
        clips = db.execute("SELECT clip_id, source_dataset, status, commit_key "
                           "FROM clips").fetchall()
        transport = {}
        for (metrics_json,) in db.execute("SELECT metrics_json FROM upload_transport"):
            for key, value in json.loads(metrics_json).items():
                transport[key] = transport.get(key, 0) + value
    expected_count = sum(item["clip_count"] for item in expected.values())
    counts = report["counts"]
    if (report["status"] != "complete" or sum(counts.values()) != expected_count
            or counts.get("failed", 0) or counts.get("pending-publish", 0)):
        raise RuntimeError(f"Benchmark {suffix} has incomplete ledger: {counts}")
    commits = {key: (clip_id, source) for clip_id, source, status, key in clips
               if status == "published"}
    store = GcsPublicationStore(config["publication"]["bucket"],
                                config["publication"]["project"])
    cloud = _audit_cloud(store, publication_prefix("dev", row["run_id"]), commits)
    return {"suffix": suffix, "sensor_workers": config["sensor_workers"],
            "upload_workers": 8, "config_sha256": config_digest(config),
            "run_id": row["run_id"], "wall_s": wall_s,
            "report_wall_s": report["run_wall_seconds"],
            "first_candidate_latency_s": report["first_candidate_latency_s"],
            "counts": counts, "phases": report["phase_metrics"],
            "source_phases": report["source_phase_metrics"],
            "upload_transport": transport, "cloud_audit": cloud}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--dashboard", default="http://127.0.0.1:8891/api/v1/dashboard")
    parser.add_argument("--prod-unit", action="append", default=[],
                        help="systemd user unit that must be inactive; repeat as needed")
    parser.add_argument("--wait-hours", type=float, default=8)
    parser.add_argument("--check-only", action="store_true",
                        help="validate frozen workloads and report the production gate")
    args = parser.parse_args()
    profile = load_upload_profile(args.profile)
    if profile["mode"] != "durable" or profile["workers"] != 8:
        raise ValueError("Phase comparison requires the frozen 8-worker upload profile")
    manifests = [json.loads(path.read_text()) for path in args.manifest]
    rows = [(row, _expected_sources(manifest))
            for manifest in manifests for row in manifest["runs"]]
    if args.check_only:
        for row, expected in rows:
            config = load_job_config(row["config"])
            _validate_workload(config, expected)
            if config["publication"]["run_id"] != row["run_id"]:
                raise ValueError("Benchmark run ID differs from manifest")
        print(json.dumps({"runs": [row["suffix"] for row, _ in rows],
                          "prod_gate": _prod_gate(
                              args.audit_report, args.dashboard,
                              tuple(args.prod_unit))}))
        return
    result_dir = args.manifest[0].parent / "results"
    result_dir.mkdir(exist_ok=True)
    status_path = result_dir / "status.json"
    deadline = time.monotonic() + args.wait_hours * 3600
    try:
        while True:
            reason = _prod_gate(args.audit_report, args.dashboard,
                                tuple(args.prod_unit))
            _save(status_path, {"phase": "waiting" if reason != "ready" else "ready",
                                "reason": reason})
            if reason == "ready":
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Production did not finish within wait window: {reason}")
            time.sleep(30)
        results = []
        for row, expected in rows:
            if _prod_gate(args.audit_report, args.dashboard,
                          tuple(args.prod_unit)) != "ready":
                raise RuntimeError("Production activity resumed during benchmark")
            _save(status_path, {"phase": "running", "run": row["suffix"],
                                "completed": [item["suffix"] for item in results]})
            result = _run_one(row, expected, args.profile, result_dir)
            _save(result_dir / f"{row['suffix']}.json", result)
            results.append(result)
        _save(status_path, {"phase": "complete",
                            "completed": [item["suffix"] for item in results]})
    except Exception as error:
        _save(status_path, {"phase": "failed", "error": str(error)})
        raise


if __name__ == "__main__":
    main()
