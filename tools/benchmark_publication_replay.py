"""Replay existing frozen dev clip payloads into an isolated GCS dev prefix.

This measures publication, not archive decoding or IMU computation. It never
changes the original dev jobs, their payloads, or the prod prefix.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

from imu_motion_simulator.contracts.common import sha256_file
from imu_motion_simulator.pipeline.machine_review import load_machine_policy
from imu_motion_simulator.production import JobState, load_job_config
from imu_motion_simulator.production.runner import _candidate_corpus, _store
from imu_motion_simulator.production.upload_queue import DurablePublisher
from imu_motion_simulator.production.upload_runtime import load_upload_profile


def _one(paths: list[Path], role: str) -> Path:
    if len(paths) != 1:
        raise ValueError(f"Expected one {role}, found {len(paths)}")
    return paths[0]


def _payloads(config_paths: list[Path], per_source: int):
    grouped = defaultdict(list)
    for config_path in config_paths:
        config = load_job_config(config_path)
        root = Path(config["output"])
        outbox = root / "outbox" / "dev" / config["publication"]["run_id"]
        catalog = Path(config["catalog"])
        manifest = json.loads((catalog / "catalog.json").read_text())
        plans = {row["source_dataset"]: catalog / row["plan"]
                 for row in manifest["sources"] if row["status"] == "plan-ready"}
        policy_path = Path(config["policy"])
        policy = load_machine_policy(policy_path)
        for receipt in outbox.glob("*.json"):
            if receipt.name.endswith(".feed.json"):
                continue
            commit = json.loads(receipt.read_text())
            if commit["schema"] != "imu_motion_simulator.candidate_commit.v1":
                raise ValueError("Unexpected source receipt")
            grouped[commit["source_dataset"]].append(
                (commit, root, plans[commit["source_dataset"]], policy_path, policy))
    for source in sorted(grouped):
        for commit, root, plan_path, policy_path, policy in sorted(
                grouped[source], key=lambda row: row[0]["candidate_id"])[:per_source]:
            clip_id = commit["candidate_id"]
            run = root / "sources" / source / "run"
            paths = {
                "motion": run / "motions" / f"{clip_id}.motion.h5",
                "selection": run / "selections" / f"{clip_id}.selection.json",
                "sensors": _one(list((run / "sensors").glob(f"{clip_id}-*.sensors.h5")),
                                "sensors"),
            }
            bundle = _one(list((run / "reviews").glob(f"{clip_id}-*.review")),
                          "review bundle")
            clip = {"id": clip_id, "source_dataset": source,
                    "source_member": commit["source_member"],
                    "label_candidates": commit["label_candidates"]}
            corpus, candidate = _candidate_corpus(
                clip, paths, {"warnings": commit["warning_flags"]},
                policy_path, policy, plan_path)
            if (candidate["objects"] != {row["role"]: row["sha256"]
                                         for row in commit["objects"]}
                    or commit["policy_sha256"] != corpus["policy"]["sha256"]):
                raise ValueError("Source receipt differs from frozen local payload")
            size = sum(path.stat().st_size for path in paths.values())
            size += sum(path.stat().st_size for path in bundle.iterdir()
                        if path.is_file())
            yield clip_id, source, {"corpus": corpus, "candidate": candidate,
                                    "bundle": str(bundle)}, size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native_config", type=Path)
    parser.add_argument("stageii_config", type=Path)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--per-source", type=int, default=5)
    parser.add_argument("--prepare-only", action="store_true",
                        help="validate and fingerprint payloads without enqueueing/uploading")
    args = parser.parse_args()
    if args.per_source < 1 or not args.run_id.startswith("perf-replay-"):
        parser.error("Use a positive limit and an isolated perf-replay-* run ID")
    native = load_job_config(args.native_config)
    stageii = load_job_config(args.stageii_config)
    if native["publication"] != stageii["publication"]:
        raise ValueError("Source jobs must share one dev publication target")
    publication = {**native["publication"], "run_id": args.run_id}
    if publication["target"] != "dev" or publication["backend"] != "gcs":
        raise ValueError("Replay benchmark is restricted to isolated GCS dev")
    profile = load_upload_profile(args.profile)
    config = {"output": str(args.output.resolve()), "publication": publication}
    state = JobState(config)
    state.bind_upload_runtime(profile)
    if state.summary()["status"] not in {"queued", "failed", "paused"}:
        raise ValueError("Benchmark job is not resumable")
    payloads = list(_payloads([args.native_config, args.stageii_config],
                              args.per_source))
    sources = {source for _, source, _, _ in payloads}
    if len(sources) != 22 or len(payloads) != 22 * args.per_source:
        raise ValueError(f"Expected {22 * args.per_source} clips from 22 sources; "
                         f"got {len(payloads)} from {len(sources)}")
    identity = [{"clip_id": clip_id,
                 "objects": payload["candidate"]["objects"],
                 "bundle_manifest_sha256": sha256_file(
                     Path(payload["bundle"]) / "manifest.json")}
                for clip_id, _, payload, _ in payloads]
    workload_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    (state.root / "workload.json").write_text(json.dumps({
        "sources": 22, "clips": len(payloads), "workload_sha256": workload_hash,
        "items": identity}, indent=2) + "\n")
    if args.prepare_only:
        print(json.dumps({"workload_sha256": workload_hash,
                          "sources": 22, "clips": len(payloads)}))
        return
    for clip_id, source, payload, size in payloads:
        if state.clip_status(clip_id) != "published":
            state.enqueue_upload(clip_id, source, payload, size)
    state.request("run")
    state.set_status("running")
    publisher = DurablePublisher(state, {**config, "upload": profile}, _store)
    started = time.monotonic()
    try:
        publisher.start()
        publisher.drain()
        if state.summary()["counts"] != {"published": len(payloads)}:
            raise RuntimeError("Replay publication did not settle all clips")
        state.set_status("complete")
    except Exception:
        state.set_status("failed")
        raise
    finally:
        publisher.stop()
    result = {"schema": "imu_motion_simulator.publication_replay_benchmark.v1",
              "workload_sha256": workload_hash, "sources": 22,
              "clips": len(payloads), "workers": profile["workers"],
              "run_id": args.run_id, "wall_s": time.monotonic() - started,
              "queue": state.upload_summary(), "phases": state.phase_summary(),
              "counts": state.summary()["counts"]}
    (state.root / "benchmark-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
