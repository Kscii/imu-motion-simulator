"""Local-only queue/concurrency microbenchmark; never contacts a publication store."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time

from imu_motion_simulator.production import JobState
from imu_motion_simulator.production import upload_queue


def measure(root: Path, workers: int, clips: int, wait_s: float) -> dict:
    config = {
        "output": str(root / f"w{workers}"),
        "publication": {"target": "dev", "run_id": "local-benchmark"},
        "upload": {"mode": "durable", "workers": workers,
                   "max_pending_bytes": 1_000_000_000, "min_free_bytes": 0},
    }
    state = JobState(config)
    for number in range(clips):
        state.enqueue_upload(f"synthetic-{number:04d}", "synthetic", {
            "corpus": {}, "candidate": {"candidate_id": f"synthetic-{number:04d}"},
            "bundle": "synthetic-only"}, 1024)

    def fake_publish(corpus, candidate, bundle, store, **kwargs):
        time.sleep(wait_s)
        return {"commit_key": "local-benchmark/" + candidate["candidate_id"]}

    original = upload_queue.publish_candidate
    upload_queue.publish_candidate = fake_publish
    publisher = upload_queue.DurablePublisher(state, config, lambda _: object())
    started = time.monotonic()
    cpu_started = time.process_time()
    try:
        publisher.start()
        publisher.drain()
    finally:
        publisher.stop()
        upload_queue.publish_candidate = original
    wall_s = time.monotonic() - started
    if state.summary()["counts"] != {"published": clips}:
        raise AssertionError("Local benchmark did not drain all queued items")
    return {"workers": workers, "clips": clips, "simulated_wait_s": wait_s,
            "wall_s": round(wall_s, 3), "process_cpu_s": round(time.process_time()-cpu_started, 3),
            "clips_per_s": round(clips/wall_s, 2)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=int, default=80)
    parser.add_argument("--wait-s", type=float, default=0.05)
    args = parser.parse_args()
    if args.clips < 1 or args.wait_s < 0:
        parser.error("clips must be positive and wait-s non-negative")
    with tempfile.TemporaryDirectory(prefix="imu-upload-benchmark-") as directory:
        rows = [measure(Path(directory), workers, args.clips, args.wait_s)
                for workers in (1, 2, 4)]
    print(json.dumps({"kind": "local-simulated-io", "results": rows}, indent=2))


if __name__ == "__main__":
    main()
