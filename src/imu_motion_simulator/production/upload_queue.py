"""Durable local upload intents, drained independently of motion/IMU computation."""
from __future__ import annotations

from pathlib import Path
import random
import shutil
from threading import current_thread, Event, Thread
import time
from uuid import uuid4

from ..publication import publish_candidate
from .state import JobState


def transient_upload_error(error: Exception) -> bool:
    """Only retry connectivity/server failures; invalid local data stays failed."""
    if blocking_upload_error(error):
        return False
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    code = getattr(error, "code", None)
    if isinstance(code, int) and (code == 429 or 500 <= code <= 599):
        return True
    name = type(error).__name__.lower()
    message = str(error).lower()
    return any(value in name for value in (
        "timeout", "connection", "retryerror", "serviceunavailable",
        "toomanyrequests", "internalservererror", "gatewaytimeout",
        "transporterror",
    )) or any(value in message for value in (
        "nameresolutionerror", "temporary failure in name resolution",
        "connection reset", "connection refused", "deadline exceeded",
        "503 service unavailable", "502 bad gateway", "429 too many requests",
    ))


def blocking_upload_error(error: Exception) -> bool:
    """Auth and immutable-object conflicts require operator repair, not retries."""
    code = getattr(error, "code", None)
    name = type(error).__name__.lower()
    message = str(error).lower()
    return (code in {401, 403}
            or name in {"unauthorized", "forbidden", "permissiondenied"}
            or any(value in message for value in (
                "401 unauthorized", "403 forbidden", "permission denied",
                "remote object could not be verified",
                "remote commit could not be verified")))


class DurablePublisher:
    def __init__(self, state: JobState, config: dict, store_factory):
        self.state = state
        self.config = config
        self.upload = config["upload"]
        self.store_factory = store_factory
        self.stop_event = Event()
        self.threads: list[Thread] = []
        self.failure: Exception | None = None
        self.on_published = None

    def start(self) -> None:
        self.state.reset_upload_claims()
        for index in range(self.upload["workers"]):
            thread = Thread(target=self._worker, name=f"upload-{index}", daemon=True)
            thread.start()
            self.threads.append(thread)

    def _worker(self) -> None:
        store = None
        worker_id = uuid4().hex + ":" + current_thread().name
        totals = {}
        activity = {"worker_busy_s": 0.0, "worker_idle_s": 0.0,
                    "worker_claims": 0}

        def snapshot(close: bool = False):
            nonlocal totals
            current = (store.metrics_snapshot()
                       if store is not None and hasattr(store, "metrics_snapshot")
                       else {})
            merged = dict(totals)
            for key, value in current.items():
                merged[key] = totals.get(key, 0) + value
            self.state.record_upload_metrics(worker_id, {**merged, **activity})
            if close:
                totals = merged

        try:
            publication = self.config["publication"]
            while not self.stop_event.is_set():
                if self.state.control() != "run":
                    idle_started = time.monotonic()
                    self.stop_event.wait(0.5)
                    activity["worker_idle_s"] += time.monotonic() - idle_started
                    continue
                task = self.state.claim_upload()
                if task is None:
                    idle_started = time.monotonic()
                    self.stop_event.wait(0.5)
                    activity["worker_idle_s"] += time.monotonic() - idle_started
                    continue
                clip_id, payload, byte_length, queue_age_s = task
                started = time.monotonic()
                activity["worker_claims"] += 1
                self.state.record_phase(clip_id, "upload-queue-age-to-claim",
                                        queue_age_s)
                published = False
                try:
                    if store is None:
                        store = self.store_factory(self.config)
                    outcome = publish_candidate(
                        payload["corpus"], payload["candidate"],
                        Path(payload["bundle"]), store,
                        target=publication["target"], run_id=publication["run_id"],
                        outbox=self.state.root / "outbox")
                    self.state.finish_upload(clip_id, outcome["commit_key"],
                                             byte_length, time.monotonic() - started)
                    published = True
                except Exception as error:
                    snapshot(close=True)
                    store = None  # discard a potentially broken HTTP session
                    if blocking_upload_error(error):
                        self.state.block_upload(clip_id, str(error))
                        self.failure = RuntimeError(
                            f"Upload blocked for {clip_id}: {error}")
                        self.stop_event.set()
                    else:
                        attempts = self.state.upload_attempts(clip_id)
                        delay = (min(300.0, 5.0 * 2 ** min(attempts, 6))
                                 * random.uniform(0.8, 1.2)) if transient_upload_error(error) else None
                        self.state.retry_upload(clip_id, str(error), delay)
                activity["worker_busy_s"] += time.monotonic() - started
                snapshot()
                if published and self.on_published is not None:
                    try:
                        self.on_published()
                    except Exception as error:
                        self.state.emit("progress-callback-error", {
                            "clip_id": clip_id, "error": str(error)[:1000]})
        except Exception as error:
            self.failure = error
            self.state.emit("upload-worker-fatal", {"error": str(error)[:1000]})
            self.stop_event.set()
        finally:
            snapshot()

    def _raise_if_failed(self) -> None:
        if self.failure is not None:
            raise RuntimeError("Upload worker failed") from self.failure

    def wait_for_capacity(self) -> bool:
        """Let the uploader drain while the local backlog or disk is constrained."""
        constrained = False
        while self.state.control() == "run" and not self.stop_event.is_set():
            self._raise_if_failed()
            queued = self.state.upload_summary()
            free = shutil.disk_usage(self.state.root).free
            if (queued["bytes"] < self.upload["max_pending_bytes"]
                    and free >= self.upload["min_free_bytes"]):
                if constrained:
                    self.state.emit("upload-capacity-restored", queued)
                return True
            if not constrained:
                self.state.emit("upload-backpressure", {
                    **queued, "free_bytes": free,
                    "max_pending_bytes": self.upload["max_pending_bytes"],
                    "min_free_bytes": self.upload["min_free_bytes"],
                })
                constrained = True
            self.stop_event.wait(1.0)
        self._raise_if_failed()
        return False

    def drain(self) -> None:
        while self.state.control() == "run" and self.state.upload_summary()["count"]:
            self._raise_if_failed()
            self.stop_event.wait(0.5)
        self._raise_if_failed()

    def stop(self) -> None:
        self.stop_event.set()
        for thread in self.threads:
            thread.join(timeout=2.0)
