"""Durable job ledger shared by the foreground CLI and local control service."""
from __future__ import annotations

from datetime import datetime, timezone
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import statistics
import time

from .config import config_digest
from .upload_runtime import SCHEMA as UPLOAD_RUNTIME_SCHEMA


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobState:
    def __init__(self, config: dict):
        self.config = config
        self.root = Path(config["output"]).resolve()
        self.path = self.root / "production.sqlite3"
        if self.root.exists() and not self.path.is_file():
            raise FileExistsError("Production output exists without a resumable job ledger")
        self.root.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS job (
                    id INTEGER PRIMARY KEY CHECK (id=1),
                    config_sha256 TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    control TEXT NOT NULL DEFAULT 'run',
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS clips (
                    clip_id TEXT PRIMARY KEY,
                    source_dataset TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT,
                    commit_key TEXT,
                    published_bytes INTEGER NOT NULL DEFAULT 0,
                    updated_at_utc TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS production_clip_status_idx ON clips(status);
                CREATE TABLE IF NOT EXISTS source_inputs (
                    source_dataset TEXT PRIMARY KEY,
                    identity_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at_utc TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS upload_queue (
                    clip_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    byte_length INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    next_attempt_at REAL NOT NULL DEFAULT 0,
                    last_error TEXT,
                    created_at_utc TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS upload_queue_due_idx
                    ON upload_queue(state, next_attempt_at);
                CREATE TABLE IF NOT EXISTS phase_metrics (
                    clip_id TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    elapsed_s REAL NOT NULL,
                    PRIMARY KEY (clip_id, phase)
                );
                CREATE TABLE IF NOT EXISTS source_phase_metrics (
                    source_dataset TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    elapsed_s REAL NOT NULL,
                    samples INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY (source_dataset, phase)
                );
                CREATE TABLE IF NOT EXISTS job_runtime (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS upload_transport (
                    worker_id TEXT PRIMARY KEY,
                    metrics_json TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL
                );
            """)
            identity = db.execute("SELECT config_sha256 FROM job WHERE id=1").fetchone()
            digest = config_digest(config)
            if identity is not None and identity[0] != digest:
                raise ValueError("Existing job has a different frozen configuration")
            if identity is None:
                now = _utc()
                db.execute("INSERT INTO job VALUES (1, ?, ?, 'queued', 'run', ?, ?)",
                           (digest, json.dumps(config, ensure_ascii=False,
                                               sort_keys=True), now, now))
                self._event(db, "created", {"config_sha256": digest})

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _event(db, kind: str, payload: dict) -> None:
        db.execute("INSERT INTO events (created_at_utc, kind, payload_json) VALUES (?, ?, ?)",
                   (_utc(), kind, json.dumps(payload, ensure_ascii=False,
                                             sort_keys=True, allow_nan=False)))

    def emit(self, kind: str, payload: dict) -> None:
        with self._connect() as db:
            self._event(db, kind, payload)

    def set_status(self, status: str) -> None:
        if status not in {"queued", "running", "paused", "cancelled", "complete", "partial", "failed"}:
            raise ValueError("Invalid job status")
        with self._connect() as db:
            db.execute("UPDATE job SET status=?, updated_at_utc=? WHERE id=1",
                       (status, _utc()))
            self._event(db, "job-status", {"status": status})

    def control(self) -> str:
        with self._connect() as db:
            return db.execute("SELECT control FROM job WHERE id=1").fetchone()[0]

    def upload_runtime(self) -> dict | None:
        with self._connect() as db:
            row = db.execute("SELECT value_json FROM job_runtime WHERE key='upload'").fetchone()
        return json.loads(row[0]) if row else None

    def bind_upload_runtime(self, profile: dict) -> None:
        """Bind once while the job is stopped; a live migration uses the job lock."""
        if (profile.get("schema") != UPLOAD_RUNTIME_SCHEMA
                or profile.get("mode") != "durable"
                or type(profile.get("workers")) is not int
                or not 1 <= profile["workers"] <= 8
                or type(profile.get("max_pending_bytes")) is not int
                or profile["max_pending_bytes"] < 1
                or type(profile.get("min_free_bytes")) is not int
                or profile["min_free_bytes"] < 0
                or set(profile) != {"schema", "mode", "workers",
                                    "max_pending_bytes", "min_free_bytes"}):
            raise ValueError("Invalid operational upload profile")
        if self.config["publication"]["target"] == "local":
            raise ValueError("Durable upload requires cloud publication")
        encoded = json.dumps(profile, sort_keys=True, separators=(",", ":"))
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            status = db.execute("SELECT status FROM job WHERE id=1").fetchone()[0]
            if status not in {"queued", "paused", "partial", "failed"}:
                raise ValueError("Upload profile can only bind to a stopped job")
            prior = db.execute("SELECT value_json FROM job_runtime WHERE key='upload'").fetchone()
            if prior is not None:
                if prior[0] != encoded:
                    raise ValueError("Operational upload profile is already bound differently")
                return
            db.execute("INSERT INTO job_runtime VALUES ('upload', ?, ?)", (encoded, _utc()))
            self._event(db, "upload-runtime-bound", profile)

    def block_upload(self, clip_id: str, error: str) -> None:
        """Keep the original intent for an explicit repair; never silently retry a conflict."""
        with self._connect() as db:
            db.execute("UPDATE upload_queue SET state='blocked', attempts=attempts+1, "
                       "last_error=? WHERE clip_id=?", (error[:1000], clip_id))
            db.execute("UPDATE clips SET error=?, updated_at_utc=? WHERE clip_id=?",
                       (error[:1000], _utc(), clip_id))
            self._event(db, "upload-blocked", {"clip_id": clip_id,
                                                "error": error[:1000]})

    def unblock_uploads(self) -> int:
        with self._connect() as db:
            cursor = db.execute("UPDATE upload_queue SET state='queued', "
                                "next_attempt_at=0 WHERE state='blocked'")
            if cursor.rowcount:
                self._event(db, "upload-unblocked", {"count": cursor.rowcount})
            return cursor.rowcount

    def request(self, action: str) -> None:
        if action not in {"run", "pause", "cancel"}:
            raise ValueError("Invalid job control")
        with self._connect() as db:
            db.execute("UPDATE job SET control=?, updated_at_utc=? WHERE id=1",
                       (action, _utc()))
            self._event(db, "control-request", {"action": action})

    def clip_status(self, clip_id: str) -> str | None:
        with self._connect() as db:
            row = db.execute("SELECT status FROM clips WHERE clip_id=?",
                             (clip_id,)).fetchone()
            return row[0] if row else None

    def bind_source(self, source: str, identity: dict) -> None:
        """Prevent a resumed job from mixing different archive/model bytes."""
        payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        with self._connect() as db:
            row = db.execute(
                "SELECT identity_json FROM source_inputs WHERE source_dataset=?",
                (source,)).fetchone()
            if row is not None:
                if row[0] != payload:
                    raise ValueError("Frozen source inputs changed: " + source)
                return
            db.execute("INSERT INTO source_inputs VALUES (?, ?)", (source, payload))
            self._event(db, "source-bound", {"source_dataset": source,
                                             **identity})

    def record_clip(self, clip_id: str, source: str, status: str, *,
                    error: str | None = None, commit_key: str | None = None,
                    published_bytes: int = 0) -> None:
        if status not in {"published", "ready-local", "excluded", "failed"}:
            raise ValueError("Invalid clip status")
        with self._connect() as db:
            previous = db.execute("SELECT status FROM clips WHERE clip_id=?",
                                  (clip_id,)).fetchone()
            if previous and previous[0] in {"published", "ready-local", "excluded"}:
                if previous[0] == status:
                    return
                raise ValueError("Completed clip cannot change status")
            db.execute("""INSERT INTO clips
                (clip_id, source_dataset, status, error, commit_key,
                 published_bytes, updated_at_utc) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(clip_id) DO UPDATE SET
                status=excluded.status, error=excluded.error,
                commit_key=excluded.commit_key,
                published_bytes=excluded.published_bytes,
                updated_at_utc=excluded.updated_at_utc""",
                (clip_id, source, status, error[:1000] if error else None,
                 commit_key, published_bytes, _utc()))
            self._event(db, "clip-status", {
                "clip_id": clip_id, "source_dataset": source, "status": status,
                "error": error[:1000] if error else None,
                "commit_key": commit_key, "published_bytes": published_bytes,
            })

    def enqueue_upload(self, clip_id: str, source: str, payload: dict,
                       byte_length: int) -> None:
        """Commit the local candidate and its upload intent in one transaction."""
        if byte_length < 0:
            raise ValueError("Invalid pending upload size")
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                             allow_nan=False)
        with self._connect() as db:
            previous = db.execute("SELECT status FROM clips WHERE clip_id=?",
                                  (clip_id,)).fetchone()
            if previous and previous[0] in {"published", "ready-local", "excluded"}:
                raise ValueError("Completed clip cannot be enqueued")
            queued = db.execute("SELECT payload_json FROM upload_queue WHERE clip_id=?",
                                (clip_id,)).fetchone()
            if queued and queued[0] != encoded:
                raise ValueError("Queued upload identity changed")
            if not queued:
                db.execute("""INSERT INTO upload_queue VALUES (?, ?, ?, 'queued', 0, 0, NULL, ?)""",
                           (clip_id, encoded, byte_length, _utc()))
            db.execute("""INSERT INTO clips
                (clip_id, source_dataset, status, error, commit_key,
                 published_bytes, updated_at_utc)
                VALUES (?, ?, 'pending-publish', NULL, NULL, 0, ?)
                ON CONFLICT(clip_id) DO UPDATE SET status='pending-publish',
                error=NULL, updated_at_utc=excluded.updated_at_utc""",
                (clip_id, source, _utc()))
            self._event(db, "upload-queued", {"clip_id": clip_id,
                                                "byte_length": byte_length})

    def reset_upload_claims(self) -> None:
        with self._connect() as db:
            db.execute("UPDATE upload_queue SET state='queued' WHERE state='uploading'")

    def claim_upload(self) -> tuple[str, dict, int, float] | None:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("""SELECT clip_id, payload_json, byte_length,
                created_at_utc
                FROM upload_queue WHERE state='queued' AND next_attempt_at<=?
                ORDER BY next_attempt_at, created_at_utc LIMIT 1""",
                (time.time(),)).fetchone()
            if row is None:
                return None
            db.execute("UPDATE upload_queue SET state='uploading' WHERE clip_id=?",
                       (row["clip_id"],))
            queue_age_s = max(0.0, time.time() -
                              datetime.fromisoformat(row["created_at_utc"]).timestamp())
            return (row["clip_id"], json.loads(row["payload_json"]),
                    row["byte_length"], queue_age_s)

    def finish_upload(self, clip_id: str, commit_key: str,
                      byte_length: int, elapsed_s: float) -> None:
        with self._connect() as db:
            row = db.execute("SELECT source_dataset FROM clips WHERE clip_id=?",
                             (clip_id,)).fetchone()
            if row is None or db.execute("SELECT 1 FROM upload_queue WHERE clip_id=?",
                                         (clip_id,)).fetchone() is None:
                raise ValueError("Upload is not queued")
            db.execute("UPDATE clips SET status='published', error=NULL, commit_key=?, "
                       "published_bytes=?, updated_at_utc=? WHERE clip_id=?",
                       (commit_key, byte_length, _utc(), clip_id))
            db.execute("DELETE FROM upload_queue WHERE clip_id=?", (clip_id,))
            db.execute("""INSERT INTO phase_metrics VALUES (?, 'upload', ?)
                ON CONFLICT(clip_id, phase) DO UPDATE SET elapsed_s=excluded.elapsed_s""",
                (clip_id, elapsed_s))
            self._event(db, "clip-status", {"clip_id": clip_id,
                "source_dataset": row["source_dataset"], "status": "published",
                "error": None, "commit_key": commit_key,
                "published_bytes": byte_length})

    def retry_upload(self, clip_id: str, error: str, delay_s: float | None) -> None:
        with self._connect() as db:
            if delay_s is None:
                db.execute("DELETE FROM upload_queue WHERE clip_id=?", (clip_id,))
                db.execute("UPDATE clips SET status='failed', error=?, updated_at_utc=? "
                           "WHERE clip_id=?", (error[:1000], _utc(), clip_id))
                self._event(db, "upload-failed", {"clip_id": clip_id,
                                                    "error": error[:1000]})
            else:
                db.execute("""UPDATE upload_queue SET state='queued',
                    attempts=attempts+1, next_attempt_at=?, last_error=?
                    WHERE clip_id=?""",
                    (time.time() + delay_s, error[:1000], clip_id))
                self._event(db, "upload-retry", {"clip_id": clip_id,
                    "delay_s": delay_s, "error": error[:1000]})

    def upload_summary(self) -> dict:
        with self._connect() as db:
            row = db.execute("""SELECT COUNT(*) AS count,
                COALESCE(SUM(byte_length), 0) AS bytes,
                COALESCE(SUM(attempts), 0) AS attempts,
                MIN(created_at_utc) AS oldest_at_utc,
                COALESCE(SUM(CASE WHEN state='blocked' THEN 1 ELSE 0 END),0) AS blocked
                FROM upload_queue""").fetchone()
        return dict(row)

    def upload_attempts(self, clip_id: str) -> int:
        with self._connect() as db:
            row = db.execute("SELECT attempts FROM upload_queue WHERE clip_id=?",
                             (clip_id,)).fetchone()
        return row[0] if row else 0

    def record_phase(self, clip_id: str, phase: str, elapsed_s: float) -> None:
        if phase not in {"extract", "prepare", "sensor", "sensor-compute",
                         "sensor-queue-and-return", "sensor-dispatch-wait",
                         "sensor-return-and-collection", "bundle", "upload",
                         "upload-queue-age-to-claim"} or elapsed_s < 0:
            raise ValueError("Invalid production phase metric")
        with self._connect() as db:
            db.execute("""INSERT INTO phase_metrics VALUES (?, ?, ?)
                ON CONFLICT(clip_id, phase) DO UPDATE SET elapsed_s=excluded.elapsed_s""",
                (clip_id, phase, elapsed_s))

    def record_source_phase(self, source: str, phase: str, elapsed_s: float) -> None:
        if phase not in {"archive-hash", "model-hash", "archive-stream-next"} \
                or elapsed_s < 0:
            raise ValueError("Invalid production source phase metric")
        with self._connect() as db:
            db.execute("""INSERT INTO source_phase_metrics VALUES (?, ?, ?, 1)
                ON CONFLICT(source_dataset, phase) DO UPDATE SET
                elapsed_s=elapsed_s+excluded.elapsed_s,
                samples=samples+1""", (source, phase, elapsed_s))

    def source_phase_summary(self) -> list[dict]:
        with self._connect() as db:
            return [dict(row) for row in db.execute(
                "SELECT source_dataset, phase, elapsed_s, samples "
                "FROM source_phase_metrics ORDER BY source_dataset, phase")]

    def record_upload_metrics(self, worker_id: str, metrics: dict) -> None:
        with self._connect() as db:
            db.execute("""INSERT INTO upload_transport VALUES (?, ?, ?)
                ON CONFLICT(worker_id) DO UPDATE SET
                    metrics_json=excluded.metrics_json,
                    updated_at_utc=excluded.updated_at_utc""",
                (worker_id, json.dumps(metrics, sort_keys=True), _utc()))

    def recent_errors(self, limit: int = 50) -> list[dict]:
        with self._connect() as db:
            rows = db.execute("""SELECT clip_id, source_dataset, status, error,
                updated_at_utc FROM clips WHERE error IS NOT NULL
                ORDER BY updated_at_utc DESC LIMIT ?""", (max(1, min(limit, 200)),))
            return [dict(row) for row in rows]

    def phase_summary(self) -> list[dict]:
        with self._connect() as db:
            rows = db.execute("""SELECT phase, elapsed_s FROM phase_metrics
                ORDER BY phase, elapsed_s""").fetchall()
        samples: dict[str, list[float]] = {}
        for row in rows:
            samples.setdefault(row["phase"], []).append(row["elapsed_s"])
        result = []
        for phase, values in samples.items():
            def percentile(ratio: float) -> float:
                position = (len(values) - 1) * ratio
                low = int(position)
                return values[low] + (values[min(low + 1, len(values) - 1)]
                                      - values[low]) * (position - low)
            result.append({"phase": phase, "count": len(values),
                           "total_s": sum(values), "mean_s": statistics.fmean(values),
                           "p50_s": percentile(0.5), "p95_s": percentile(0.95),
                           "max_s": values[-1]})
        return result

    def summary(self) -> dict:
        with self._connect() as db:
            job = db.execute("SELECT * FROM job WHERE id=1").fetchone()
            rows = db.execute("SELECT status, COUNT(*) AS count, "
                              "SUM(published_bytes) AS bytes FROM clips GROUP BY status")
            counts = {row["status"]: row["count"] for row in rows}
            published_bytes = db.execute(
                "SELECT COALESCE(SUM(published_bytes),0) FROM clips"
            ).fetchone()[0]
            latest = db.execute("SELECT COALESCE(MAX(seq),0) FROM events").fetchone()[0]
            return {"status": job["status"], "control": job["control"],
                    "config_sha256": job["config_sha256"],
                    "target": self.config["publication"]["target"],
                    "created_at_utc": job["created_at_utc"],
                    "updated_at_utc": job["updated_at_utc"],
                    "counts": counts, "published_bytes": published_bytes,
                    "last_event_seq": latest,
                    "upload_runtime": self.upload_runtime(),
                    "upload_queue": self.upload_summary()}

    def source_summary(self, source: str, planned: int) -> dict:
        with self._connect() as db:
            rows = db.execute(
                "SELECT status, COUNT(*) AS count FROM clips "
                "WHERE source_dataset=? GROUP BY status", (source,)).fetchall()
        counts = {row["status"]: row["count"] for row in rows}
        processed = sum(counts.values())
        return {"planned": planned, "processed": processed,
                "published": counts.get("published", 0),
                "ready_local": counts.get("ready-local", 0),
                "pending_publish": counts.get("pending-publish", 0),
                "excluded": counts.get("excluded", 0),
                "failed": counts.get("failed", 0),
                "status": "complete" if processed == planned
                else "incomplete"}

    def events_since(self, seq: int, limit: int = 100) -> list[dict]:
        with self._connect() as db:
            rows = db.execute("""SELECT seq, created_at_utc, kind, payload_json
                FROM events WHERE seq>? ORDER BY seq LIMIT ?""", (seq, limit))
            return [{"seq": row["seq"], "created_at_utc": row["created_at_utc"],
                     "kind": row["kind"], "payload": json.loads(row["payload_json"])}
                    for row in rows]
