"""Read-only loopback production overview and explicit immutable core export."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
from threading import Lock, Thread

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse

from .config import load_job_config


DEFAULT_UNITS: tuple[str, ...] = ()


def _table(db: sqlite3.Connection, name: str) -> bool:
    return db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                      (name,)).fetchone() is not None


def _ledger(config: dict) -> dict:
    path = Path(config["output"]) / "production.sqlite3"
    result = {"output": config["output"],
              "upload_mode": config.get("upload", {}).get("mode", "sync"),
              "target": config["publication"]["target"],
              "ledger_exists": path.is_file()}
    if not path.is_file():
        return result
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=2) as db:
        db.row_factory = sqlite3.Row
        if _table(db, "job_runtime"):
            row = db.execute("SELECT value_json FROM job_runtime WHERE key='upload'").fetchone()
            if row:
                result["upload_mode"] = json.loads(row[0])["mode"]
                result["upload_runtime"] = json.loads(row[0])
        job = db.execute("SELECT status, control, created_at_utc, updated_at_utc "
                         "FROM job WHERE id=1").fetchone()
        if job is None:
            raise ValueError("Production ledger has no job row")
        result.update(dict(job))
        rows = db.execute("SELECT status, COUNT(*) AS n FROM clips GROUP BY status")
        result["counts"] = {row["status"]: row["n"] for row in rows}
        result["published_last_10m"] = db.execute(
            "SELECT COUNT(*) FROM clips WHERE status='published' AND updated_at_utc>=?",
            (datetime.fromtimestamp(datetime.now(timezone.utc).timestamp() - 600,
                                    timezone.utc).isoformat(),)).fetchone()[0]
        result["published_bytes"] = db.execute(
            "SELECT COALESCE(SUM(published_bytes),0) FROM clips").fetchone()[0]
        result["latest_published_at_utc"] = db.execute(
            "SELECT MAX(updated_at_utc) FROM clips WHERE status='published'").fetchone()[0]
        result["recent_errors"] = [dict(row) for row in db.execute(
            "SELECT clip_id, source_dataset, status, error, updated_at_utc "
            "FROM clips WHERE error IS NOT NULL ORDER BY updated_at_utc DESC LIMIT 12")]
        result["recent_events"] = [
            {**dict(row), "payload": json.loads(row["payload_json"])}
            for row in db.execute("SELECT seq, created_at_utc, kind, payload_json "
                                  "FROM events ORDER BY seq DESC LIMIT 12")]
        if _table(db, "upload_queue"):
            result["upload_queue"] = dict(db.execute(
                "SELECT COUNT(*) AS count, COALESCE(SUM(byte_length),0) AS bytes, "
                "COALESCE(SUM(attempts),0) AS retries, MIN(created_at_utc) AS oldest_at_utc, "
                "COALESCE(SUM(CASE WHEN state='blocked' THEN 1 ELSE 0 END),0) AS blocked "
                "FROM upload_queue").fetchone())
            result["upload_errors"] = [dict(row) for row in db.execute(
                "SELECT clip_id, attempts, last_error FROM upload_queue "
                "WHERE last_error IS NOT NULL ORDER BY attempts DESC LIMIT 5")]
        else:
            result["upload_queue"] = {"count": 0, "bytes": 0, "retries": 0,
                                      "oldest_at_utc": None}
            result["upload_errors"] = []
        if _table(db, "phase_metrics"):
            result["phases"] = [dict(row) for row in db.execute(
                "SELECT phase, COUNT(*) AS count, SUM(elapsed_s) AS total_s, "
                "AVG(elapsed_s) AS mean_s, MAX(elapsed_s) AS max_s "
                "FROM phase_metrics GROUP BY phase ORDER BY phase")]
        else:
            result["phases"] = []
        result["source_phases"] = [dict(row) for row in db.execute(
            "SELECT source_dataset, phase, elapsed_s, samples "
            "FROM source_phase_metrics ORDER BY source_dataset, phase")] \
            if _table(db, "source_phase_metrics") else []
        result["upload_transport"] = {}
        if _table(db, "upload_transport"):
            for row in db.execute("SELECT metrics_json FROM upload_transport"):
                for key, value in json.loads(row[0]).items():
                    result["upload_transport"][key] = (
                        result["upload_transport"].get(key, 0) + value)
    result["disk_free_bytes"] = shutil.disk_usage(config["output"]).free
    return result


def _unit(name: str) -> dict:
    try:
        process = subprocess.run(
            ["systemctl", "--user", "show", name, "--property=ActiveState,SubState",
             "--no-pager"], capture_output=True, text=True, timeout=2, check=False)
        values = dict(line.split("=", 1) for line in process.stdout.splitlines()
                      if "=" in line)
        return {"name": name, "active": values.get("ActiveState", "unknown"),
                "sub": values.get("SubState", "unknown"),
                "error": process.stderr.strip()[:300] if process.returncode else None}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"name": name, "active": "unknown", "sub": "unknown",
                "error": str(error)[:300]}


def _wifi() -> dict:
    try:
        process = subprocess.run(
            ["nmcli", "-t", "-f", "TYPE,STATE,CONNECTION", "device"],
            capture_output=True, text=True, timeout=2, check=False)
        rows = [line for line in process.stdout.splitlines() if line.startswith("wifi:")]
        return {"interfaces": rows, "error": process.stderr.strip()[:300]
                if process.returncode else None}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"interfaces": [], "error": str(error)[:300]}


def create_app(config_paths: list[Path], *, audit_report: Path | None = None,
               units: tuple[str, ...] = DEFAULT_UNITS) -> FastAPI:
    if not config_paths:
        raise ValueError("At least one production job is required")
    app = FastAPI(title="IMU Motion Production Dashboard", version="0.1.0")
    gate = Lock()
    export = {"status": "idle", "result": None, "error": None}

    def configs() -> list[dict]:
        return [load_job_config(path) for path in config_paths]

    @app.get("/api/v1/dashboard")
    def overview():
        jobs = []
        for path in config_paths:
            try:
                jobs.append({"config": str(path), **_ledger(load_job_config(path))})
            except Exception as error:
                jobs.append({"config": str(path), "error": str(error)[:500]})
        audit = None
        if audit_report is not None:
            try:
                audit = json.loads(audit_report.read_text(encoding="utf-8"))
            except FileNotFoundError:
                audit = {"phase": "not-created"}
            except (OSError, ValueError) as error:
                audit = {"phase": "unreadable", "error": str(error)[:500]}
        with gate:
            current_export = dict(export)
        return {"jobs": jobs, "services": [_unit(unit) for unit in units],
                "wifi": _wifi(), "audit": audit, "export": current_export,
                "updated_at_utc": datetime.now(timezone.utc).isoformat()}

    @app.get("/", response_class=HTMLResponse)
    def index():
        template = (Path(__file__).with_name("dashboard.html")
                    .read_text(encoding="utf-8"))
        initial = json.dumps(overview(), ensure_ascii=False, allow_nan=False)
        for character, escape in (("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026")):
            initial = initial.replace(character, escape)
        return template.replace("<!--INITIAL_STATE-->", initial)

    @app.post("/api/v1/export-core")
    def export_core(request: Request):
        # A loopback page can be reached by arbitrary sites; require an explicit
        # same-origin XHR header and reject cross-origin browser requests.
        if request.headers.get("x-imu-dashboard") != "export-core":
            raise HTTPException(status_code=403, detail="Missing dashboard action header")
        origin = request.headers.get("origin")
        if origin and origin != str(request.base_url).rstrip("/"):
            raise HTTPException(status_code=403, detail="Cross-origin action denied")
        with gate:
            if export["status"] == "running":
                raise HTTPException(status_code=409, detail="An export is already running")
        try:
            available = [item for item in configs()
                         if (Path(item["output"]) / "production.sqlite3").is_file()
                         and _ledger(item).get("counts", {}).get("published", 0) > 0]
            if not available:
                raise ValueError("No published production ledger is available")
            if available[0]["publication"]["target"] == "local":
                raise ValueError("Core export requires dev or prod publication")
        except Exception as error:
            raise HTTPException(status_code=409, detail=str(error)[:500]) from error

        def run() -> None:
            from ..provisional import export_provisional
            try:
                result = export_provisional(available[0], additional_configs=available[1:])
                with gate:
                    export.update(status="complete", result=result, error=None)
            except Exception as error:
                with gate:
                    export.update(status="failed", result=None, error=str(error)[:1000])

        with gate:
            if export["status"] == "running":
                raise HTTPException(status_code=409, detail="An export is already running")
            export.update(status="running", result=None, error=None)
        Thread(target=run, name="manual-core-export", daemon=True).start()
        return {"status": "running", "coverage": "partial-until-audited"}

    return app


def serve(config_paths: list[Path], *, audit_report: Path | None = None,
          port: int = 8891, units: tuple[str, ...] = DEFAULT_UNITS) -> None:
    import uvicorn
    uvicorn.run(create_app(config_paths, audit_report=audit_report, units=units),
                host="127.0.0.1", port=port)
