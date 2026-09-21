"""Loopback-only HTTP commands and resumable server-sent job events."""
from __future__ import annotations

import asyncio
import json
from threading import Lock, Thread

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from .config import config_digest
from .runner import run_job
from .state import JobState


def create_app(config: dict):
    state = JobState(config)
    job_id = config_digest(config)[:16]
    app = FastAPI(title="IMU Motion Production Control", version="0.1.0")
    gate = Lock()
    active: Thread | None = None

    def require_job(value: str) -> None:
        if value != job_id:
            raise HTTPException(status_code=404, detail="Unknown job")

    def launch() -> None:
        nonlocal active
        with gate:
            if active is not None and active.is_alive():
                raise HTTPException(status_code=409, detail="Job is already running")
            active = Thread(target=run_job, args=(config,), daemon=True,
                            name=f"imu-production-{job_id}")
            active.start()

    @app.get("/api/v1/health")
    def health():
        return {"ok": True, "job_id": job_id}

    @app.get("/api/v1/jobs")
    def jobs():
        return {"jobs": [{"job_id": job_id, **state.summary()}]}

    @app.post("/api/v1/jobs")
    def start():
        if state.summary()["status"] not in {"queued", "paused", "partial", "failed"}:
            raise HTTPException(status_code=409, detail="Job cannot be started in this state")
        launch()
        return {"job_id": job_id, "status": "starting"}

    @app.get("/api/v1/jobs/{requested_id}")
    def detail(requested_id: str):
        require_job(requested_id)
        return {"job_id": job_id, **state.summary()}

    @app.post("/api/v1/jobs/{requested_id}/{action}")
    def action(requested_id: str, action: str):
        require_job(requested_id)
        status = state.summary()["status"]
        if action == "pause" and status == "running":
            state.request("pause")
            return {"job_id": job_id, "control": "pause"}
        if action == "cancel" and status in {"queued", "running", "paused", "partial", "failed"}:
            state.request("cancel")
            if status != "running":
                state.set_status("cancelled")
            return {"job_id": job_id, "control": "cancel"}
        if action == "resume" and status == "paused":
            launch()
            return {"job_id": job_id, "status": "resuming"}
        if action == "retry" and status in {"partial", "failed"}:
            launch()
            return {"job_id": job_id, "status": "retrying"}
        raise HTTPException(status_code=409, detail="Action is not valid in this job state")

    @app.get("/api/v1/jobs/{requested_id}/events")
    async def events(requested_id: str, request: Request, after: int = 0):
        require_job(requested_id)
        header = request.headers.get("last-event-id")
        if header:
            try:
                after = max(after, int(header))
            except ValueError as error:
                raise HTTPException(status_code=422, detail="Invalid Last-Event-ID") from error
        if after < 0:
            raise HTTPException(status_code=422, detail="Invalid event cursor")

        async def stream():
            cursor = after
            while not await request.is_disconnected():
                batch = state.events_since(cursor)
                if batch:
                    for event in batch:
                        cursor = event["seq"]
                        yield (f"id: {cursor}\nevent: {event['kind']}\ndata: "
                               + json.dumps(event, ensure_ascii=False) + "\n\n")
                else:
                    yield ": heartbeat\n\n"
                    await asyncio.sleep(1)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    return app


def serve(config: dict, *, host: str = "127.0.0.1", port: int = 8890) -> None:
    if host not in {"127.0.0.1", "::1"}:
        raise ValueError("Control API is loopback-only until an authenticated proxy is deployed")
    import uvicorn
    uvicorn.run(create_app(config), host=host, port=port)
