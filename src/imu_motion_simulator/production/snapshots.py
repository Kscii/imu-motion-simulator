"""Public, one-shot snapshot operations for a frozen production job."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
from pathlib import Path

from ..pipeline.plan import resolve_inside
from ..snapshot_worker import (retry_failed_snapshot as retry_frozen_snapshot,
                               run_pending_once)
from .runner import _store
from .state import JobState


def _inputs(config: dict):
    publication = config["publication"]
    if publication["target"] == "local":
        raise ValueError("Snapshot worker requires dev or prod publication")
    output = Path(config["output"])
    if not (output / "production.sqlite3").is_file():
        raise FileNotFoundError("Start the production job before its snapshot worker")
    # A snapshot must use the exact frozen job that published its candidates.
    JobState(config)
    catalog = json.loads((Path(config["catalog"]) / "catalog.json").read_text())
    inputs = catalog["inputs"]
    model = resolve_inside(config["library_root"], inputs["smplh_archive"])
    dmpl = resolve_inside(config["library_root"], inputs["dmpl_archive"])
    layout = resolve_inside(config["checkout"], inputs["layout"])
    for required in (model, layout):
        if not required.is_file():
            raise FileNotFoundError(required)
    return publication, output / "snapshots", model, dmpl if dmpl.is_file() else None, layout


@contextmanager
def _single_worker(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".single-worker.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Snapshot worker is already running") from error
        yield


def run_pending_snapshots(config: dict) -> list[dict]:
    """Process pending requests once; failures remain available for explicit retry."""
    publication, output, model, dmpl, layout = _inputs(config)
    with _single_worker(output):
        return run_pending_once(
            _store(config), publication["run_id"], output,
            model_archive=model, dmpl_archive=dmpl, layout=layout,
            target=publication["target"])


def retry_snapshot(config: dict, snapshot_id: str) -> dict:
    """Retry one failed immutable request without changing its frozen inputs."""
    publication, output, model, dmpl, layout = _inputs(config)
    with _single_worker(output):
        return retry_frozen_snapshot(
            _store(config), publication["run_id"], snapshot_id,
            output / snapshot_id, model_archive=model, dmpl_archive=dmpl,
            layout=layout, target=publication["target"])
