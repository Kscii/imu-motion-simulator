"""Strict, human-editable configuration for one immutable production job."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from ..contracts.common import sha256_file
from ..publication import publication_prefix


SCHEMA = "imu_motion_simulator.production_job.v1"
FIELDS = {"schema", "library_root", "catalog", "checkout", "output", "policy",
          "sensor_workers", "sources", "clips_per_source", "publication"}
UPLOAD_FIELDS = {"mode", "workers", "max_pending_bytes", "min_free_bytes"}
PUBLICATION_FIELDS = {"target", "backend", "bucket", "project", "root", "run_id"}


def _path(base: Path, value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty path")
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else base / path).resolve())


def load_job_config(path: str | Path) -> dict:
    """Load YAML once; a job stores this normalized value and its digest."""
    path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (not isinstance(raw, dict) or not FIELDS <= set(raw)
            or set(raw) - FIELDS - {"upload"} or raw.get("schema") != SCHEMA):
        raise ValueError("Invalid production job configuration fields or schema")
    base = path.parent
    result = {"schema": SCHEMA}
    for name in ("library_root", "catalog", "checkout", "output", "policy"):
        result[name] = _path(base, raw[name], name)
    for name, kind in (("library_root", "dir"), ("catalog", "dir"),
                       ("checkout", "dir"), ("policy", "file")):
        candidate = Path(result[name])
        if not (candidate.is_dir() if kind == "dir" else candidate.is_file()):
            raise FileNotFoundError(candidate)
    result["policy_sha256"] = sha256_file(result["policy"])
    catalog_manifest = Path(result["catalog"]) / "catalog.json"
    if not catalog_manifest.is_file():
        raise FileNotFoundError(catalog_manifest)
    result["catalog_sha256"] = sha256_file(catalog_manifest)
    if Path(result["output"]).is_relative_to(Path(result["library_root"])):
        raise ValueError("Production output must not be inside the private source library")
    workers = raw["sensor_workers"]
    if type(workers) is not int or not 1 <= workers <= 32:
        raise ValueError("sensor_workers must be an integer in [1, 32]")
    result["sensor_workers"] = workers
    sources = raw["sources"]
    if (not isinstance(sources, list)
            or any(not isinstance(item, str) or not item for item in sources)
            or len(sources) != len(set(sources))):
        raise ValueError("sources must be a list of unique source names; [] means all")
    result["sources"] = sources
    limit = raw["clips_per_source"]
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("clips_per_source must be null or a positive integer")
    result["clips_per_source"] = limit
    publication = raw["publication"]
    if not isinstance(publication, dict) or set(publication) != PUBLICATION_FIELDS:
        raise ValueError("Invalid publication configuration")
    target = publication["target"]
    if target not in {"local", "dev", "prod"}:
        raise ValueError("Publication target must be local, dev or prod")
    backend = publication["backend"]
    if backend not in {"none", "local", "gcs"}:
        raise ValueError("Publication backend must be none, local or gcs")
    if (target == "local") != (backend == "none"):
        raise ValueError("Local target uses no publication backend")
    run_id = publication["run_id"]
    if target != "local":
        publication_prefix(target, run_id)
    elif run_id is not None:
        raise ValueError("Local target has no run ID")
    bucket = publication["bucket"]
    project = publication["project"]
    root = publication["root"]
    if backend == "gcs" and (not isinstance(bucket, str) or not bucket
                             or root is not None):
        raise ValueError("GCS publication requires bucket and no local root")
    if backend == "local" and (not isinstance(root, str) or not root
                               or bucket is not None):
        raise ValueError("Local publication requires root and no bucket")
    if backend == "none" and (bucket is not None or root is not None):
        raise ValueError("Local-only job cannot specify publication storage")
    if project is not None and not isinstance(project, str):
        raise ValueError("project must be a string or null")
    result["publication"] = {
        "target": target, "backend": backend, "bucket": bucket,
        "project": project,
        "root": _path(base, root, "publication.root") if root else None,
        "run_id": run_id,
    }
    if "upload" in raw:
        upload = raw["upload"]
        if (not isinstance(upload, dict) or set(upload) != UPLOAD_FIELDS
                or upload.get("mode") not in {"sync", "durable"}
                or type(upload.get("workers")) is not int
                or not 1 <= upload["workers"] <= 4
                or type(upload.get("max_pending_bytes")) is not int
                or upload["max_pending_bytes"] < 1
                or type(upload.get("min_free_bytes")) is not int
                or upload["min_free_bytes"] < 0):
            raise ValueError("Invalid durable upload configuration")
        if upload["mode"] == "durable" and target == "local":
            raise ValueError("Durable upload requires dev or prod publication")
        result["upload"] = upload
    return result


def config_digest(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()
