"""Build a replay-only HDF5 3.3 snapshot from frozen platform decisions."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .contracts.common import sha256_file
from .contracts.delivery import validate_delivery
from .delivery_kinematic import export_kinematic
from .publication import ID, _json_bytes, publication_prefix
from .review.bundle import validate_bundle
from .snapshot_pack import StreamingShard, estimate_clip_delivery

SNAPSHOT_ID = re.compile(r"^synthetic-[0-9a-f]{32}$")
MAX_SHARD_BYTES = 4 * 1024 ** 3
CHECKPOINT_SCHEMA = "imu_motion_simulator.snapshot_shard_checkpoint.v1"


def _verified_json(store, key, digest):
    value = store.read_json(key)
    if hashlib.sha256(_json_bytes(value)).hexdigest() != digest:
        raise ValueError("Frozen JSON hash differs: " + key)
    return value


def _candidate(entry):
    return {"candidate_id": entry["candidate_id"],
            "version_id": entry["version_id"],
            "review_sha256": entry["review_sha256"]}


def _load_checkpoints(store, output, prefix, snapshot_id, request_sha256, entries):
    """Verify local and published shards before skipping frozen entries."""
    checkpoints = output / "checkpoints"
    shards, processed = [], 0
    for ordinal, path in enumerate(sorted(checkpoints.glob("shard-*.json")), 1):
        if path.name != f"shard-{ordinal:04d}.json":
            raise ValueError("Snapshot shard checkpoints are not contiguous")
        checkpoint = json.loads(path.read_text())
        shard = checkpoint.get("shard")
        start, stop = checkpoint.get("start_ordinal"), checkpoint.get("stop_ordinal")
        shard_path = output / f"shard-{ordinal:04d}.h5"
        shard_key = f"{prefix}/snapshots/{snapshot_id}/shards/{shard_path.name}"
        if checkpoint.get("schema") != CHECKPOINT_SCHEMA \
                or checkpoint.get("request_sha256") != request_sha256 \
                or start != processed + 1 or not isinstance(stop, int) \
                or stop < start or stop > len(entries) \
                or not isinstance(shard, dict) \
                or shard.get("object_key") != shard_key \
                or shard.get("candidates") != [
                    _candidate(entry) for entry in entries[start - 1:stop]]:
            raise ValueError("Snapshot shard checkpoint differs from frozen request")
        if not shard_path.exists():
            store.download_file(shard_key, shard_path, shard["sha256"],
                                shard["byte_length"])
        if shard_path.stat().st_size != shard["byte_length"] \
                or sha256_file(shard_path) != shard["sha256"]:
            raise ValueError("Snapshot shard checkpoint file differs")
        validate_delivery(shard_path)
        store.put_file(shard_path, shard_key, shard["sha256"])
        for index in range(start, stop + 1):
            (output / "clip-deliveries" / f"clip-{index:04d}.h5").unlink(missing_ok=True)
            shutil.rmtree(output / "clips" /
                          f"{index:04d}-{entries[index - 1]['candidate_id']}",
                          ignore_errors=True)
        shards.append(shard)
        processed = stop
    return shards, processed


def _write_checkpoint(output, ordinal, request_sha256, start, stop, shard):
    directory = output / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"shard-{ordinal:04d}.json"
    payload = {"schema": CHECKPOINT_SCHEMA,
               "request_sha256": request_sha256,
               "start_ordinal": start, "stop_ordinal": stop, "shard": shard}
    temporary = target.with_name(target.name + "." + uuid4().hex + ".partial")
    try:
        temporary.write_bytes(_json_bytes(payload))
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def _run_snapshot(store, run_id, snapshot_id, output, *, model_archive,
                  dmpl_archive, layout, target="dev"):
    if not SNAPSHOT_ID.fullmatch(snapshot_id):
        raise ValueError("Invalid snapshot ID")
    prefix = publication_prefix(target, run_id)
    request_key = f"{prefix}/snapshots/requests/{snapshot_id}.json"
    result_key = f"{prefix}/snapshots/results/{snapshot_id}.json"
    try:
        return store.read_json(result_key)
    except FileNotFoundError:
        pass
    intent = store.read_json(request_key)
    entries = intent.get("entries")
    if intent.get("schema") != "imu_motion_simulator.snapshot_intent.v1" \
            or intent.get("snapshot_id") != snapshot_id \
            or not isinstance(entries, list) or not entries:
        raise ValueError("Invalid frozen snapshot request")
    identities = {(entry["candidate_id"], entry["version_id"]) for entry in entries}
    if len(identities) != len(entries):
        raise ValueError("Duplicate candidate in snapshot request")
    if any(not isinstance(entry["candidate_id"], str)
           or not ID.fullmatch(entry["candidate_id"])
           or not isinstance(entry["version_id"], str)
           or not re.fullmatch(r"[0-9a-f]{64}", entry["version_id"])
           for entry in entries):
        raise ValueError("Invalid frozen candidate identity")
    request_sha256 = hashlib.sha256(_json_bytes(intent)).hexdigest()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "snapshot-id.txt"
    if identity_path.exists():
        if identity_path.read_text().strip() != snapshot_id:
            raise ValueError("Worker output belongs to a different snapshot")
    else:
        identity_path.write_text(snapshot_id + "\n")
    request_path = output / "request-sha256.txt"
    if request_path.exists():
        if request_path.read_text().strip() != request_sha256:
            raise ValueError("Worker output belongs to a different frozen request")
    else:
        request_path.write_text(request_sha256 + "\n")
    shards, processed = _load_checkpoints(
        store, output, prefix, snapshot_id, request_sha256, entries)
    for stale in output.glob(".shard-*.building"):
        stale.unlink()
    group, body_bytes, model_blobs, stream = [], 0, {}, None
    target_bytes = int(MAX_SHARD_BYTES * .85)

    def publish_group():
        nonlocal group, body_bytes, model_blobs, stream
        if not group:
            return
        ordinal = len(shards) + 1
        shard = stream.finish(max_shard_bytes=MAX_SHARD_BYTES)
        size = shard.stat().st_size
        digest = sha256_file(shard)
        shard_key = f"{prefix}/snapshots/{snapshot_id}/shards/{shard.name}"
        store.put_file(shard, shard_key, digest)
        record = {"candidates": [_candidate(entry) for _, entry in group],
                  "object_key": shard_key, "sha256": digest,
                  "byte_length": size}
        _write_checkpoint(output, ordinal, request_sha256,
                          group[0][0], group[-1][0], record)
        shards.append(record)
        group, body_bytes, model_blobs, stream = [], 0, {}, None

    try:
        for ordinal, entry in enumerate(entries[processed:], processed + 1):
            candidate_id, version_id = entry["candidate_id"], entry["version_id"]
            if entry["commit_key"] != f"{prefix}/candidates/{candidate_id}/{version_id}.json" \
                    or not entry["review_key"].startswith(
                        f"{prefix}/reviews/{candidate_id}/{version_id}/revisions/"):
                raise ValueError("Frozen entry escapes candidate namespace")
            commit = _verified_json(store, entry["commit_key"], entry["commit_sha256"])
            revision = _verified_json(store, entry["review_key"], entry["review_sha256"])
            if commit.get("candidate_id") != candidate_id \
                    or commit.get("version_id") != version_id \
                    or revision.get("candidate_id") != candidate_id \
                    or revision.get("version_id") != version_id \
                    or revision.get("candidate_commit_sha256") != entry["commit_sha256"] \
                    or revision.get("decision") != "pass":
                raise ValueError("Snapshot entry is not a bound human pass")
            labels = revision.get("labels") or []
            if "label_key" in entry:
                label_key = entry["label_key"]
                if not label_key.startswith(f"{prefix}/labels/{candidate_id}/{version_id}/"):
                    raise ValueError("Frozen label escapes candidate namespace")
                label_revision = _verified_json(store, label_key, entry["label_sha256"])
                if label_revision.get("candidate_id") != candidate_id \
                        or label_revision.get("version_id") != version_id \
                        or label_revision.get("candidate_commit_sha256") != entry["commit_sha256"] \
                        or not isinstance(label_revision.get("label"), dict):
                    raise ValueError("Frozen label is not bound to candidate")
                labels = [label_revision["label"]]
            if len(labels) != 1 or not labels[0].get("code"):
                raise ValueError("Snapshot entry has no resolved activity label")
            clip_dir = output / "clips" / f"{ordinal:04}-{candidate_id}"
            clip_dir.mkdir(parents=True, exist_ok=True)
            roles = {}
            for item in commit["objects"]:
                if item["role"] not in {"motion", "sensors", "selection"} \
                        or not item["key"].startswith(prefix + "/"):
                    raise ValueError("Invalid candidate object in snapshot")
                suffix = ".json" if item["role"] == "selection" else ".h5"
                path = clip_dir / (item["role"] + suffix)
                store.download_file(item["key"], path, item["sha256"], item["byte_length"])
                roles[item["role"]] = path
            if set(roles) != {"motion", "sensors", "selection"}:
                raise ValueError("Snapshot candidate lacks a required object")
            bundle = clip_dir / "review.review"
            bundle.mkdir(exist_ok=True)
            for item in commit["bundle_files"]:
                name = item["name"]
                if Path(name).name != name or not item["key"].startswith(prefix + "/"):
                    raise ValueError("Invalid preview file in snapshot")
                store.download_file(item["key"], bundle / name,
                                    item["sha256"], item["byte_length"])
            bundle_report = validate_bundle(bundle)
            latest_path = bundle / f"review-r{bundle_report['latest_revision']}.json"
            latest = json.loads(latest_path.read_text())
            if bundle_report["decision"] == "accepted":
                if latest["reviewer"] != revision["reviewer"] \
                        or latest["labels"] != labels:
                    raise ValueError("Existing export review differs")
            else:
                accepted = dict(latest, revision=bundle_report["latest_revision"] + 1,
                                decision="accepted", reviewer=revision["reviewer"],
                                reason=revision.get("reason") or "platform-human-pass",
                                labels=labels, automatic_qa_passed=True)
                accepted_path = bundle / f"review-r{accepted['revision']}.json"
                accepted_path.write_text(json.dumps(accepted, ensure_ascii=False, indent=2) + "\n")
            validate_bundle(bundle)
            clip = output / "clip-deliveries" / f"clip-{ordinal:04}.h5"
            clip.parent.mkdir(exist_ok=True)
            if clip.exists():
                validate_delivery(clip)
            else:
                export_kinematic(
                    roles["motion"], roles["sensors"], roles["selection"], bundle,
                    layout, clip, dataset_id=snapshot_id,
                    model_archive=model_archive, dmpl_archive=dmpl_archive,
                    include_replay=True)
            body, blobs = estimate_clip_delivery(clip, dataset_id=snapshot_id)
            projected = body_bytes + body + sum({**model_blobs, **blobs}.values())
            if group and projected > target_bytes:
                publish_group()
            if stream is None:
                stream = StreamingShard(output / f"shard-{len(shards) + 1:04d}.h5",
                                        snapshot_id)
            stream.append(clip, frozen_entry=entry)
            group.append((ordinal, entry))
            body_bytes += body
            model_blobs.update(blobs)
            clip.unlink(missing_ok=True)
            shutil.rmtree(clip_dir)
        publish_group()
    finally:
        if stream is not None:
            stream.abort()
    result = {
        "schema": "imu_motion_simulator.synthetic_snapshot_result.v1",
        "snapshot_id": snapshot_id, "state": "complete",
        "request_key": request_key,
        "request_sha256": request_sha256,
        "hdf5_version": "3.3.0", "media_policy": "replay-only-no-mp4",
        "candidate_count": len(entries), "shards": shards,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    store.put_json(result_key, result)
    return result


def run_snapshot(store, run_id, snapshot_id, output, *, model_archive,
                 dmpl_archive, layout, target="dev"):
    try:
        return _run_snapshot(store, run_id, snapshot_id, output,
                             model_archive=model_archive,
                             dmpl_archive=dmpl_archive, layout=layout,
                             target=target)
    except Exception as error:
        prefix = publication_prefix(target, run_id)
        failure_key = (f"{prefix}/snapshots/failures/{snapshot_id}/"
                       + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
                       + ".json")
        try:
            store.put_json(failure_key, {
                "schema": "imu_motion_simulator.snapshot_failure.v1",
                "snapshot_id": snapshot_id, "state": "failed",
                "error": str(error)[:1000],
                "created_at_utc": datetime.now(UTC).isoformat(),
            })
        except Exception:
            logging.getLogger(__name__).warning(
                "Could not publish snapshot failure record", exc_info=True)
        raise


def pending_snapshot_ids(store, run_id, *, target="dev"):
    """Find requests not yet completed or failed for a single worker process."""
    prefix = publication_prefix(target, run_id) + "/snapshots/"
    requests = set()
    completed = set()
    failed = set()
    for key in store.list_keys(prefix + "requests/"):
        name = key.removeprefix(prefix + "requests/")
        if name.endswith(".json") and "/" not in name \
                and SNAPSHOT_ID.fullmatch(name[:-5]):
            requests.add(name[:-5])
    for key in store.list_keys(prefix + "results/"):
        name = key.removeprefix(prefix + "results/")
        if name.endswith(".json"):
            completed.add(name[:-5])
    for key in store.list_keys(prefix + "failures/"):
        name = key.removeprefix(prefix + "failures/")
        if "/" in name:
            failed.add(name.split("/", 1)[0])
    return sorted(requests - completed - failed)


def retry_failed_snapshot(store, run_id, snapshot_id, output, *, model_archive,
                          dmpl_archive, layout, target="dev"):
    """Explicitly retry one failed request without changing its frozen inputs."""
    if not SNAPSHOT_ID.fullmatch(snapshot_id):
        raise ValueError("Invalid snapshot ID")
    prefix = publication_prefix(target, run_id) + "/snapshots/"
    store.read_json(f"{prefix}requests/{snapshot_id}.json")
    try:
        store.read_json(f"{prefix}results/{snapshot_id}.json")
    except FileNotFoundError:
        pass
    else:
        raise ValueError("Snapshot is already complete")
    if not store.list_keys(f"{prefix}failures/{snapshot_id}/"):
        raise ValueError("Snapshot has no recorded failure to retry")
    return run_snapshot(store, run_id, snapshot_id, output,
                        model_archive=model_archive,
                        dmpl_archive=dmpl_archive, layout=layout,
                        target=target)


def run_pending_once(store, run_id, output_root, *, model_archive,
                     dmpl_archive, layout, target="dev"):
    """Process every queued request; failures remain visible for manual retry."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for snapshot_id in pending_snapshot_ids(store, run_id, target=target):
        try:
            result = run_snapshot(
                store, run_id, snapshot_id, output_root / snapshot_id,
                model_archive=model_archive, dmpl_archive=dmpl_archive,
                layout=layout, target=target)
            results.append({"snapshot_id": snapshot_id, "state": result["state"]})
        except Exception as error:  # noqa: BLE001 - keep processing other requests
            results.append({"snapshot_id": snapshot_id, "state": "failed",
                            "error": str(error)[:1000]})
    return results
