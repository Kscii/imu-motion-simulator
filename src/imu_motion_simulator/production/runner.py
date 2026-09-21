"""One-pass archive reader with bounded per-clip sensor and publication work."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import tarfile
import time
from uuid import uuid4

from ..contracts.common import sha256_file
from ..contracts.internal import read_internal
from ..motion.kinematics import load_model_member
from ..motion.selection import validate_selection, write_selection
from ..motion.smplh import NATIVE_ADAPTER, decode_amass_member
from ..pipeline.kinematic import _sensor_task, _token
from ..pipeline.machine_review import (_object_descriptor, evaluate_machine_qa,
                                       load_machine_policy, validate_candidate_corpus)
from ..pipeline.plan import load_plan, resolve_inside
from ..publication import (GcsPublicationStore, LocalPublicationStore,
                           publish_candidate)
from ..review.bundle import build_bundle, review_recipe_sha256, validate_bundle
from ..sensors.convergence import convergence_recipe_sha256
from ..sensors.derive import sensor_recipe_sha256
from ..sensors.layout import load_layout, load_profile
from .state import JobState
from .upload_queue import DurablePublisher


def _timed_sensor_task(request):
    started = time.monotonic()
    clip_id, result = _sensor_task(request)
    result["_sensor_compute_s"] = time.monotonic() - started
    result["_sensor_started_at"] = started
    return clip_id, result


def _timed_archive_members(handle, timing: dict):
    """Measure stream advancement, including decompression of skipped members."""
    iterator = iter(handle)
    while True:
        started = time.monotonic()
        try:
            member = next(iterator)
        except StopIteration:
            timing["archive_stream_next_s"] += time.monotonic() - started
            return
        timing["archive_stream_next_s"] += time.monotonic() - started
        yield member


def _store(config: dict):
    publication = config["publication"]
    if publication["backend"] == "local":
        return LocalPublicationStore(publication["root"])
    if publication["backend"] == "gcs":
        return GcsPublicationStore(publication["bucket"], publication["project"])
    return None


def _candidate_corpus(clip: dict, paths: dict, qa: dict, policy_path: Path,
                      policy: dict, plan_path: Path) -> tuple[dict, dict]:
    descriptors = [_object_descriptor(paths[role], role)
                   for role in ("motion", "selection", "sensors")]
    candidate = {
        "candidate_id": clip["id"], "source_dataset": clip["source_dataset"],
        "source_member": clip["source_member"],
        "label_candidates": clip["label_candidates"],
        "warning_flags": list(qa["warnings"]),
        "objects": {row["role"]: row["sha256"] for row in descriptors},
    }
    corpus = {
        "schema": "imu_motion_simulator.candidate_corpus.v1",
        "corpus_id": clip["id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prefix": "synthetic-motion/v1",
        "policy": {"path": str(policy_path), "sha256": sha256_file(policy_path),
                   "policy_id": policy["policy_id"]},
        "inputs": {"plan": str(plan_path), "plan_sha256": sha256_file(plan_path)},
        "statistics": {"total_clips": 1, "machine_passed_before_quarantine": 1,
                       "published_candidates": 1, "candidate_rate": 1.0,
                       "publishable": True,
                       "minimum_corpus_pass_rate": policy["minimum_corpus_pass_rate"]},
        "objects": sorted(descriptors, key=lambda row: row["sha256"]),
        "candidates": [candidate], "quarantined_sources": [], "excluded": [],
    }
    validate_candidate_corpus(corpus)
    return corpus, candidate


def _prepare_clip(clip: dict, *, raw: bytes | None, archive: Path, archive_hash: str,
                  smplh: Path, adapter_id: str, run: Path, layout_id: str,
                  layout_hash: str, profile_hash: str, model_hash: str,
                  layout_path: Path, profile_path: Path,
                  model_cache: dict) -> tuple[dict, tuple]:
    motion = run / "motions" / f"{clip['id']}.motion.h5"
    if not motion.exists():
        if raw is None:
            raise ValueError("Motion is missing and source member was not read")
        decode_amass_member(
            archive, smplh, clip["source_member"], motion,
            source_dataset=clip["source_dataset"],
            original_archive_sha256=archive_hash, source_bytes=raw,
            model_cache=model_cache, adapter_id=adapter_id)
    description, metadata, _ = read_internal(motion, "motion")
    info = metadata["kind_metadata"]
    if (description["frames"] != clip["expected_frames"]
            or info["source_gender"] != clip["expected_gender"]
            or info["source_member"] != clip["source_member"]
            or info["original_archive_sha256"] != archive_hash):
        raise ValueError("Motion differs from the frozen source plan")
    selection = run / "selections" / f"{clip['id']}.selection.json"
    if not selection.exists():
        write_selection(selection, motion, start_frame=clip["frame_range"][0],
                        stop_frame=clip["frame_range"][1],
                        label_candidates=clip["label_candidates"])
    validate_selection(selection, motion)
    motion_sha = sha256_file(motion)
    selection_sha = sha256_file(selection)
    sensor_recipe = sensor_recipe_sha256()
    identity = _token(motion_sha, selection_sha, layout_hash, profile_hash,
                      sensor_recipe)
    sensors = run / "sensors" / f"{clip['id']}-{layout_id}-{identity}.sensors.h5"
    task = (
        clip["id"], motion, selection, sensors, smplh, layout_path, profile_path,
        model_hash, None, None, None, sensor_recipe,
        convergence_recipe_sha256(),
    )
    return {"clip": clip, "motion": motion, "selection": selection,
            "sensors": sensors, "motion_sha": motion_sha,
            "selection_sha": selection_sha, "run": run,
            "smplh": smplh, "layout_path": layout_path,
            "model_cache": model_cache}, task


def _finish_clip(context: dict, result: dict, *, state: JobState, store,
                 policy: dict, policy_path: Path, plan_path: Path,
                 publication: dict, publisher: DurablePublisher | None = None) -> None:
    clip = context["clip"]
    paths = {role: context[role] for role in ("motion", "selection", "sensors")}
    if Path(result["sensors"]).resolve() != paths["sensors"].resolve():
        raise ValueError("Sensor worker returned a different artifact")
    if sha256_file(paths["sensors"]) != result["sensors_sha256"]:
        raise ValueError("Sensor worker output hash differs")
    decision = evaluate_machine_qa(result["automatic_qa"], policy)
    if not decision["passed"]:
        state.record_clip(clip["id"], clip["source_dataset"], "excluded",
                          error=",".join(decision["reasons"]))
        return
    bundle_started = time.monotonic()
    review_identity = _token(context["motion_sha"], result["sensors_sha256"],
                             context["selection_sha"], review_recipe_sha256())
    review = context["run"] / "reviews" / f"{clip['id']}-{review_identity}.review"
    if not review.exists():
        temporary = review.with_name(review.name + ".partial-" + uuid4().hex)
        try:
            gender = read_internal(paths["motion"], "motion")[1]["kind_metadata"]["source_gender"]
            cached = context["model_cache"].get(gender)
            model = cached[1] if cached else context["review_models"].get(gender)
            if model is None:
                model = load_model_member(context["smplh"], gender)
                context["review_models"][gender] = model
            build_bundle(paths["motion"], paths["sensors"], context["smplh"],
                         context["layout_path"], temporary,
                         selection=paths["selection"],
                         convergence=result["convergence"], model=model)
            validate_bundle(temporary)
            temporary.rename(review)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    validate_bundle(review)
    corpus, candidate = _candidate_corpus(
        clip, paths, result["automatic_qa"], policy_path, policy, plan_path)
    state.record_phase(clip["id"], "bundle", time.monotonic() - bundle_started)
    if publication["target"] == "local":
        state.record_clip(clip["id"], clip["source_dataset"], "ready-local")
        return
    size = sum(path.stat().st_size for path in paths.values())
    size += sum(path.stat().st_size for path in review.iterdir() if path.is_file())
    if publisher is not None:
        state.enqueue_upload(clip["id"], clip["source_dataset"], {
            "corpus": corpus, "candidate": candidate, "bundle": str(review),
        }, size)
        return
    upload_started = time.monotonic()
    outcome = publish_candidate(
        corpus, candidate, review, store, target=publication["target"],
        run_id=publication["run_id"], outbox=state.root / "outbox")
    state.record_phase(clip["id"], "upload", time.monotonic() - upload_started)
    state.record_clip(clip["id"], clip["source_dataset"], "published",
                      commit_key=outcome["commit_key"], published_bytes=size)


def _source_plan_rows(config: dict) -> list[tuple[str, Path, dict, list[dict]]]:
    catalog_root = Path(config["catalog"])
    catalog_path = catalog_root / "catalog.json"
    if sha256_file(catalog_path) != config["catalog_sha256"]:
        raise ValueError("Frozen source catalog has changed")
    catalog = json.loads(catalog_path.read_text())
    if catalog.get("schema") not in {"imu_motion_simulator.amass_catalog.v1",
                                     "imu_motion_simulator.amass_catalog.v2"}:
        raise ValueError("Unsupported AMASS source catalog")
    available = {row["source_dataset"]: row for row in catalog["sources"]
                 if row["status"] == "plan-ready"}
    selected = config["sources"] or sorted(available)
    if set(selected) - set(available):
        raise ValueError("Requested source is not plan-ready: "
                         + ", ".join(sorted(set(selected) - set(available))))
    rows = []
    for source in selected:
        entry = available[source]
        plan_path = catalog_root / entry["plan"]
        if sha256_file(plan_path) != entry["plan_sha256"]:
            raise ValueError("Source plan hash differs: " + source)
        plan = load_plan(plan_path)
        clips = plan["clips"]
        if config["clips_per_source"] is not None:
            clips = clips[:config["clips_per_source"]]
        rows.append((source, plan_path, plan, clips))
    return rows


def _run_job(config: dict, *, on_progress=None, state: JobState | None = None) -> dict:
    """Run or resume a job. Every committed clip is terminal and never rebuilt."""
    state = state or JobState(config)
    run_started = time.monotonic()
    initial_counts = state.summary()["counts"]
    initial_published = initial_counts.get("published", 0)
    initial_settled = sum(initial_counts.get(status, 0) for status in
                          ("published", "ready-local", "excluded", "failed"))
    first_candidate_latency_s = None
    state.request("run")
    state.set_status("running")
    policy_path = Path(config["policy"])
    if sha256_file(policy_path) != config["policy_sha256"]:
        raise ValueError("Frozen machine policy has changed")
    policy = load_machine_policy(policy_path)
    rows = _source_plan_rows(config)
    state.emit("plan", {"sources": len(rows),
                        "planned_clips": sum(len(clips) for _, _, _, clips in rows)})
    publication = config["publication"]
    runtime_upload = state.upload_runtime()
    upload = runtime_upload or config.get("upload", {})
    if state.upload_summary()["count"] and upload.get("mode") != "durable":
        raise RuntimeError("Pending upload intents require their durable runtime profile")
    effective_config = {**config, "upload": upload}
    publisher = (DurablePublisher(state, effective_config, _store)
                 if upload.get("mode") == "durable" else None)
    store = None if publisher is not None else _store(config)
    source_errors = []
    source_results = {}

    def notify():
        nonlocal first_candidate_latency_s
        summary = state.summary()
        if (first_candidate_latency_s is None
                and summary["counts"].get("published", 0) > initial_published):
            first_candidate_latency_s = time.monotonic() - run_started
        if on_progress is not None:
            on_progress(summary)

    if publisher is not None:
        state._active_publisher = publisher
        publisher.on_published = notify
        publisher.start()

    for source, plan_path, plan, clips in rows:
        if state.control() != "run":
            break
        if source in policy["paused_sources"]:
            source_errors.append(source)
            state.emit("source-paused", {"source_dataset": source})
            source_results[source] = {"planned": len(clips), "processed": 0,
                                      "published": 0, "ready_local": 0,
                                      "pending_publish": 0, "excluded": 0,
                                      "failed": 0, "status": "policy-paused"}
            continue
        run = state.root / "sources" / source / "run"
        for directory in ("motions", "selections", "sensors", "reviews"):
            (run / directory).mkdir(parents=True, exist_ok=True)
        archive = resolve_inside(config["library_root"], plan["inputs"]["amass_archive"])
        smplh = resolve_inside(config["library_root"], plan["inputs"]["smplh_archive"])
        layout_path = resolve_inside(config["checkout"], plan["sensor"]["layout"])
        profile_path = resolve_inside(config["checkout"], plan["sensor"]["profile"])
        for required in (archive, smplh, layout_path, profile_path):
            if not required.is_file():
                raise FileNotFoundError(required)
        layout_id = load_layout(layout_path)["layout_id"]
        load_profile(profile_path)
        source_hash_started = time.monotonic()
        archive_hash = sha256_file(archive)
        state.record_source_phase(source, "archive-hash",
                                  time.monotonic() - source_hash_started)
        model_hash_started = time.monotonic()
        model_hash = sha256_file(smplh)
        state.record_source_phase(source, "model-hash",
                                  time.monotonic() - model_hash_started)
        layout_hash = sha256_file(layout_path)
        profile_hash = sha256_file(profile_path)
        state.bind_source(source, {
            "plan_sha256": sha256_file(plan_path),
            "archive_sha256": archive_hash, "model_sha256": model_hash,
            "layout_sha256": layout_hash, "profile_sha256": profile_hash})
        adapter_id = plan.get("source_adapter", {}).get("id", NATIVE_ADAPTER)
        pending = {clip["source_member"]: clip for clip in clips
                   if state.clip_status(clip["id"]) not in {
                       "published", "ready-local", "excluded", "pending-publish"}}
        if not pending:
            source_results[source] = state.source_summary(source, len(clips))
            continue
        state.emit("source-started", {"source_dataset": source,
                                      "pending_clips": len(pending)})
        model_cache = {}
        review_models = {}
        futures = {}
        source_started = time.monotonic()
        extracted_clips = 0

        def complete(future):
            context = futures.pop(future)
            clip = context["clip"]
            try:
                result_id, result = future.result()
                if result_id != clip["id"]:
                    raise ValueError("Sensor worker clip ID differs")
                sensor_elapsed = time.monotonic() - context["submitted_at"]
                state.record_phase(clip["id"], "sensor", sensor_elapsed)
                compute = result.pop("_sensor_compute_s", None)
                started_at = result.pop("_sensor_started_at", None)
                if compute is not None:
                    state.record_phase(clip["id"], "sensor-compute", compute)
                    state.record_phase(clip["id"], "sensor-queue-and-return",
                                       max(0.0, sensor_elapsed - compute))
                    if started_at is not None:
                        dispatch = max(0.0, started_at - context["submitted_at"])
                        state.record_phase(clip["id"], "sensor-dispatch-wait",
                                           dispatch)
                        state.record_phase(clip["id"], "sensor-return-and-collection",
                                           max(0.0, sensor_elapsed - dispatch - compute))
                _finish_clip(context, result, state=state, store=store,
                             policy=policy, policy_path=policy_path,
                             plan_path=plan_path, publication=publication,
                             publisher=publisher)
            except Exception as error:
                state.record_clip(clip["id"], source, "failed", error=str(error))
            notify()

        scan_timing = {"archive_stream_next_s": 0.0}
        with ProcessPoolExecutor(max_workers=config["sensor_workers"]) as pool:
            with tarfile.open(archive, "r|bz2") as handle:
                for member in _timed_archive_members(handle, scan_timing):
                    if publisher is not None and not publisher.wait_for_capacity():
                        break
                    clip = pending.pop(member.name, None)
                    if clip is None:
                        continue
                    if not member.isfile():
                        state.record_clip(clip["id"], source, "failed",
                                          error="Source member is not a file")
                        notify()
                        continue
                    try:
                        extract_started = time.monotonic()
                        raw = handle.extractfile(member).read()
                        state.record_phase(clip["id"], "extract",
                                           time.monotonic() - extract_started)
                        extracted_clips += 1
                        prepare_started = time.monotonic()
                        context, task = _prepare_clip(
                            clip, raw=raw, archive=archive, archive_hash=archive_hash,
                            smplh=smplh, adapter_id=adapter_id, run=run,
                            layout_id=layout_id, layout_hash=layout_hash,
                            profile_hash=profile_hash, model_hash=model_hash,
                            layout_path=layout_path, profile_path=profile_path,
                            model_cache=model_cache)
                        state.record_phase(clip["id"], "prepare",
                                           time.monotonic() - prepare_started)
                        context["review_models"] = review_models
                        context["submitted_at"] = time.monotonic()
                        future = pool.submit(_timed_sensor_task, task)
                        futures[future] = context
                    except Exception as error:
                        state.record_clip(clip["id"], source, "failed", error=str(error))
                        notify()
                    while len(futures) >= 2 * config["sensor_workers"]:
                        done, _ = wait(futures, return_when=FIRST_COMPLETED)
                        for future in done:
                            complete(future)
                    if state.control() != "run":
                        break
                    if not pending:
                        break
            for future in list(futures):
                complete(future)
        state.record_source_phase(source, "archive-stream-next",
                                  scan_timing["archive_stream_next_s"])
        if state.control() == "run":
            for clip in pending.values():
                state.record_clip(clip["id"], source, "failed",
                                  error="Planned source member was not found in archive")
                notify()
        state.emit("source-finished", {"source_dataset": source})
        state.emit("source-timing", {"source_dataset": source,
                                     "archive_wall_s": round(time.monotonic() - source_started, 3),
                                     "selected_clips": extracted_clips})
        source_results[source] = state.source_summary(source, len(clips))

    if publisher is not None:
        publisher.drain()
        source_results = {source: (source_results[source]
                                   if source_results.get(source, {}).get("status") == "policy-paused"
                                   else state.source_summary(source, len(clips)))
                          for source, _, _, clips in rows}

    for source, _, _, clips in rows:
        source_results.setdefault(source, state.source_summary(source, len(clips)))
    source_rate_warnings = [source for source, value in source_results.items()
                            if value["status"] == "complete" and value["planned"]
                            and (value["published"] + value["ready_local"])
                            / value["planned"] < policy["minimum_source_pass_rate"]]
    for source in source_rate_warnings:
        state.emit("source-pass-rate-warning", {
            "source_dataset": source,
            "minimum": policy["minimum_source_pass_rate"],
            "action": "monitor-only; no automatic source quarantine"})
    planned_total = sum(len(clips) for _, _, _, clips in rows)
    settled_total = sum(value["published"] + value["ready_local"]
                        + value["excluded"] + value["failed"]
                        for value in source_results.values())
    passed_total = sum(value["published"] + value["ready_local"]
                       for value in source_results.values())
    corpus_rate = passed_total / planned_total if planned_total else None
    corpus_rate_warning = (settled_total == planned_total
                           and corpus_rate is not None
                           and corpus_rate < policy["minimum_corpus_pass_rate"])
    if corpus_rate_warning:
        state.emit("corpus-pass-rate-warning", {
            "candidate_rate": corpus_rate,
            "minimum": policy["minimum_corpus_pass_rate"],
            "action": "monitor-only; no automatic publication veto"})

    control = state.control()
    if control == "pause":
        state.set_status("paused")
    elif control == "cancel":
        state.set_status("cancelled")
    else:
        summary = state.summary()
        state.set_status("partial" if source_errors or summary["counts"].get("failed")
                         else "complete")
    report = state.summary()
    report["source_errors"] = source_errors
    report["source_rate_warnings"] = source_rate_warnings
    report["sources"] = source_results
    report["planned_clips"] = planned_total
    report["candidate_rate"] = corpus_rate
    report["corpus_rate_warning"] = corpus_rate_warning
    report["run_wall_seconds"] = round(time.monotonic() - run_started, 3)
    report["first_candidate_latency_s"] = (
        round(first_candidate_latency_s, 3)
        if first_candidate_latency_s is not None else None)
    report["settled_clips_per_minute"] = round(
        max(0, settled_total - initial_settled)
        / max(report["run_wall_seconds"], 0.001) * 60, 3)
    report["phase_metrics"] = state.phase_summary()
    report["source_phase_metrics"] = state.source_phase_summary()
    report_path = state.root / "production-report.json"
    temporary = report_path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, report_path)
    notify()
    return report


def run_job(config: dict, *, on_progress=None, upload_profile: dict | None = None) -> dict:
    """Run or resume a job, retaining a failed status on fatal errors."""
    state = JobState(config)
    if state.summary()["status"] in {"complete", "cancelled"}:
        raise ValueError("A completed or cancelled production job cannot be restarted")
    lock_path = state.root / ".production.lock"
    try:
        with lock_path.open("w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Production job is already running") from error
            if upload_profile is not None:
                state.bind_upload_runtime(upload_profile)
            return _run_job(config, on_progress=on_progress, state=state)
    except Exception as error:
        try:
            if str(error) != "Production job is already running":
                state.emit("fatal-error", {"error": str(error)[:1000]})
                state.set_status("failed")
        except Exception:
            pass
        raise
    finally:
        publisher = getattr(state, "_active_publisher", None)
        if publisher is not None:
            publisher.stop()
