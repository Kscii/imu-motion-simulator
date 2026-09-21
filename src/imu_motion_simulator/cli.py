"""Unified command line interface for the simulator research repository."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def project_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src/imu_motion_simulator").is_dir():
            return candidate
    raise FileNotFoundError("could not locate the imu-motion-simulator checkout")


def print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def contracts_validate(args: argparse.Namespace) -> int:
    from .contracts.delivery import validate_delivery
    from .contracts.internal import validate_internal

    report = validate_delivery(args.path, full=not args.core_only) if args.kind == "delivery" else validate_internal(args.path, args.kind)
    print_json(report)
    return 0


def contracts_fixtures(args: argparse.Namespace) -> int:
    from .contracts.fixtures import build

    print_json(build(args.output))
    return 0


def contracts_migrate(args: argparse.Namespace) -> int:
    from .contracts.delivery import CATALOG, VERSIONS, migrate_v32, table

    metadata = json.loads(args.metadata.read_text())
    labels = metadata.get("labels")
    if labels is not None:
        def rows(name: str, fields):
            return table(
                [tuple(row[field] for field, _ in fields) for row in labels[name]],
                fields,
            )

        labels = {
            "catalog": rows("catalog", CATALOG),
            "sequence_versions": rows("sequence_versions", VERSIONS),
        }
    report = migrate_v32(
        args.source,
        args.target,
        labels=labels,
        provenance=metadata["provenance"],
    )
    print_json(report)
    return 0


def library_acquire(args: argparse.Namespace) -> int:
    from .library.acquire import main as acquire_main

    return acquire_main(
        [
            "--root",
            str(args.root),
            "--manifest",
            str(args.manifest),
            "--workers",
            str(args.workers),
            "--reserve-gib",
            str(args.reserve_gib),
        ]
    )


def library_status(args: argparse.Namespace) -> int:
    from .library.status import inspect_library

    report = inspect_library(args.root, manifest=args.manifest, inventory=args.inventory,
                             verify_content=args.verify_content)
    print_json(report)
    failures = {"hash_mismatch", "missing", "size_mismatch", "unreceipted_final"}
    return 0 if not failures & set(report["counts"]) else 2


def library_prepare(args: argparse.Namespace) -> int:
    from .library.sources import prepare

    print_json(prepare(args.output, args.library_root, args.catalog))
    return 0


def pipeline_validate(args: argparse.Namespace) -> int:
    from .pipeline.plan import describe_plan

    print_json(describe_plan(args.plan))
    return 0


def pipeline_run(args: argparse.Namespace) -> int:
    from .pipeline.run import run_pipeline

    report = run_pipeline(
        args.plan, args.library_root, args.workspace, args.checkout, args.output,
        source_artifact=args.source_artifact, through=args.through,
        sensor_workers=args.sensor_workers)
    print_json(report)
    return (0 if report.get("stage") != "review"
            or report["automatic_qa_passed"] else 2)


def review_validate(args: argparse.Namespace) -> int:
    from .review import validate_bundle

    print_json(validate_bundle(args.path))
    return 0


def review_serve(args: argparse.Namespace) -> int:
    from .review import serve
    serve(args.path, host=args.host, port=args.port, open_browser=args.open)
    return 0


def review_render(args: argparse.Namespace) -> int:
    from .review.render import render_mp4
    print_json(render_mp4(
        args.motion, args.sensors, args.model_archive, args.dmpl_archive,
        args.layout, args.blender, args.output, selection=args.selection,
        fps=args.fps, width=args.width, height=args.height))
    return 0


def review_flag_quality(args: argparse.Namespace) -> int:
    from .review import append_quality_flag
    print_json(append_quality_flag(
        args.path, reviewer=args.reviewer, reason=args.reason,
        start_frame=args.start_frame, stop_frame=args.stop_frame,
        joints=args.joint))
    return 0


def sensors_derive(args: argparse.Namespace) -> int:
    from .sensors import derive_ideal
    print_json(derive_ideal(args.motion, args.model_archive, args.layout,
                            args.profile, args.output, selection=args.selection))
    return 0


def sensors_calibrate(args: argparse.Namespace) -> int:
    from .sensors import derive_calibrated
    print_json(derive_calibrated(args.ideal, args.profile, args.output,
                                 seed=args.seed))
    return 0


def delivery_export_kinematic(args: argparse.Namespace) -> int:
    from .delivery_kinematic import export_kinematic
    print_json(export_kinematic(
        args.motion, args.sensors, args.selection, args.review, args.layout,
        args.output, dataset_id=args.dataset_id,
        model_archive=args.model_archive, dmpl_archive=args.dmpl_archive,
        include_replay=args.include_replay, mp4=args.mp4))
    return 0


def motion_import_bvh(args: argparse.Namespace) -> int:
    from .motion.bvh import decode_bvh
    print_json(decode_bvh(
        args.source, args.model_archive, args.mapping, args.output,
        zip_member=args.zip_member, source_dataset=args.source_dataset,
        source_gender=args.source_gender))
    return 0


def motion_select(args: argparse.Namespace) -> int:
    from .motion.selection import write_selection
    labels = [] if args.labels is None else json.loads(args.labels.read_text())
    print_json(write_selection(
        args.output, args.motion, start_frame=args.start_frame,
        stop_frame=args.stop_frame, label_candidates=labels))
    return 0


def labels_babel(args: argparse.Namespace) -> int:
    from .labels import candidates_for_member
    print_json({'candidate': candidates_for_member(
        args.archive, args.source_member,
        source_dataset=args.source_dataset)})
    return 0


def pipeline_make_amass_plan(args: argparse.Namespace) -> int:
    from .pipeline.amass_plan import build_amass_plan
    print_json(build_amass_plan(
        args.library_root, args.output, study_id=args.study_id,
        source_dataset=args.source_dataset, amass_archive=args.amass_archive,
        smplh_archive=args.smplh_archive, dmpl_archive=args.dmpl_archive,
        layout=args.layout, profile=args.profile,
        babel_archive=args.babel_archive, adapter_id=args.adapter))
    return 0


def pipeline_make_amass_catalog(args: argparse.Namespace) -> int:
    from .pipeline.amass_plan import build_amass_catalog
    adapters = {}
    for value in args.adapter:
        source, separator, adapter = value.partition('=')
        if not separator or not source or not adapter or source in adapters:
            raise ValueError(
                '--adapter must be a unique SOURCE=ADAPTER_ID value')
        adapters[source] = adapter
    print_json(build_amass_catalog(
        args.library_root, args.output,
        amass_directory=args.amass_directory,
        smplh_archive=args.smplh_archive, dmpl_archive=args.dmpl_archive,
        layout=args.layout, profile=args.profile,
        babel_archive=args.babel_archive,
        include_sources=args.include_source, source_adapters=adapters))
    return 0


def pipeline_run_amass_catalog(args: argparse.Namespace) -> int:
    from .pipeline.amass_batch import run_amass_catalog
    reuse = {}
    for value in args.reuse_source:
        source, separator, path = value.partition('=')
        if not separator or not source or not path or source in reuse:
            raise ValueError(
                '--reuse-source must be a unique SOURCE=STUDY_PATH value')
        reuse[source] = Path(path)
    report = run_amass_catalog(
        args.catalog, args.output, library_root=args.library_root,
        checkout=args.checkout, workspace=args.workspace,
        through=args.through, skip_sources=args.skip_source,
        reuse_sources=reuse, workers=args.workers,
        sensor_workers=args.sensor_workers)
    print_json(report)
    return 2 if report['errors'] else 0


def pipeline_audit_amass_batch(args: argparse.Namespace) -> int:
    from .pipeline.amass_batch import audit_amass_batch
    report = audit_amass_batch(args.catalog, args.batch)
    print_json(report)
    return 0 if report['production_complete'] else 2


def pipeline_build_machine_corpus(args: argparse.Namespace) -> int:
    from .pipeline.machine_review import build_machine_corpus
    report = build_machine_corpus(
        args.catalog, args.batch, args.output, args.policy)
    print_json(report)
    return 0 if report['statistics']['publishable'] else 2


def pipeline_merge_machine_corpora(args: argparse.Namespace) -> int:
    from .pipeline.machine_review import merge_candidate_corpora
    print_json(merge_candidate_corpora(args.corpus, args.output))
    return 0


def handoff_fixtures(args: argparse.Namespace) -> int:
    from .contracts.handoff import build_handoff_fixture
    print_json(build_handoff_fixture(args.output))
    return 0


def handoff_validate(args: argparse.Namespace) -> int:
    value = json.loads(args.path.read_text())
    if args.kind == 'candidate':
        from .pipeline.machine_review import validate_candidate_corpus
        report = validate_candidate_corpus(value)
    elif args.kind == 'review':
        from .contracts.handoff import validate_review_revision
        report = validate_review_revision(value)
    else:
        from .contracts.handoff import validate_snapshot_manifest
        report = validate_snapshot_manifest(
            value, root=args.path.parent, full=not args.core_only)
    print_json(report)
    return 0


def handoff_snapshot(args: argparse.Namespace) -> int:
    from .contracts.handoff import build_snapshot
    print_json(build_snapshot(
        args.candidates, args.reviews, args.shard, args.output,
        snapshot_id=args.snapshot_id))
    return 0


def handoff_publish_samples(args: argparse.Namespace) -> int:
    from .publication import (GcsPublicationStore, LocalPublicationStore,
                              choose_pilot, publish_candidate)
    if args.store == 'local' and args.root is None:
        raise ValueError('--root is required for a local publication store')
    if args.store == 'gcs' and not args.bucket:
        raise ValueError('--bucket is required for GCS publication')
    corpus = json.loads(args.candidates.read_text())
    index = json.loads(args.index.read_text())
    store = (LocalPublicationStore(args.root) if args.store == 'local'
             else GcsPublicationStore(args.bucket, args.project))
    candidates = {row['candidate_id']: row for row in corpus['candidates']}
    index_root = args.index.parent.resolve()
    results = []
    for row in choose_pilot(index):
        bundle = (index_root / row['bundle']).resolve()
        if not bundle.is_relative_to(index_root):
            raise ValueError('Review bundle escapes the index directory')
        results.append(publish_candidate(
            corpus, candidates[row['candidate_id']], bundle, store,
            run_id=args.run_id, outbox=args.outbox))
        print_json(results[-1])
    return 0


def handoff_snapshot_worker(args: argparse.Namespace) -> int:
    from .publication import GcsPublicationStore, LocalPublicationStore
    from .snapshot_worker import run_snapshot
    if args.store == 'local' and args.root is None:
        raise ValueError('--root is required for a local publication store')
    if args.store == 'gcs' and not args.bucket:
        raise ValueError('--bucket is required for GCS publication')
    store = (LocalPublicationStore(args.root) if args.store == 'local'
             else GcsPublicationStore(args.bucket, args.project))
    print_json(run_snapshot(
        store, args.run_id, args.snapshot_id, args.output,
        model_archive=args.model_archive, dmpl_archive=args.dmpl_archive,
        layout=args.layout, target=args.target))
    return 0


def handoff_snapshot_worker_auto(args: argparse.Namespace) -> int:
    import fcntl
    import time
    from .publication import GcsPublicationStore, LocalPublicationStore
    from .snapshot_worker import run_pending_once
    if args.store == 'local' and args.root is None:
        raise ValueError('--root is required for a local publication store')
    if args.store == 'gcs' and not args.bucket:
        raise ValueError('--bucket is required for GCS publication')
    if args.poll_interval_s < 5:
        raise ValueError('--poll-interval-s must be at least 5')
    store = (LocalPublicationStore(args.root) if args.store == 'local'
             else GcsPublicationStore(args.bucket, args.project))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.single-worker.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Snapshot worker is already running in this output directory') from error
        while True:
            results = run_pending_once(
                store, args.run_id, args.output,
                model_archive=args.model_archive, dmpl_archive=args.dmpl_archive,
                layout=args.layout, target=args.target)
            if results or args.once:
                print_json({'processed': results})
            if args.once:
                return 0 if all(item['state'] == 'complete' for item in results) else 2
            time.sleep(args.poll_interval_s)


def validation_imucoco_audit(args: argparse.Namespace) -> int:
    from .validation import audit_imucoco
    report = audit_imucoco(args.archive)
    print_json(report)
    return 0 if report['passed'] else 2


def validation_imucoco_benchmark(args: argparse.Namespace) -> int:
    from .validation import benchmark_imucoco
    report = benchmark_imucoco(args.archive, args.output,
                               all_takes=args.all_takes)
    print_json({
        'output': str(args.output), 'takes': len(report['takes']),
        'requested_takes': report['requested_takes'],
        'failed_takes': len(report['failed_takes']),
        'aggregate': report['aggregate']})
    return 0


def sensors_convergence(args: argparse.Namespace) -> int:
    from .sensors.convergence import convergence_report
    print_json(convergence_report(
        args.motion, args.model_archive, args.layout, args.profile,
        selection=args.selection))
    return 0


def review_build(args: argparse.Namespace) -> int:
    from .review.bundle import build_bundle
    from .sensors.convergence import convergence_report
    convergence = convergence_report(
        args.motion, args.model_archive, args.layout, args.profile,
        selection=args.selection)
    print_json(build_bundle(
        args.motion, args.sensors, args.model_archive, args.layout,
        args.output, selection=args.selection, convergence=convergence))
    return 0


def docs_check(args: argparse.Namespace) -> int:
    from .docs_check import check

    report = check(args.root)
    print_json(report)
    return 0 if report["passed"] else 2


def production_validate(args: argparse.Namespace) -> int:
    from .production.config import config_digest, load_job_config
    from .production.runner import _source_plan_rows
    config = load_job_config(args.config)
    rows = _source_plan_rows(config)
    print_json({"valid": True, "config_sha256": config_digest(config),
                "target": config["publication"]["target"],
                "sources": len(rows),
                "planned_clips": sum(len(clips) for _, _, _, clips in rows)})
    return 0


def production_run(args: argparse.Namespace) -> int:
    from .production import load_job_config, run_job
    from .production.upload_runtime import load_upload_profile
    from .production.runner import _source_plan_rows
    config = load_job_config(args.config)
    upload_profile = load_upload_profile(args.upload_profile) if args.upload_profile else None
    if args.json or not sys.stdout.isatty():
        last = None
        def progress(summary):
            nonlocal last
            current = (summary["status"], tuple(sorted(summary["counts"].items())))
            if current != last:
                print(json.dumps(summary, ensure_ascii=False), flush=True)
                last = current
        report = run_job(config, on_progress=progress, upload_profile=upload_profile)
    else:
        from rich.console import Console
        from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                                   SpinnerColumn, TextColumn, TimeElapsedColumn)
        total = sum(len(clips) for _, _, _, clips in _source_plan_rows(config))
        console = Console()
        with Progress(SpinnerColumn(), TextColumn("[cyan]片段生产与发布"),
                      BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
                      console=console, expand=True) as display:
            task = display.add_task("production", total=total)
            def progress(summary):
                counts = summary["counts"]
                display.update(task, completed=sum(counts.values()))
            report = run_job(config, on_progress=progress,
                             upload_profile=upload_profile)
        console.print(f"[bold]任务状态[/bold] {report['status']}  "
                      f"[green]已发布[/green] {report['counts'].get('published', 0)}  "
                      f"[yellow]排除[/yellow] {report['counts'].get('excluded', 0)}  "
                      f"[red]失败[/red] {report['counts'].get('failed', 0)}")
    return 0 if report["status"] == "complete" else 2


def production_status(args: argparse.Namespace) -> int:
    from .production import JobState, load_job_config
    config = load_job_config(args.config)
    if not (Path(config["output"]) / "production.sqlite3").is_file():
        raise FileNotFoundError("Production job has not been started")
    print_json(JobState(config).summary())
    return 0


def production_upload_configure(args: argparse.Namespace) -> int:
    """Bind a frozen job's operational profile only with its producer stopped."""
    import fcntl
    from .production import JobState, load_job_config
    from .production.upload_runtime import load_upload_profile
    config = load_job_config(args.config)
    if not (Path(config["output"]) / "production.sqlite3").is_file():
        raise FileNotFoundError("Production job has not been started")
    state = JobState(config)
    with (state.root / ".production.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Production job is already running") from error
        state.bind_upload_runtime(load_upload_profile(args.profile))
    print_json(state.summary())
    return 0


def production_upload_unblock(args: argparse.Namespace) -> int:
    import fcntl
    from .production import JobState, load_job_config
    state = JobState(load_job_config(args.config))
    with (state.root / ".production.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Production job is already running") from error
        count = state.unblock_uploads()
    print_json({"unblocked": count, **state.summary()})
    return 0


def production_control(args: argparse.Namespace) -> int:
    from .production import JobState, load_job_config, run_job
    config = load_job_config(args.config)
    if not (Path(config["output"]) / "production.sqlite3").is_file():
        raise FileNotFoundError("Production job has not been started")
    state = JobState(config)
    if args.command in {"pause", "cancel"}:
        current = state.summary()["status"]
        if args.command == "pause" and current != "running":
            raise ValueError("Only a running job may be paused")
        if args.command == "cancel" and current not in {
                "queued", "running", "paused", "partial", "failed"}:
            raise ValueError("Job cannot be cancelled in this state")
        state.request(args.command)
        if args.command == "cancel" and current != "running":
            state.set_status("cancelled")
        print_json(state.summary())
        return 0
    status = state.summary()["status"]
    if args.command == "resume" and status != "paused":
        raise ValueError("Only a paused job may be resumed")
    if args.command == "retry" and status not in {"partial", "failed"}:
        raise ValueError("Only a partial or failed job may be retried")
    report = run_job(config)
    print_json(report)
    return 0 if report["status"] == "complete" else 2


def production_watch(args: argparse.Namespace) -> int:
    from .production import JobState, load_job_config
    state = JobState(load_job_config(args.config))
    cursor = args.after
    while True:
        batch = state.events_since(cursor)
        for event in batch:
            cursor = event["seq"]
            print(json.dumps(event, ensure_ascii=False), flush=True)
        if args.once or (state.summary()["status"] in {
                "complete", "partial", "failed", "paused", "cancelled"} and not batch):
            return 0
        time.sleep(1)


def production_serve(args: argparse.Namespace) -> int:
    from .production import load_job_config
    from .production.control import serve
    serve(load_job_config(args.config), port=args.port)
    return 0


def production_snapshot_worker(args: argparse.Namespace) -> int:
    """Run the existing single-instance snapshot worker from the job YAML."""
    from .production import (load_job_config, retry_snapshot,
                             run_pending_snapshots)
    config = load_job_config(args.config)
    if args.retry_failed:
        print_json(retry_snapshot(config, args.retry_failed))
        return 0
    if args.poll_interval_s < 5:
        raise ValueError("--poll-interval-s must be at least 5")
    while True:
        results = run_pending_snapshots(config)
        if results or args.once:
            print_json({"processed": results})
        if args.once:
            return 0 if all(item["state"] == "complete" for item in results) else 2
        time.sleep(args.poll_interval_s)


def production_export_core(args: argparse.Namespace) -> int:
    """Publish a separate, immutable machine-QA-only HDF5 3.3 dataset."""
    from .production.config import load_job_config
    from .provisional import export_provisional
    configs = [load_job_config(path) for path in args.config]
    rules = (json.loads(args.rules_file.read_text(encoding="utf-8"))
             if args.rules_file is not None else None)
    print_json(export_provisional(configs[0], additional_configs=configs[1:],
                                  audit_report=args.audit_report, rules=rules))
    return 0


def production_export_core_watch(args: argparse.Namespace) -> int:
    """Wait for the independent full cloud audit before exporting prod core data."""
    if args.poll_interval_s < 5:
        raise ValueError("--poll-interval-s must be at least 5")
    while True:
        if args.audit_report.is_file():
            audit = json.loads(args.audit_report.read_text(encoding="utf-8"))
            if audit.get("phase") == "complete":
                return production_export_core(args)
            if audit.get("phase") == "failed":
                print_json({"state": "blocked", "reason": "prod cloud audit failed",
                            "detail": audit.get("error")})
                return 3
        service = subprocess.run(
            ["systemctl", "--user", "show", args.chain_unit,
             "--property=ActiveState", "--value"],
            check=True, text=True, capture_output=True).stdout.strip()
        if service != "active":
            print_json({"state": "blocked", "reason": "prod chain stopped before full audit",
                        "chain_unit": args.chain_unit, "service_state": service})
            return 3
        time.sleep(args.poll_interval_s)


def production_dashboard(args: argparse.Namespace) -> int:
    from .production.dashboard import DEFAULT_UNITS, serve
    serve(args.config, audit_report=args.audit_report, port=args.port,
          units=tuple(args.unit) if args.unit else DEFAULT_UNITS)
    return 0


def build_parser() -> argparse.ArgumentParser:
    try:
        root = project_root()
    except FileNotFoundError:
        # Production jobs use explicit paths from their YAML. An installed SDK
        # must be callable from a service working directory outside a checkout.
        root = Path.cwd().resolve()
    parser = argparse.ArgumentParser(prog="imu-sim", description=__doc__)
    groups = parser.add_subparsers(dest="group", required=True)

    motion = groups.add_parser("motion", help="convert source motion to canonical SMPL+H")
    motion_commands = motion.add_subparsers(dest="command", required=True)
    import_bvh = motion_commands.add_parser("import-bvh")
    import_bvh.add_argument("source", type=Path)
    import_bvh.add_argument("output", type=Path)
    import_bvh.add_argument("--model-archive", type=Path, required=True)
    import_bvh.add_argument("--mapping", type=Path, required=True)
    import_bvh.add_argument("--zip-member")
    import_bvh.add_argument("--source-dataset", default="BVH")
    import_bvh.add_argument("--source-gender", choices=["male", "female", "neutral"],
                            default="neutral")
    import_bvh.set_defaults(func=motion_import_bvh)
    select = motion_commands.add_parser("select")
    select.add_argument("motion", type=Path)
    select.add_argument("output", type=Path)
    select.add_argument("--start-frame", type=int, default=0)
    select.add_argument("--stop-frame", type=int)
    select.add_argument("--labels", type=Path,
                        help="JSON array of non-authoritative label candidates")
    select.set_defaults(func=motion_select)

    labels = groups.add_parser("labels", help="inspect non-authoritative label candidates")
    label_commands = labels.add_subparsers(dest="command", required=True)
    babel = label_commands.add_parser("babel")
    babel.add_argument("archive", type=Path); babel.add_argument("source_member")
    babel.add_argument("--source-dataset")
    babel.set_defaults(func=labels_babel)

    contracts = groups.add_parser("contracts", help="validate, generate, or migrate HDF5 contracts")
    contract_commands = contracts.add_subparsers(dest="command", required=True)
    validate_contract = contract_commands.add_parser("validate")
    validate_contract.add_argument("path", type=Path)
    validate_contract.add_argument(
        "--kind", choices=["delivery", "source", "motion", "sensors"],
        default="delivery")
    validate_contract.add_argument("--core-only", action="store_true")
    validate_contract.set_defaults(func=contracts_validate)
    fixtures = contract_commands.add_parser("fixtures")
    fixtures.add_argument("output", type=Path)
    fixtures.set_defaults(func=contracts_fixtures)
    migrate = contract_commands.add_parser("migrate")
    migrate.add_argument("source", type=Path)
    migrate.add_argument("target", type=Path)
    migrate.add_argument("--metadata", type=Path, required=True)
    migrate.set_defaults(func=contracts_migrate)
    export_kinematic = contract_commands.add_parser("export-kinematic")
    export_kinematic.add_argument("output", type=Path)
    export_kinematic.add_argument("--motion", type=Path, required=True)
    export_kinematic.add_argument("--sensors", type=Path, required=True)
    export_kinematic.add_argument("--selection", type=Path, required=True)
    export_kinematic.add_argument("--review", type=Path, required=True)
    export_kinematic.add_argument("--layout", type=Path, required=True)
    export_kinematic.add_argument("--dataset-id", required=True)
    export_kinematic.add_argument("--include-replay", action="store_true")
    export_kinematic.add_argument("--model-archive", type=Path)
    export_kinematic.add_argument("--dmpl-archive", type=Path)
    export_kinematic.add_argument("--mp4", type=Path)
    export_kinematic.set_defaults(func=delivery_export_kinematic)

    handoff = groups.add_parser(
        "handoff", help="build and validate candidate-review snapshot handoffs")
    handoff_commands = handoff.add_subparsers(dest="command", required=True)
    handoff_fixture = handoff_commands.add_parser("fixtures")
    handoff_fixture.add_argument("output", type=Path)
    handoff_fixture.set_defaults(func=handoff_fixtures)
    handoff_check = handoff_commands.add_parser("validate")
    handoff_check.add_argument("path", type=Path)
    handoff_check.add_argument(
        "--kind", choices=["candidate", "review", "snapshot"],
        required=True)
    handoff_check.add_argument("--core-only", action="store_true")
    handoff_check.set_defaults(func=handoff_validate)
    handoff_build = handoff_commands.add_parser("snapshot")
    handoff_build.add_argument("--candidates", type=Path, required=True)
    handoff_build.add_argument("--reviews", type=Path, required=True)
    handoff_build.add_argument("--shard", type=Path, action="append",
                               required=True)
    handoff_build.add_argument("--output", type=Path, required=True)
    handoff_build.add_argument("--snapshot-id", required=True)
    handoff_build.set_defaults(func=handoff_snapshot)
    handoff_pilot = handoff_commands.add_parser(
        "publish-samples", help="publish the deterministic 12-clip dev pilot")
    handoff_pilot.add_argument("--candidates", type=Path, required=True)
    handoff_pilot.add_argument("--index", type=Path, required=True)
    handoff_pilot.add_argument("--store", choices=["local", "gcs"], required=True)
    handoff_pilot.add_argument("--root", type=Path)
    handoff_pilot.add_argument("--bucket")
    handoff_pilot.add_argument("--project")
    handoff_pilot.add_argument("--run-id", required=True)
    handoff_pilot.add_argument("--outbox", type=Path, required=True)
    handoff_pilot.set_defaults(func=handoff_publish_samples)
    handoff_worker = handoff_commands.add_parser(
        "snapshot-worker", help="build one frozen synthetic HDF5 3.3 snapshot")
    handoff_worker.add_argument("--store", choices=["local", "gcs"], required=True)
    handoff_worker.add_argument("--root", type=Path)
    handoff_worker.add_argument("--bucket")
    handoff_worker.add_argument("--project")
    handoff_worker.add_argument("--target", choices=["dev", "prod"], default="dev")
    handoff_worker.add_argument("--run-id")
    handoff_worker.add_argument("--snapshot-id", required=True)
    handoff_worker.add_argument("--output", type=Path, required=True)
    handoff_worker.add_argument("--model-archive", type=Path, required=True)
    handoff_worker.add_argument("--dmpl-archive", type=Path)
    handoff_worker.add_argument("--layout", type=Path, required=True)
    handoff_worker.set_defaults(func=handoff_snapshot_worker)
    handoff_auto = handoff_commands.add_parser(
        "snapshot-worker-auto", help="poll frozen requests with one worker")
    handoff_auto.add_argument("--store", choices=["local", "gcs"], required=True)
    handoff_auto.add_argument("--root", type=Path)
    handoff_auto.add_argument("--bucket")
    handoff_auto.add_argument("--project")
    handoff_auto.add_argument("--target", choices=["dev", "prod"], default="dev")
    handoff_auto.add_argument("--run-id")
    handoff_auto.add_argument("--output", type=Path, required=True)
    handoff_auto.add_argument("--model-archive", type=Path, required=True)
    handoff_auto.add_argument("--dmpl-archive", type=Path)
    handoff_auto.add_argument("--layout", type=Path, required=True)
    handoff_auto.add_argument("--poll-interval-s", type=float, default=30.0)
    handoff_auto.add_argument("--once", action="store_true")
    handoff_auto.set_defaults(func=handoff_snapshot_worker_auto)

    production = groups.add_parser(
        "production", help="run and control resumable per-clip production jobs")
    production_commands = production.add_subparsers(dest="command", required=True)
    for name, handler in (("validate", production_validate),
                          ("run", production_run),
                          ("status", production_status),
                          ("upload-configure", production_upload_configure),
                          ("upload-unblock", production_upload_unblock),
                          ("watch", production_watch),
                          ("pause", production_control),
                          ("resume", production_control),
                          ("retry", production_control),
                          ("cancel", production_control),
                          ("serve", production_serve),
                          ("dashboard", production_dashboard),
                          ("snapshot-worker", production_snapshot_worker),
                          ("export-core", production_export_core),
                          ("export-core-watch", production_export_core_watch)):
        command = production_commands.add_parser(name)
        command.add_argument("config", type=Path,
                             nargs="+" if name in {"export-core", "export-core-watch", "dashboard"} else None,
                             help="production job YAML with Chinese comments")
        if name in {"export-core", "export-core-watch"}:
            command.add_argument("--audit-report", type=Path,
                                 required=name == "export-core-watch",
                                 help="completed full cloud audit required for prod coverage")
            command.add_argument("--rules-file", type=Path,
                                 help="immutable local weak-label rules JSON; does not change cloud rules")
        if name == "export-core-watch":
            command.add_argument(
                "--chain-unit", default="imu-motion-production-chain.service")
            command.add_argument("--poll-interval-s", type=int, default=60)
        if name == "run":
            command.add_argument("--json", action="store_true",
                                 help="emit machine-readable progress events")
            command.add_argument("--upload-profile", type=Path,
                                 help="bind a durable operational profile on first run")
        if name == "upload-configure":
            command.add_argument("--profile", type=Path, required=True)
        if name == "watch":
            command.add_argument("--after", type=int, default=0)
            command.add_argument("--once", action="store_true")
        if name == "serve":
            command.add_argument("--port", type=int, default=8890)
        if name == "dashboard":
            command.add_argument("--port", type=int, default=8891)
            command.add_argument("--audit-report", type=Path)
            command.add_argument("--unit", action="append",
                                 help="systemd user unit to show; repeat as needed")
        if name == "snapshot-worker":
            command.add_argument("--poll-interval-s", type=float, default=30.0)
            command.add_argument("--once", action="store_true")
            command.add_argument("--retry-failed", metavar="SNAPSHOT_ID",
                                 help="explicitly retry one failed frozen request")
        command.set_defaults(func=handler)

    library = groups.add_parser("library", help="acquire and audit source datasets")
    library_commands = library.add_subparsers(dest="command", required=True)
    acquire = library_commands.add_parser("acquire")
    acquire.add_argument("--root", type=Path, required=True)
    acquire.add_argument("--manifest", type=Path, required=True)
    acquire.add_argument("--workers", type=int, default=3)
    acquire.add_argument("--reserve-gib", type=int, default=100)
    acquire.set_defaults(func=library_acquire)
    status = library_commands.add_parser("status")
    status.add_argument("--root", type=Path, required=True)
    status_input = status.add_mutually_exclusive_group()
    status_input.add_argument("--manifest", type=Path)
    status_input.add_argument("--inventory", type=Path)
    status.add_argument("--verify-content", action="store_true")
    status.set_defaults(func=library_status)
    prepare = library_commands.add_parser("prepare-sources")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--library-root", type=Path, required=True)
    prepare.add_argument("--catalog", type=Path, default=root / "configs/source-catalog-v1.json")
    prepare.set_defaults(func=library_prepare)

    pipeline = groups.add_parser("pipeline", help="validate or run the kinematic production pipeline")
    pipeline_commands = pipeline.add_subparsers(dest="command", required=True)
    pipeline_plan = pipeline_commands.add_parser("validate")
    pipeline_plan.add_argument("plan", type=Path)
    pipeline_plan.set_defaults(func=pipeline_validate)
    make_amass = pipeline_commands.add_parser("make-amass-plan")
    make_amass.add_argument("output", type=Path)
    make_amass.add_argument("--library-root", type=Path, required=True)
    make_amass.add_argument("--study-id", required=True)
    make_amass.add_argument("--source-dataset", required=True)
    make_amass.add_argument("--amass-archive", type=Path, required=True)
    make_amass.add_argument("--smplh-archive", type=Path,
                            default=Path("models/smplh/original/smplh.tar.xz"))
    make_amass.add_argument("--dmpl-archive", type=Path,
                            default=Path("models/dmpl/original/dmpls.tar.xz"))
    make_amass.add_argument("--babel-archive", type=Path)
    make_amass.add_argument("--layout", type=Path,
                            default=Path("configs/sensors/layouts/chest-1.json"))
    make_amass.add_argument("--profile", type=Path,
                            default=Path("configs/sensors/profiles/ideal-25hz-v3.json"))
    make_amass.add_argument(
        '--adapter', choices=['amass-smplh-g-v1', 'stageii-smplh-v1'],
        default='amass-smplh-g-v1')
    make_amass.set_defaults(func=pipeline_make_amass_plan)
    make_catalog = pipeline_commands.add_parser("make-amass-catalog")
    make_catalog.add_argument("output", type=Path)
    make_catalog.add_argument("--library-root", type=Path, required=True)
    make_catalog.add_argument("--amass-directory", type=Path,
                              default=Path("datasets/amass/smplh-g"))
    make_catalog.add_argument("--smplh-archive", type=Path,
                              default=Path("models/smplh/original/smplh.tar.xz"))
    make_catalog.add_argument("--dmpl-archive", type=Path,
                              default=Path("models/dmpl/original/dmpls.tar.xz"))
    make_catalog.add_argument("--babel-archive", type=Path)
    make_catalog.add_argument("--layout", type=Path,
                              default=Path("configs/sensors/layouts/chest-1.json"))
    make_catalog.add_argument("--profile", type=Path,
                              default=Path("configs/sensors/profiles/ideal-25hz-v3.json"))
    make_catalog.add_argument('--include-source', action='append', default=[])
    make_catalog.add_argument('--adapter', action='append', default=[],
                              metavar='SOURCE=ADAPTER_ID')
    make_catalog.set_defaults(func=pipeline_make_amass_catalog)
    run_catalog = pipeline_commands.add_parser("run-amass-catalog")
    run_catalog.add_argument("catalog", type=Path)
    run_catalog.add_argument("--library-root", type=Path, required=True)
    run_catalog.add_argument("--workspace", type=Path, default=root)
    run_catalog.add_argument("--checkout", type=Path, default=root)
    run_catalog.add_argument("--output", type=Path, required=True)
    run_catalog.add_argument("--through", choices=["convert", "sensors", "review"],
                             default="review")
    run_catalog.add_argument("--skip-source", action="append", default=[])
    run_catalog.add_argument("--reuse-source", action="append", default=[],
                             metavar="SOURCE=STUDY_PATH")
    run_catalog.add_argument("--workers", type=int, default=1)
    run_catalog.add_argument("--sensor-workers", type=int, default=1,
                             help="clip-level sensor processes per source (opt-in)")
    run_catalog.set_defaults(func=pipeline_run_amass_catalog)
    audit_catalog = pipeline_commands.add_parser("audit-amass-batch")
    audit_catalog.add_argument("catalog", type=Path)
    audit_catalog.add_argument("batch", type=Path)
    audit_catalog.set_defaults(func=pipeline_audit_amass_batch)
    machine_corpus = pipeline_commands.add_parser("build-machine-corpus")
    machine_corpus.add_argument("catalog", type=Path)
    machine_corpus.add_argument("batch", type=Path)
    machine_corpus.add_argument("output", type=Path)
    machine_corpus.add_argument(
        "--policy", type=Path,
        default=root / "configs/quality/machine-review-v1.json")
    machine_corpus.set_defaults(func=pipeline_build_machine_corpus)
    merge_corpora = pipeline_commands.add_parser("merge-machine-corpora")
    merge_corpora.add_argument("output", type=Path)
    merge_corpora.add_argument("--corpus", type=Path, action="append",
                               required=True)
    merge_corpora.set_defaults(func=pipeline_merge_machine_corpora)
    pipeline_execute = pipeline_commands.add_parser("run")
    pipeline_execute.add_argument("plan", type=Path)
    pipeline_execute.add_argument("--library-root", type=Path, required=True)
    pipeline_execute.add_argument("--workspace", type=Path, default=root,
                                  help=argparse.SUPPRESS)
    pipeline_execute.add_argument("--checkout", type=Path, default=root)
    pipeline_execute.add_argument("--output", type=Path, required=True)
    pipeline_execute.add_argument("--source-artifact", type=Path)
    pipeline_execute.add_argument(
        "--through", choices=["convert", "sensors", "review"], default="review")
    pipeline_execute.add_argument("--sensor-workers", type=int, default=1,
                                  help="clip-level sensor processes (opt-in)")
    pipeline_execute.set_defaults(func=pipeline_run)

    sensors = groups.add_parser("sensors", help="derive named virtual IMU layouts")
    sensor_commands = sensors.add_subparsers(dest="command", required=True)
    derive = sensor_commands.add_parser("derive")
    derive.add_argument("motion", type=Path); derive.add_argument("output", type=Path)
    derive.add_argument("--model-archive", type=Path, required=True)
    derive.add_argument("--layout", type=Path, required=True)
    derive.add_argument("--profile", type=Path, required=True)
    derive.add_argument("--selection", type=Path)
    derive.set_defaults(func=sensors_derive)
    convergence = sensor_commands.add_parser("convergence")
    convergence.add_argument("motion", type=Path)
    convergence.add_argument("--model-archive", type=Path, required=True)
    convergence.add_argument("--layout", type=Path, required=True)
    convergence.add_argument("--profile", type=Path, required=True)
    convergence.add_argument("--selection", type=Path)
    convergence.set_defaults(func=sensors_convergence)
    calibrate_sensor = sensor_commands.add_parser("calibrate")
    calibrate_sensor.add_argument("ideal", type=Path)
    calibrate_sensor.add_argument("output", type=Path)
    calibrate_sensor.add_argument("--profile", type=Path, required=True)
    calibrate_sensor.add_argument("--seed", type=int, required=True)
    calibrate_sensor.set_defaults(func=sensors_calibrate)

    review = groups.add_parser("review", help="validate or serve kinematic review bundles")
    review_commands = review.add_subparsers(dest="command", required=True)
    review_build_parser = review_commands.add_parser("build")
    review_build_parser.add_argument("motion", type=Path)
    review_build_parser.add_argument("sensors", type=Path)
    review_build_parser.add_argument("output", type=Path)
    review_build_parser.add_argument("--model-archive", type=Path, required=True)
    review_build_parser.add_argument("--layout", type=Path, required=True)
    review_build_parser.add_argument("--profile", type=Path, required=True)
    review_build_parser.add_argument("--selection", type=Path)
    review_build_parser.set_defaults(func=review_build)
    review_check = review_commands.add_parser("validate")
    review_check.add_argument("path", type=Path)
    review_check.set_defaults(func=review_validate)
    review_server = review_commands.add_parser("serve")
    review_server.add_argument("path", type=Path)
    review_server.add_argument("--host", default="127.0.0.1")
    review_server.add_argument("--port", type=int, default=8765)
    review_server.add_argument("--open", action="store_true")
    review_server.set_defaults(func=review_serve)
    review_video = review_commands.add_parser("render")
    review_video.add_argument("motion", type=Path)
    review_video.add_argument("sensors", type=Path)
    review_video.add_argument("output", type=Path)
    review_video.add_argument("--model-archive", type=Path, required=True)
    review_video.add_argument("--dmpl-archive", type=Path)
    review_video.add_argument("--layout", type=Path, required=True)
    review_video.add_argument("--selection", type=Path)
    review_video.add_argument("--blender", type=Path, required=True)
    review_video.add_argument("--fps", type=int, default=30)
    review_video.add_argument("--width", type=int, default=1280)
    review_video.add_argument("--height", type=int, default=720)
    review_video.set_defaults(func=review_render)
    review_flag = review_commands.add_parser(
        "flag-quality", help="append a non-blocking source-fit observation")
    review_flag.add_argument("path", type=Path)
    review_flag.add_argument("--reviewer", required=True)
    review_flag.add_argument("--reason", required=True)
    review_flag.add_argument("--start-frame", type=int, required=True)
    review_flag.add_argument("--stop-frame", type=int, required=True)
    review_flag.add_argument("--joint", action="append", required=True)
    review_flag.set_defaults(func=review_flag_quality)

    documents = groups.add_parser("docs", help="check the active document set")
    document_commands = documents.add_subparsers(dest="command", required=True)
    check_parser = document_commands.add_parser("check")
    check_parser.add_argument("--root", type=Path, default=root)
    check_parser.set_defaults(func=docs_check)

    validation = groups.add_parser(
        "validation", help="run read-only scientific reality baselines")
    validation_commands = validation.add_subparsers(
        dest="command", required=True)
    imucoco_audit = validation_commands.add_parser("imucoco-audit")
    imucoco_audit.add_argument("archive", type=Path)
    imucoco_audit.set_defaults(func=validation_imucoco_audit)
    imucoco_benchmark = validation_commands.add_parser("imucoco-benchmark")
    imucoco_benchmark.add_argument("archive", type=Path)
    imucoco_benchmark.add_argument("output", type=Path)
    imucoco_benchmark.add_argument("--all-takes", action="store_true",
                                   help="benchmark every complete pose/IMU/calibration triplet")
    imucoco_benchmark.set_defaults(func=validation_imucoco_benchmark)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
