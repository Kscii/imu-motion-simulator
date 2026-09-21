"""Resumable execution of every plan-ready source in an AMASS catalog."""
from __future__ import annotations

from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import json
import os
from pathlib import Path
import shutil

from ..contracts.common import sha256_file
from .plan import load_plan


def _atomic_json(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')
    temporary.replace(path)


def _summary(report):
    return {
        key: report.get(key) for key in (
            'study_id', 'mode', 'stage', 'clips_planned', 'clips_ready',
            'automatic_qa_complete', 'automatic_qa_passed', 'human_review',
            'reviewed_clips', 'accepted_clips')
        if key in report
    } | {'human_review_queue_size': len(report.get('human_review_queue', []))}


def _execution_identity(plan):
    """Return the source/motion identity that an existing run must share."""
    return {
        'inputs': plan['inputs'],
        'source_adapter': plan.get('source_adapter'),
        'layout': plan['sensor']['layout'],
        'clips': [{key: clip[key] for key in (
            'id', 'source_dataset', 'source_member', 'expected_gender',
            'expected_frames', 'frame_range')}
                  for clip in plan['clips']]}


def _reuse_evidence(source, catalog_plan, study):
    study = Path(study).resolve()
    prior_plan = study / 'plan.json'
    prior_report = study / 'run/study-report.json'
    if not prior_plan.is_file() or not prior_report.is_file():
        raise ValueError('Reuse study lacks plan/report: ' + source)
    current = load_plan(catalog_plan)
    prior = load_plan(prior_plan)
    if _execution_identity(current) != _execution_identity(prior):
        raise ValueError('Reuse study has different source motion identity: ' + source)
    report = json.loads(prior_report.read_text())
    if report.get('clips_planned') != len(current['clips']) \
            or report.get('clips_ready') != len(current['clips']):
        raise ValueError('Reuse study is incomplete: ' + source)
    return {
        'status': 'reused-existing-study',
        'evidence_path': str(study),
        'evidence_plan_sha256': sha256_file(prior_plan),
        'evidence_report_sha256': sha256_file(prior_report),
        'evidence_profile': prior['sensor']['profile'],
        'catalog_profile': current['sensor']['profile'],
        'report': _summary(report)}


def _execute_source(request):
    """Process-pool entry point; source outputs remain mutually independent."""
    from .run import run_pipeline
    (source, plan, library_root, workspace, checkout, study, through,
     sensor_workers) = request
    options = {'through': through}
    if sensor_workers != 1:
        options['sensor_workers'] = sensor_workers
    report = run_pipeline(
        plan, library_root, workspace, checkout, Path(study) / 'run',
        **options)
    return source, _summary(report)


def run_amass_catalog(catalog_directory, output, *, library_root, checkout,
                      workspace, through='review', skip_sources=(),
                      reuse_sources=None, workers=1, sensor_workers=1):
    """Run all native plans, preserving per-source failures and resumable state."""
    if type(workers) is not int or not 1 <= workers <= 32:
        raise ValueError('AMASS batch workers must be in [1, 32]')
    worker_limit = min(32, os.cpu_count() or 1)
    if type(sensor_workers) is not int or not 1 <= sensor_workers <= worker_limit:
        raise ValueError('AMASS batch sensor_workers exceeds available CPUs')
    if sensor_workers > 1 and workers * sensor_workers > worker_limit:
        raise ValueError('Source and sensor workers exceed available CPUs')
    catalog_directory = Path(catalog_directory).resolve()
    output = Path(output).resolve()
    library_root = Path(library_root).resolve()
    checkout = Path(checkout).resolve()
    workspace = Path(workspace).resolve()
    catalog_path = catalog_directory / 'catalog.json'
    catalog = json.loads(catalog_path.read_text())
    if catalog.get('schema') not in {
            'imu_motion_simulator.amass_catalog.v1',
            'imu_motion_simulator.amass_catalog.v2'}:
        raise ValueError('Unsupported AMASS catalog')
    ready = {item['source_dataset']: item for item in catalog['sources']
             if item['status'] == 'plan-ready'}
    skip_sources = set(skip_sources)
    reuse_sources = {source: str(Path(path).resolve())
                     for source, path in (reuse_sources or {}).items()}
    if skip_sources & set(reuse_sources):
        raise ValueError('A source cannot be both skipped and reused')
    unknown = (skip_sources | set(reuse_sources)) - set(ready)
    if unknown:
        raise ValueError('Requested source is not plan-ready: '
                         + ', '.join(sorted(unknown)))
    config = {
        'catalog_sha256': sha256_file(catalog_path),
        'library_root': str(library_root),
        'checkout': str(checkout),
        'workspace': str(workspace),
        'through': through,
        'skip_sources': sorted(skip_sources),
        'reuse_sources': dict(sorted(reuse_sources.items()))}
    if sensor_workers != 1:
        config['sensor_workers'] = sensor_workers
    state_path = output / 'batch-state.json'
    if output.exists():
        if not state_path.is_file():
            raise FileExistsError('Batch output exists without resumable state')
        state = json.loads(state_path.read_text())
        if state.get('schema') != 'imu_motion_simulator.amass_batch_state.v1' \
                or state.get('config') != config:
            raise ValueError('Existing AMASS batch has different inputs')
    else:
        output.mkdir(parents=True)
        state = {
            'schema': 'imu_motion_simulator.amass_batch_state.v1',
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'config': config, 'sources': {}}
        _atomic_json(state_path, state)
    sources_root = output / 'sources'; sources_root.mkdir(exist_ok=True)
    pending = []
    for source, item in ready.items():
        previous = state['sources'].get(source, {})
        catalog_plan = catalog_directory / item['plan']
        if sha256_file(catalog_plan) != item['plan_sha256']:
            raise ValueError('Catalog plan hash mismatch: ' + source)
        if source in reuse_sources:
            state['sources'][source] = {
                **_reuse_evidence(source, catalog_plan, reuse_sources[source]),
                'plan_sha256': item['plan_sha256']}
            _atomic_json(state_path, state)
            continue
        if source in skip_sources:
            state['sources'][source] = {
                'status': 'skipped-by-request',
                'plan_sha256': item['plan_sha256']}
            _atomic_json(state_path, state)
            continue
        study = sources_root / source
        if previous.get('status') == 'complete' \
                and (study / 'run/study-report.json').is_file():
            continue
        study.mkdir(exist_ok=True)
        plan = study / 'plan.json'
        if plan.exists():
            if sha256_file(plan) != item['plan_sha256']:
                raise ValueError('Batch plan changed: ' + source)
        else:
            shutil.copyfile(catalog_plan, plan)
        state['sources'][source] = {
            'status': 'running', 'plan_sha256': item['plan_sha256']}
        _atomic_json(state_path, state)
        pending.append((source, plan, library_root, workspace, checkout,
                        study, through, sensor_workers))

    def record(source, *, report=None, error=None):
        item = ready[source]
        if error is None and not (
                sources_root / source / 'run/study-report.json').is_file():
            error = RuntimeError(
                'Pipeline returned without persistent study report: ' + source)
        if error is not None:
            state['sources'][source] = {
                'status': 'error', 'plan_sha256': item['plan_sha256'],
                'error_type': type(error).__name__, 'error': str(error)}
        else:
            state['sources'][source] = {
                'status': 'complete', 'plan_sha256': item['plan_sha256'],
                'report': report}
        _atomic_json(state_path, state)

    if workers == 1:
        for request in pending:
            source = request[0]
            try:
                _, report = _execute_source(request)
            except Exception as error:
                record(source, error=error)
            else:
                record(source, report=report)
    elif pending:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_execute_source, request): request[0]
                       for request in pending}
            for future in as_completed(futures):
                source = futures[future]
                try:
                    _, report = future.result()
                except Exception as error:
                    record(source, error=error)
                else:
                    record(source, report=report)
    counts = {}
    for value in state['sources'].values():
        counts[value['status']] = counts.get(value['status'], 0) + 1
    result = {
        'output': str(output), 'catalog_sha256': config['catalog_sha256'],
        'plan_ready_sources': len(ready), 'counts': counts,
        'errors': sorted(source for source, value in state['sources'].items()
                         if value['status'] == 'error')}
    _atomic_json(output / 'batch-report.json', result)
    return result


def audit_amass_batch(catalog_directory, batch_directory):
    """Audit batch state against source reports and return clip-level totals."""
    catalog_directory = Path(catalog_directory).resolve()
    batch_directory = Path(batch_directory).resolve()
    catalog_path = catalog_directory / 'catalog.json'
    state_path = batch_directory / 'batch-state.json'
    catalog = json.loads(catalog_path.read_text())
    state = json.loads(state_path.read_text())
    if catalog.get('schema') not in {
            'imu_motion_simulator.amass_catalog.v1',
            'imu_motion_simulator.amass_catalog.v2'} \
            or state.get('schema') \
            != 'imu_motion_simulator.amass_batch_state.v1':
        raise ValueError('Unsupported AMASS catalog or batch state')
    if state['config']['catalog_sha256'] != sha256_file(catalog_path):
        raise ValueError('Batch state does not match catalog')
    ready = {item['source_dataset']: item for item in catalog['sources']
             if item['status'] == 'plan-ready'}
    rows, warnings, failed_checks = [], Counter(), Counter()
    totals = Counter()
    for source, item in ready.items():
        record = state['sources'].get(source, {})
        status = record.get('status', 'missing')
        report_path = None
        if status == 'complete':
            report_path = (batch_directory / 'sources' / source
                           / 'run/study-report.json')
        elif status == 'reused-existing-study':
            report_path = Path(record['evidence_path']) / 'run/study-report.json'
            if sha256_file(report_path) != record['evidence_report_sha256']:
                raise ValueError('Reused report changed: ' + source)
        if report_path is None:
            rows.append({'source_dataset': source, 'status': status,
                         'clips': item['clips']})
            totals[status] += 1
            continue
        report = json.loads(report_path.read_text())
        clips = report.get('clips', {})
        if report.get('clips_planned') != item['clips'] \
                or report.get('clips_ready') != item['clips'] \
                or len(clips) != item['clips']:
            raise ValueError('Source report clip count differs: ' + source)
        qa = [value.get('automatic_qa') for value in clips.values()]
        if any(value is None for value in qa):
            raise ValueError('Source report lacks clip QA: ' + source)
        passed = sum(value['passed'] for value in qa)
        warning_clips = sum(bool(value['warnings']) for value in qa)
        source_warnings = Counter(
            warning for value in qa for warning in value['warnings'])
        source_failed = Counter(
            name for value in qa for name, check in value['checks'].items()
            if not check['pass'])
        warnings.update(source_warnings); failed_checks.update(source_failed)
        totals['sources_audited'] += 1
        totals['clips'] += len(qa); totals['clips_passed'] += passed
        totals['clips_failed'] += len(qa) - passed
        totals['warning_clips'] += warning_clips
        totals['human_review_queue'] += len(report['human_review_queue'])
        rows.append({
            'source_dataset': source, 'status': status, 'clips': len(qa),
            'clips_passed': passed, 'clips_failed': len(qa) - passed,
            'warning_clips': warning_clips,
            'human_review_queue': len(report['human_review_queue']),
            'warnings': dict(sorted(source_warnings.items())),
            'failed_checks': dict(sorted(source_failed.items())),
            'report': str(report_path),
            'report_sha256': sha256_file(report_path)})
    finished = totals['sources_audited'] == len(ready)
    if finished and totals['clips'] != catalog['clips']:
        raise ValueError('Audited clip total differs from catalog')
    return {
        'schema': 'imu_motion_simulator.amass_batch_audit.v1',
        'catalog': str(catalog_path), 'catalog_sha256': sha256_file(catalog_path),
        'batch': str(batch_directory), 'production_complete': finished,
        'plan_ready_sources': len(ready),
        'adapter_required_sources': catalog['adapter_required_archives'],
        'catalog_clips': catalog['clips'],
        'catalog_source_duration_hours': catalog['source_duration_hours'],
        'totals': dict(sorted(totals.items())),
        'warnings': dict(sorted(warnings.items())),
        'failed_checks': dict(sorted(failed_checks.items())),
        'sources': rows}
