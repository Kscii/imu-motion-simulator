"""Resumable pure-kinematic production pipeline."""
from __future__ import annotations

from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import tarfile

from ..contracts.common import sha256_file
from ..contracts.internal import read_internal, validate_internal
from ..motion.selection import validate_selection, write_selection
from ..motion.kinematics import load_model_member
from ..motion.smplh import NATIVE_ADAPTER, decode_amass_member
from ..review.bundle import automatic_qa, build_bundle, validate_bundle
from ..review.bundle import review_recipe_sha256
from ..sensors.convergence import (convergence_recipe_sha256,
                                   convergence_report,
                                   upgrade_legacy_convergence)
from ..sensors.derive import derive_ideal, sensor_recipe_sha256
from ..sensors.layout import load_layout, load_profile
from .plan import resolve_inside


_WORKER_MODELS = {}


def _sensor_task(request):
    """Produce one independent clip; the source process owns study state."""
    (clip_id, motion, selection, sensors, smplh, layout_path, profile_path,
     model_archive_hash, convergence, prior_recipe, prior_sensor_recipe,
     sensor_recipe, convergence_recipe) = request
    _, motion_metadata, _ = read_internal(motion, 'motion')
    gender = motion_metadata['kind_metadata']['source_gender']
    key = (str(smplh), model_archive_hash, gender)
    if key not in _WORKER_MODELS:
        _WORKER_MODELS[key] = load_model_member(smplh, gender)
    model = _WORKER_MODELS[key]
    if not sensors.exists():
        derive_ideal(motion, smplh, layout_path, profile_path, sensors,
                     selection=selection, model=model,
                     model_archive_sha256=model_archive_hash)
    description = validate_internal(sensors, 'sensors')
    if prior_recipe != convergence_recipe:
        convergence = (upgrade_legacy_convergence(convergence)
                       if prior_sensor_recipe == sensor_recipe else None)
    if convergence is None:
        convergence = convergence_report(
            motion, smplh, layout_path, profile_path, selection=selection,
            model=model)
    qa = automatic_qa(motion, sensors, selection=selection,
                      convergence=convergence)
    return clip_id, {
        'status': 'sensors-ready', 'sensors': str(sensors),
        'sensors_sha256': sha256_file(sensors),
        'sensor_samples': description['samples'],
        'convergence': convergence, 'automatic_qa': qa,
        'sensor_recipe_sha256': sensor_recipe,
        'convergence_recipe_sha256': convergence_recipe}


def _atomic_json(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')
    temporary.replace(path)


def _relative(path, root):
    return str(Path(path).resolve().relative_to(Path(root).resolve()))


def _token(*values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode()); digest.update(b'\0')
    return digest.hexdigest()[:12]


def _checkpoint(index, total, state_path, state, *, every=25):
    if (index + 1) % every == 0 or index + 1 == total:
        _atomic_json(state_path, state)


def _state(output, plan_path, plan):
    output = Path(output); state_path = output / 'study-state.json'
    plan_hash = sha256_file(plan_path)
    if output.exists():
        if not state_path.is_file():
            raise FileExistsError('Output exists without kinematic state')
        value = json.loads(state_path.read_text())
        if value.get('schema') != 'imu_motion_simulator.kinematic_state.v1' \
                or value.get('plan_sha256') != plan_hash:
            raise ValueError('Existing output differs from immutable plan')
        return value, state_path
    output.mkdir(parents=True)
    value = {'schema': 'imu_motion_simulator.kinematic_state.v1',
             'study_id': plan['study_id'], 'plan_sha256': plan_hash,
             'created_at_utc': datetime.now(timezone.utc).isoformat(),
             'stage': 'created', 'clips': {}}
    _atomic_json(state_path, value)
    return value, state_path


def run_kinematic_pipeline(plan_path, plan, library_root, checkout, output, *,
                           source_artifact=None, through='review',
                           sensor_workers=1):
    if through not in {'convert', 'sensors', 'review'}:
        raise ValueError('Unknown kinematic pipeline stage')
    worker_limit = min(32, os.cpu_count() or 1)
    if type(sensor_workers) is not int or not 1 <= sensor_workers <= worker_limit:
        raise ValueError('Kinematic sensor_workers exceeds available CPUs')
    library_root, checkout, output = map(
        lambda path: Path(path).resolve(), (library_root, checkout, output))
    amass = resolve_inside(library_root, plan['inputs']['amass_archive'])
    smplh = resolve_inside(library_root, plan['inputs']['smplh_archive'])
    dmpl = resolve_inside(library_root, plan['inputs']['dmpl_archive'])
    layout_path = resolve_inside(checkout, plan['sensor']['layout'])
    profile_path = resolve_inside(checkout, plan['sensor']['profile'])
    adapter_id = plan.get('source_adapter', {}).get('id', NATIVE_ADAPTER)
    required_paths = [amass, smplh, layout_path, profile_path]
    if adapter_id == NATIVE_ADAPTER:
        required_paths.append(dmpl)
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    layout = load_layout(layout_path); load_profile(profile_path)
    state, state_path = _state(output, plan_path, plan)
    for name in ('motions', 'selections', 'sensors', 'reviews'):
        (output / name).mkdir(exist_ok=True)
    archive_hash = sha256_file(amass)
    model_archive_hash = sha256_file(smplh)
    layout_hash = sha256_file(layout_path)
    profile_hash = sha256_file(profile_path)
    decode_cache = {}
    missing = {
        clip['source_member']: (
            clip, output / 'motions' / f"{clip['id']}.motion.h5")
        for clip in plan['clips']
        if not (output / 'motions' / f"{clip['id']}.motion.h5").exists()
    }
    if missing:
        state['stage'] = 'streaming-source'; _atomic_json(state_path, state)
        with tarfile.open(amass, 'r|bz2') as archive:
            for member in archive:
                target = missing.get(member.name)
                if target is None or not member.isfile():
                    continue
                raw = archive.extractfile(member).read()
                clip, motion = target
                decode_amass_member(
                    amass, smplh, clip['source_member'], motion,
                    source_artifact=source_artifact,
                    source_dataset=clip['source_dataset'],
                    original_archive_sha256=archive_hash,
                    source_bytes=raw, model_cache=decode_cache,
                    adapter_id=adapter_id)
                del missing[member.name]
                if not missing:
                    break
        if missing:
            names = ', '.join(sorted(missing)[:3])
            raise ValueError(f'AMASS members missing from source archive: {names}')
    total = len(plan['clips'])
    for index, clip in enumerate(plan['clips']):
        record = state['clips'].setdefault(clip['id'], {})
        motion = output / 'motions' / f"{clip['id']}.motion.h5"
        if not motion.exists():
            raise RuntimeError('Streamed AMASS conversion did not create ' + str(motion))
        description, metadata, _ = read_internal(motion, 'motion')
        info = metadata['kind_metadata']
        if description['frames'] != clip['expected_frames'] \
                or info['source_gender'] != clip['expected_gender'] \
                or info['source_member'] != clip['source_member']:
            raise ValueError('Canonical motion differs from plan: ' + clip['id'])
        selection = output / 'selections' / f"{clip['id']}.selection.json"
        if not selection.exists():
            write_selection(selection, motion,
                            start_frame=clip['frame_range'][0],
                            stop_frame=clip['frame_range'][1],
                            label_candidates=clip['label_candidates'])
        selection_value = validate_selection(selection, motion)
        record.update(
            status='converted', motion=_relative(motion, output),
            motion_sha256=sha256_file(motion),
            selection=_relative(selection, output),
            selection_sha256=sha256_file(selection),
            selection_id=selection_value['selection_id'])
        state['stage'] = 'converting'
        _checkpoint(index, total, state_path, state)
    state['stage'] = 'converted'; _atomic_json(state_path, state)
    if through == 'convert':
        report = _report(plan, state, layout['layout_id'], through)
        _atomic_json(output / 'study-report.json', report)
        return report

    runtime_models = {gender: cached[1]
                      for gender, cached in decode_cache.items()}

    def runtime_model(motion_metadata):
        gender = motion_metadata['kind_metadata']['source_gender']
        if gender not in runtime_models:
            runtime_models[gender] = load_model_member(smplh, gender)
        return runtime_models[gender]

    sensor_recipe = sensor_recipe_sha256()
    convergence_recipe = convergence_recipe_sha256()
    if sensor_workers > 1:
        tasks = {}
        for clip in plan['clips']:
            record = state['clips'][clip['id']]
            motion = output / record['motion']
            selection = output / record['selection']
            identity = _token(record['motion_sha256'], record['selection_sha256'],
                              layout_hash, profile_hash, sensor_recipe)
            sensors = output / 'sensors' / f"{clip['id']}-{layout['layout_id']}-{identity}.sensors.h5"
            tasks[clip['id']] = (
                clip['id'], motion, selection, sensors, smplh, layout_path,
                profile_path, model_archive_hash, record.get('convergence'),
                record.get('convergence_recipe_sha256'),
                record.get('sensor_recipe_sha256'), sensor_recipe,
                convergence_recipe)
        with ProcessPoolExecutor(max_workers=sensor_workers) as pool:
            futures = {pool.submit(_sensor_task, task): clip_id
                       for clip_id, task in tasks.items()}
            for index, future in enumerate(as_completed(futures)):
                clip_id, result = future.result()
                result['sensors'] = _relative(result['sensors'], output)
                state['clips'][clip_id].update(result)
                state['stage'] = 'deriving-sensors'
                _checkpoint(index, total, state_path, state)
    else:
        for index, clip in enumerate(plan['clips']):
            record = state['clips'][clip['id']]
            motion = output / record['motion']; selection = output / record['selection']
            _, motion_metadata, _ = read_internal(motion, 'motion')
            model = runtime_model(motion_metadata)
            identity = _token(record['motion_sha256'], record['selection_sha256'],
                              layout_hash, profile_hash, sensor_recipe)
            sensors = output / 'sensors' / f"{clip['id']}-{layout['layout_id']}-{identity}.sensors.h5"
            if not sensors.exists():
                derive_ideal(motion, smplh, layout_path, profile_path, sensors,
                             selection=selection, model=model,
                             model_archive_sha256=model_archive_hash)
            sensor_description = validate_internal(sensors, 'sensors')
            convergence = (record.get('convergence')
                           if record.get('convergence_recipe_sha256')
                           == convergence_recipe else None)
            if convergence is None \
                    and record.get('sensor_recipe_sha256') == sensor_recipe:
                convergence = upgrade_legacy_convergence(record.get('convergence'))
            if convergence is None:
                convergence = convergence_report(
                    motion, smplh, layout_path, profile_path, selection=selection,
                    model=model)
            qa = automatic_qa(
                motion, sensors, selection=selection, convergence=convergence)
            record.update(status='sensors-ready', sensors=_relative(sensors, output),
                          sensors_sha256=sha256_file(sensors),
                          sensor_samples=sensor_description['samples'],
                          convergence=convergence, automatic_qa=qa,
                          sensor_recipe_sha256=sensor_recipe,
                          convergence_recipe_sha256=convergence_recipe)
            state['stage'] = 'deriving-sensors'
            _checkpoint(index, total, state_path, state)
    state['stage'] = 'sensors-ready'; _atomic_json(state_path, state)
    if through == 'sensors':
        report = _report(plan, state, layout['layout_id'], through)
        _atomic_json(output / 'study-report.json', report)
        return report

    review_recipe = review_recipe_sha256()
    for index, clip in enumerate(plan['clips']):
        record = state['clips'][clip['id']]
        motion = output / record['motion']; selection = output / record['selection']
        sensors = output / record['sensors']
        _, motion_metadata, _ = read_internal(motion, 'motion')
        model = runtime_model(motion_metadata)
        identity = _token(record['motion_sha256'], record['sensors_sha256'],
                          record['selection_sha256'], review_recipe)
        review = output / 'reviews' / f"{clip['id']}-{identity}.review"
        if not review.exists():
            build_bundle(motion, sensors, smplh, layout_path, review,
                         selection=selection, convergence=record['convergence'],
                         model=model)
        validation = validate_bundle(review)
        qa = json.loads((review / 'qa.json').read_text())
        record.update(status='review-ready', review=_relative(review, output),
                      review_validation=validation, automatic_qa=qa,
                      review_recipe_sha256=review_recipe)
        state['stage'] = 'building-reviews'
        _checkpoint(index, total, state_path, state)
    state['stage'] = 'review-ready'; _atomic_json(state_path, state)
    report = _report(plan, state, layout['layout_id'], through)
    _atomic_json(output / 'study-report.json', report)
    return report


def _report(plan, state, layout_id, stage):
    records = list(state['clips'].values())
    qa = [record.get('automatic_qa') for record in records]
    queue = _human_review_queue(plan, state)
    decisions = [record.get('review_validation', {}).get('decision', 'unreviewed')
                 for record in records]
    accepted = sum(decision == 'accepted' for decision in decisions)
    reviewed = sum(decision in ('accepted', 'rejected') for decision in decisions)
    human_state = ('accepted' if records and accepted == len(records)
                   else 'partial' if reviewed else 'unreviewed')
    return {
        'schema': 'imu_motion_simulator.kinematic_study_report.v1',
        'study_id': plan['study_id'], 'mode': 'kinematic', 'stage': stage,
        'clips_planned': len(plan['clips']), 'clips_ready': len(records),
        'sensor_layout': layout_id,
        'automatic_qa_complete': all(item is not None for item in qa),
        'automatic_qa_passed': bool(qa) and all(
            item is not None and item['passed'] for item in qa),
        'human_review': human_state, 'reviewed_clips': reviewed,
        'accepted_clips': accepted, 'human_review_queue': queue,
        'clips': state['clips']}


def _human_review_queue(plan, state):
    """Return a deterministic exception and stratified human-review queue."""
    pending, ordinary = [], {}
    for clip in plan['clips']:
        record = state['clips'].get(clip['id'], {})
        decision = record.get('review_validation', {}).get(
            'decision', 'unreviewed')
        if decision in ('accepted', 'rejected'):
            continue
        qa = record.get('automatic_qa') or {}
        reasons = []
        if qa and not qa.get('passed'):
            reasons.append('automatic-qa-failed')
        if qa.get('warnings'):
            reasons.append('automatic-warning')
        candidates = clip['label_candidates']
        recording_codes = sorted({item.get('code') for item in candidates
                                  if isinstance(item, dict) and item.get('code')
                                  and item.get('kind') != 'temporal-candidate'})
        if not recording_codes:
            reasons.append('label-unresolved')
        elif len(recording_codes) > 1:
            reasons.append('label-ambiguous')
        risk = float(qa.get('risk_score', 1.))
        row = {'clip_id': clip['id'], 'source_member': clip['source_member'],
               'risk_score': risk, 'reasons': reasons,
               'review': record.get('review')}
        if reasons:
            pending.append(row); continue
        subject = Path(clip['source_member']).parent.name
        key = (clip['source_dataset'], subject, recording_codes[0])
        previous = ordinary.get(key)
        if previous is None or (risk, clip['id']) > (
                previous['risk_score'], previous['clip_id']):
            ordinary[key] = row
    for row in ordinary.values():
        row['reasons'] = ['stratified-sample']
        pending.append(row)
    return sorted(pending, key=lambda row: (
        row['reasons'] == ['stratified-sample'], -row['risk_score'],
        row['clip_id']))
