"""Build immutable corpus plans from AMASS SMPL+H G archives."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
import zipfile

import numpy as np

from ..contracts.common import sha256_file
from ..labels.babel import candidate_from_index, load_index
from ..motion.smplh import (NATIVE_ADAPTER, STAGEII_ADAPTER,
                            normalize_amass_gender)
from .plan import load_plan, resolve_inside


class AMASSAdapterRequired(ValueError):
    """A source member cannot satisfy the native SMPL+H G contract."""


class AMASSNotMotion(ValueError):
    """A source member is structurally valid but cannot encode motion."""


def _atomic_json(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')
    temporary.replace(path)


def _recipe_sha256():
    digest = hashlib.sha256()
    for path in (Path(__file__),
                 Path(__file__).parents[1] / 'labels/babel.py',
                 Path(__file__).parents[1] / 'motion/smplh.py'):
        digest.update(path.name.encode()); digest.update(b'\0')
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _clip_id(member):
    name = member.removesuffix('_poses.npz').removesuffix('_stageii.npz')
    slug = re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')[-72:]
    return slug + '-' + hashlib.sha256(member.encode()).hexdigest()[:8]


def _npy_header(archive, name):
    with archive.open(name + '.npy') as stream:
        version = np.lib.format.read_magic(stream)
        reader = (np.lib.format.read_array_header_1_0
                  if version == (1, 0)
                  else np.lib.format.read_array_header_2_0)
        return reader(stream)


def _small_npy(archive, name):
    return np.lib.format.read_array(
        io.BytesIO(archive.read(name + '.npy')), allow_pickle=False)


def _inspect_npz(raw, member, adapter_id):
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = {Path(name).stem for name in archive.namelist()}
        if adapter_id == NATIVE_ADAPTER:
            required = {'poses', 'trans', 'betas', 'dmpls', 'gender',
                        'mocap_framerate'}
            rate_field = 'mocap_framerate'
        elif adapter_id == STAGEII_ADAPTER:
            required = {'poses', 'trans', 'betas', 'gender',
                        'mocap_frame_rate', 'surface_model_type', 'num_betas'}
            rate_field = 'mocap_frame_rate'
        else:
            raise AMASSAdapterRequired('Unknown AMASS adapter: ' + adapter_id)
        if not required <= names:
            raise AMASSAdapterRequired(
                'Missing required AMASS member fields: ' + member)
        arrays = ['poses', 'trans', 'betas']
        if adapter_id == NATIVE_ADAPTER:
            arrays.append('dmpls')
        shapes = {name: _npy_header(archive, name)[0] for name in arrays}
        frames = shapes['poses'][0] if len(shapes['poses']) == 2 else 0
        fps = float(_small_npy(archive, rate_field))
        gender = normalize_amass_gender(_small_npy(archive, 'gender'))
        expected_shapes = {'poses': (frames, 156), 'trans': (frames, 3),
                           'betas': (16,)}
        if adapter_id == NATIVE_ADAPTER:
            expected_shapes['dmpls'] = (frames, 8)
        else:
            surface = str(_small_npy(archive, 'surface_model_type').item())
            components = int(_small_npy(archive, 'num_betas'))
            if surface != 'smplh' or components != 16:
                raise AMASSAdapterRequired(
                    'Stage-II member is not 16-beta SMPL+H: ' + member)
            split = {'root_orient': (frames, 3),
                     'pose_body': (frames, 63),
                     'pose_hand': (frames, 90)}
            for name, shape in split.items():
                if name in names and _npy_header(archive, name)[0] != shape:
                    raise AMASSAdapterRequired(
                        'Unsupported Stage-II pose split: ' + member)
        if shapes != expected_shapes \
                or gender not in ('male', 'female', 'neutral') \
                or not np.isfinite(fps) or fps <= 0:
            raise AMASSAdapterRequired(
                'Unsupported AMASS member structure: ' + member)
        if frames < 2:
            raise AMASSNotMotion(
                'AMASS member has fewer than two frames: ' + member)
    return frames, fps, gender


def _stageii_candidate(member, source_dataset):
    stem = Path(member).name.removesuffix('_stageii.npz')
    parts = stem.split('_')
    if source_dataset == 'SOMA':
        action = parts[0]
        attributes = {'take': '_'.join(parts[1:]) or None}
    elif source_dataset == 'GRAB' and len(parts) >= 2:
        take = parts[-1]
        action = parts[-2]
        attributes = {'object': '_'.join(parts[:-2]), 'take': take}
    else:
        return []
    return [{
        'kind': 'recording-candidate',
        'code': action.lower(),
        'name': action.replace('-', ' ').replace('_', ' ').strip().title(),
        'is_fall': False,
        'origin': 'stageii-source-member',
        'attributes': attributes,
    }]


def build_amass_plan(library_root, output, *, study_id, source_dataset,
                     amass_archive, smplh_archive, dmpl_archive, layout,
                     profile, babel_archive=None, adapter_id=NATIVE_ADAPTER):
    """Inventory one source archive and freeze every motion into a plan."""
    library_root, output = Path(library_root).resolve(), Path(output)
    amass_path = resolve_inside(library_root, amass_archive)
    if output.exists():
        raise FileExistsError(output)
    babel = ({} if babel_archive is None else
             load_index(resolve_inside(library_root, babel_archive)))
    clips, excluded, duration_s, matched = [], [], 0., 0
    with tarfile.open(amass_path, 'r|bz2') as archive:
        for member in archive:
            if not member.isfile():
                continue
            if member.name.endswith('_stageii.npz') \
                    and adapter_id != STAGEII_ADAPTER:
                raise AMASSAdapterRequired(
                    'Stage-II AMASS member requires an explicit adapter: '
                    + member.name)
            suffix = ('_stageii.npz' if adapter_id == STAGEII_ADAPTER
                      else '_poses.npz')
            if not member.name.endswith(suffix):
                continue
            raw = archive.extractfile(member).read()
            try:
                frames, fps, gender = _inspect_npz(
                    raw, member.name, adapter_id)
            except AMASSNotMotion:
                excluded.append({'source_member': member.name,
                                 'reason': 'fewer-than-two-frames'})
                continue
            if adapter_id == STAGEII_ADAPTER:
                labels = _stageii_candidate(member.name, source_dataset)
                candidate = None
            else:
                candidate = candidate_from_index(
                    babel, member.name, source_dataset=source_dataset)
                labels = [] if candidate is None else candidate['labels']
                matched += candidate is not None
            duration_s += (frames - 1) / fps
            clips.append({
                'id': _clip_id(member.name),
                'source_dataset': source_dataset,
                'source_member': member.name,
                'expected_gender': gender,
                'expected_frames': frames,
                'frame_range': [0, frames],
                'label_candidates': labels})
    clips.sort(key=lambda clip: clip['source_member'])
    if not clips:
        raise AMASSAdapterRequired(
            'AMASS archive contains no recognized SMPL+H G motion members: '
            + source_dataset)
    value = {
        'schema': ('imu_motion_simulator.kinematic_plan.v1'
                   if adapter_id == NATIVE_ADAPTER
                   else 'imu_motion_simulator.kinematic_plan.v2'),
        'study_id': study_id,
        'description': (f'Pure-kinematic full {source_dataset} corpus plan; '
                        'generated from immutable AMASS inventory; includes '
                        'only sequences with at least two frames'),
        'inputs': {'amass_archive': str(amass_archive),
                   'smplh_archive': str(smplh_archive),
                   'dmpl_archive': str(dmpl_archive)},
        'sensor': {'layout': str(layout), 'profile': str(profile)},
        'review': {'primary': 'threejs', 'mp4': 'on-demand',
                   'policy': 'automatic-all-risk-stratified-human'},
        'clips': clips}
    if adapter_id != NATIVE_ADAPTER:
        value['source_adapter'] = {
            'id': adapter_id,
            'member_suffix': '_stageii.npz',
            'dynamic_shape': 'disabled-zero'}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')
    temporary.replace(output)
    load_plan(output)
    return {'output': str(output), 'sha256': sha256_file(output),
            'clips': len(clips), 'source_duration_hours': duration_s / 3600,
            'babel_matched_clips': matched,
            'excluded_members': excluded}


def build_amass_catalog(library_root, output, *, amass_directory,
                        smplh_archive, dmpl_archive, layout, profile,
                        babel_archive=None, include_sources=(),
                        source_adapters=None):
    """Freeze one independently resumable plan for every local AMASS archive."""
    library_root, output = Path(library_root).resolve(), Path(output).resolve()
    directory = resolve_inside(library_root, amass_directory)
    if output.exists():
        raise FileExistsError(output)
    include_sources = tuple(dict.fromkeys(include_sources))
    source_adapters = dict(source_adapters or {})
    allowed_adapters = {NATIVE_ADAPTER, STAGEII_ADAPTER}
    if set(source_adapters.values()) - allowed_adapters:
        raise ValueError('AMASS catalog contains an unknown source adapter')
    archives = sorted(directory.glob('*.tar.bz2'))
    if include_sources:
        wanted = set(include_sources)
        archives = [path for path in archives
                    if path.name.removesuffix('.tar.bz2') in wanted]
        missing = wanted - {
            path.name.removesuffix('.tar.bz2') for path in archives}
        if missing:
            raise FileNotFoundError(
                'Requested AMASS source archive missing: '
                + ', '.join(sorted(missing)))
    if not archives:
        raise ValueError('AMASS catalog directory contains no source archives')
    archive_sources = {
        path.name.removesuffix('.tar.bz2') for path in archives}
    unknown_adapter_sources = set(source_adapters) - archive_sources
    if unknown_adapter_sources:
        raise ValueError(
            'AMASS adapter source is not selected or present: '
            + ', '.join(sorted(unknown_adapter_sources)))
    working = output.with_name(output.name + '.partial')
    plans = working / 'plans'; plans.mkdir(parents=True, exist_ok=True)
    state_path = working / 'catalog-state.json'
    config = {
        'generator_recipe_sha256': _recipe_sha256(),
        'amass_directory': str(amass_directory),
        'archives': [{'path': str(path.relative_to(library_root)),
                      'bytes': path.stat().st_size} for path in archives],
        'smplh_archive': str(smplh_archive), 'dmpl_archive': str(dmpl_archive),
        'babel_archive': None if babel_archive is None else str(babel_archive),
        'layout': str(layout), 'profile': str(profile),
        'include_sources': list(include_sources),
        'source_adapters': dict(sorted(source_adapters.items()))}
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get('schema') != 'imu_motion_simulator.amass_catalog_state.v1' \
                or state.get('config') != config:
            raise ValueError('Existing AMASS catalog partial has different inputs')
    else:
        state = {'schema': 'imu_motion_simulator.amass_catalog_state.v1',
                 'config': config, 'completed': {}}
        _atomic_json(state_path, state)
    for archive in archives:
        source = archive.name.removesuffix('.tar.bz2')
        if source in state['completed']:
            completed = state['completed'][source]
            if completed['status'] == 'plan-ready':
                plan = working / completed['plan']
                if not plan.is_file() \
                        or sha256_file(plan) != completed['plan_sha256']:
                    raise ValueError('Completed AMASS catalog plan changed: ' + source)
                load_plan(plan)
            continue
        plan = plans / f'{source}.plan.json'
        if plan.exists():
            raise ValueError('Untracked partial AMASS plan: ' + str(plan))
        identity = {
            'source_dataset': source,
            'archive': str(archive.relative_to(library_root)),
            'archive_bytes': archive.stat().st_size,
            'adapter_id': source_adapters.get(source, NATIVE_ADAPTER)}
        try:
            report = build_amass_plan(
                library_root, plan,
                study_id=f'kinematic-{source.lower()}-full-v1',
                source_dataset=source,
                amass_archive=archive.relative_to(library_root),
                smplh_archive=smplh_archive, dmpl_archive=dmpl_archive,
                layout=layout, profile=profile, babel_archive=babel_archive,
                adapter_id=identity['adapter_id'])
        except AMASSAdapterRequired as error:
            state['completed'][source] = {
                **identity, 'status': 'adapter-required',
                'reason': str(error), 'plan': None, 'plan_sha256': None,
                'clips': None, 'source_duration_hours': None,
                'babel_matched_clips': None, 'excluded_members': []}
        else:
            state['completed'][source] = {
                **identity, 'status': 'plan-ready', 'reason': None,
                'plan': str(plan.relative_to(working)),
                'plan_sha256': report['sha256'], 'clips': report['clips'],
                'source_duration_hours': report['source_duration_hours'],
                'babel_matched_clips': report['babel_matched_clips'],
                'excluded_members': report['excluded_members']}
        _atomic_json(state_path, state)
    sources = [state['completed'][name]
               for name in sorted(state['completed'])]
    ready = [item for item in sources if item['status'] == 'plan-ready']
    catalog = {
        'schema': ('imu_motion_simulator.amass_catalog.v2'
                   if include_sources or source_adapters
                   else 'imu_motion_simulator.amass_catalog.v1'),
        'inputs': config, 'source_archives': len(sources),
        'plan_ready_archives': len(ready),
        'adapter_required_archives': len(sources) - len(ready),
        'excluded_members': sum(
            len(item['excluded_members']) for item in sources),
        'clips': sum(item['clips'] for item in ready),
        'source_duration_hours': sum(
            item['source_duration_hours'] for item in ready),
        'babel_matched_clips': sum(
            item['babel_matched_clips'] for item in ready),
        'sources': sources}
    _atomic_json(working / 'catalog.json', catalog)
    state_path.unlink()
    working.rename(output)
    return {key: catalog[key] for key in (
        'source_archives', 'plan_ready_archives',
        'adapter_required_archives', 'excluded_members', 'clips',
        'source_duration_hours', 'babel_matched_clips')} | {
            'output': str(output),
            'sha256': sha256_file(output / 'catalog.json')}
