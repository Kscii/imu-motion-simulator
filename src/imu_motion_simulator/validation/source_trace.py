"""Compare archived SMPL+H pose steps with immutable canonical motions."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from imu_motion_simulator.contracts.common import get_json, sha256_file


def _step_degrees(quaternion_xyzw):
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    length = np.linalg.norm(quaternion, axis=-1)
    cosine = np.sum(quaternion[1:] * quaternion[:-1], axis=-1) \
        / (length[1:] * length[:-1])
    return np.degrees(2 * np.arccos(np.clip(np.abs(cosine), 0, 1)))


def _trace(candidate, objects, raw):
    with np.load(io.BytesIO(raw), allow_pickle=False) as source:
        poses = np.asarray(source['poses'], dtype=np.float64)
        translation = np.asarray(source['trans'], dtype=np.float64)
    paths = {role: Path(objects[digest]['local_path'])
             for role, digest in candidate['objects'].items()}
    selection = json.loads(paths['selection'].read_text())
    with h5py.File(paths['motion'], 'r') as motion:
        metadata = get_json(motion, 'metadata')
        local = motion['data/joint_local_quaternion_wxyz'][:]
        root = motion['data/root_position_m'][:]
    if len(poses) != len(local) or poses.shape != (len(local), 156):
        raise ValueError('Archive and canonical motion frame counts differ')
    source_quaternion = Rotation.from_rotvec(
        poses.reshape(-1, 3)).as_quat().reshape(len(poses), 52, 4)
    source_step = _step_degrees(source_quaternion)
    canonical_step = _step_degrees(local[..., [1, 2, 3, 0]])
    source_root_step = np.linalg.norm(np.diff(translation, axis=0), axis=1)
    canonical_root_step = np.linalg.norm(np.diff(root, axis=0), axis=1)
    start, stop = selection['start_frame'], selection['stop_frame']
    if not 0 <= start < stop <= len(poses):
        raise ValueError('Invalid source trace selection')
    step_slice = slice(start, stop - 1)
    joint_names = metadata['kind_metadata']['joint_names']
    peak_frame, peak_joint = np.unravel_index(
        int(np.argmax(source_step[step_slice])),
        source_step[step_slice].shape)
    peak_frame += start
    return {
        'candidate_id': candidate['candidate_id'],
        'source_dataset': candidate['source_dataset'],
        'source_member': candidate['source_member'],
        'source_member_sha256': hashlib.sha256(raw).hexdigest(),
        'source_archive_sha256': metadata['kind_metadata']['original_archive_sha256'],
        'input_objects': candidate['objects'],
        'frames': len(poses),
        'max_joint_step_difference_degrees': float(np.max(
            np.abs(source_step[step_slice] - canonical_step[step_slice]))),
        'max_root_step_difference_m': float(np.max(
            np.abs(source_root_step[step_slice]
                   - canonical_root_step[step_slice]))),
        'first_joint_step_degrees': float(source_step[start].max()),
        'last_joint_step_degrees': float(source_step[stop - 2].max()),
        'source_largest_joint_step': {
            'source_frame': int(peak_frame),
            'joint': joint_names[peak_joint],
            'degrees': float(source_step[peak_frame, peak_joint])},
        'source_root_net_displacement_m': float(np.linalg.norm(
            translation[stop - 1] - translation[start])),
    }


def trace_sources(corpus_path, production, library, output, candidate_ids):
    corpus_path, production, library, output = (
        Path(value).resolve()
        for value in (corpus_path, production, library, output))
    if output.exists():
        raise FileExistsError(output)
    corpus = json.loads(corpus_path.read_text())
    candidates = {row['candidate_id']: row for row in corpus['candidates']}
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError('Duplicate candidate ID')
    missing = set(candidate_ids) - set(candidates)
    if missing:
        raise ValueError('Unknown candidate IDs: ' + ', '.join(sorted(missing)))
    objects = {item['sha256']: item for item in corpus['objects']}
    groups = defaultdict(lambda: defaultdict(list))
    for candidate_id in candidate_ids:
        row = candidates[candidate_id]
        groups[row['source_dataset']][row['source_member']].append(row)
    results = []
    for source, wanted in sorted(groups.items()):
        group = 'stageii' if source in {'GRAB', 'SOMA'} else 'native'
        plan = json.loads((production / group / 'sources' / source
                           / 'plan.json').read_text())
        archive = library / plan['inputs']['amass_archive']
        remaining = dict(wanted)
        with tarfile.open(archive, 'r|bz2') as handle:
            for member in handle:
                if member.name not in remaining:
                    continue
                raw = handle.extractfile(member).read()
                for candidate in remaining.pop(member.name):
                    results.append(_trace(candidate, objects, raw))
                if not remaining:
                    break
        if remaining:
            raise ValueError('Archive members missing: ' + ', '.join(remaining))
    report = {
        'schema': 'imu_motion_simulator.source_motion_trace.v1',
        'corpus_sha256': sha256_file(corpus_path),
        'code_sha256': sha256_file(__file__),
        'candidate_count': len(candidate_ids),
        'results': sorted(results, key=lambda row: row['candidate_id'])}
    temporary = output.with_suffix(output.suffix + '.partial')
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.rename(output)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('corpus', type=Path)
    parser.add_argument('production', type=Path)
    parser.add_argument('library', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('candidate_ids', nargs='+')
    args = parser.parse_args()
    report = trace_sources(
        args.corpus, args.production, args.library,
        args.output, args.candidate_ids)
    print(json.dumps({'candidate_count': report['candidate_count'],
                      'output': str(args.output)}, indent=2))
