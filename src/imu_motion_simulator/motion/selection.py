"""Small immutable views over canonical motion artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..contracts.common import sha256_file
from ..contracts.internal import read_internal


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode()


def write_selection(path, motion, *, start_frame=0, stop_frame=None,
                    label_candidates=None):
    path, motion = Path(path), Path(motion)
    description, metadata, _ = read_internal(motion, 'motion')
    if description.get('motion_contract_version') != 2:
        raise ValueError('Selections require canonical motion-v2')
    stop_frame = description['frames'] if stop_frame is None else stop_frame
    if type(start_frame) is not int or type(stop_frame) is not int \
            or not 0 <= start_frame < stop_frame <= description['frames']:
        raise ValueError('Invalid selection frame range')
    value = {
        'schema': 'imu_motion_simulator.motion_selection.v1',
        'motion_id': metadata['kind_metadata']['motion_id'],
        'motion_sha256': sha256_file(motion),
        'start_frame': start_frame,
        'stop_frame': stop_frame,
        'label_candidates': list(label_candidates or []),
    }
    value['selection_id'] = hashlib.sha256(_canonical(value)).hexdigest()[:32]
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')
    temporary.replace(path)
    return validate_selection(path, motion)


def validate_selection(path, motion):
    path, motion = Path(path), Path(motion)
    value = json.loads(path.read_text())
    required = {'schema', 'selection_id', 'motion_id', 'motion_sha256',
                'start_frame', 'stop_frame', 'label_candidates'}
    if set(value) != required or value['schema'] != 'imu_motion_simulator.motion_selection.v1':
        raise ValueError('Invalid motion selection fields')
    expected = dict(value); selection_id = expected.pop('selection_id')
    if hashlib.sha256(_canonical(expected)).hexdigest()[:32] != selection_id:
        raise ValueError('Motion selection identity mismatch')
    description, metadata, _ = read_internal(motion, 'motion')
    if value['motion_sha256'] != sha256_file(motion) \
            or value['motion_id'] != metadata['kind_metadata']['motion_id']:
        raise ValueError('Motion selection parent mismatch')
    if not 0 <= value['start_frame'] < value['stop_frame'] <= description['frames']:
        raise ValueError('Motion selection range')
    if not isinstance(value['label_candidates'], list):
        raise ValueError('Motion selection labels')
    return value


def selection_slice(selection, motion):
    value = validate_selection(selection, motion)
    return slice(value['start_frame'], value['stop_frame']), value
