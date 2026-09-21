"""Shared strict JSON, HDF5 types, clocks and immutable file publication."""
from __future__ import annotations

from contextlib import contextmanager
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Callable
from uuid import UUID

import h5py
import numpy as np


class ContractError(ValueError):
    """An artifact does not satisfy its declared contract."""


UTF8 = h5py.string_dtype('utf-8')
LIBVER = ('v110', 'v114')
COORDINATES = dict(length_unit='m', angle_unit='rad', time_unit='s',
                   world='RH-Xforward-Yleft-Zup', quaternion='wxyz-active')


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def text(value) -> str:
    return value.decode('utf-8') if isinstance(value, bytes) else str(value)


def json_load(value):
    def pairs(items):
        result = {}
        for key, item in items:
            require(key not in result, f'duplicate JSON key: {key}')
            result[key] = item
        return result

    def invalid(value):
        raise ContractError(f'nonfinite JSON number: {value}')

    try:
        result = json.loads(text(value), object_pairs_hook=pairs, parse_constant=invalid)
    except json.JSONDecodeError as error:
        raise ContractError('invalid JSON') from error
    def finite_json(item):
        if isinstance(item, float):
            require(math.isfinite(item), 'nonfinite JSON number')
        elif isinstance(item, dict):
            for child in item.values(): finite_json(child)
        elif isinstance(item, list):
            for child in item: finite_json(child)
    finite_json(result)
    return result


def json_dump(value) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(',', ':'))


def put_json(group, name, value):
    return group.create_dataset(name, data=json_dump(value), dtype=UTF8)


def get_json(group, name):
    dataset = group[name]
    require(isinstance(dataset, h5py.Dataset) and dataset.shape == (), f'{name}: expected scalar JSON')
    info = h5py.check_string_dtype(dataset.dtype)
    require(info is not None and info.encoding == 'utf-8', f'{name}: expected UTF-8 JSON')
    return json_load(dataset[()])


def fields_dtype(fields):
    return np.dtype([(name, UTF8 if dtype == 'UTF-8' else np.dtype(dtype))
                     for name, dtype in fields])


def check_table(dataset, fields):
    require(isinstance(dataset, h5py.Dataset) and dataset.ndim == 1, 'expected one-dimensional table')
    require(dataset.dtype.names == tuple(name for name, _ in fields), f'{dataset.name}: field order')
    for name, expected in fields:
        actual = dataset.dtype.fields[name][0]
        info = h5py.check_string_dtype(actual)
        valid = (info is not None and info.encoding == 'utf-8') if expected == 'UTF-8' else actual == np.dtype(expected)
        require(valid, f'{dataset.name}.{name}: dtype')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def digest_dataset(dataset):
    digest = hashlib.sha256()
    for start in range(0, len(dataset), 1024 * 1024):
        digest.update(dataset[start:start + 1024 * 1024].tobytes())
    return digest.hexdigest()


def sha_string(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def uuid_string(value):
    try:
        return str(UUID(value)) == value
    except (ValueError, TypeError, AttributeError):
        return False


def relative_path(value):
    require(isinstance(value, str) and value and '\\' not in value, 'invalid logical path')
    path = PurePosixPath(value)
    require(not path.is_absolute() and '..' not in path.parts and '.' not in value.split('/'), 'path escapes package')
    require(str(path) == value and ':' not in value and '\x00' not in value, 'noncanonical logical path')
    return path


def local_tree(handle):
    """Reject external/soft links, cycles/aliases, VDS and external dataset storage."""
    seen = set()

    def visit(group):
        for name in group:
            require(isinstance(group.get(name, getlink=True), h5py.HardLink), 'external/soft HDF5 link')
            obj = group[name]
            address = h5py.h5o.get_info(obj.id).addr
            require(address not in seen, 'aliased/cyclic HDF5 object')
            seen.add(address)
            if isinstance(obj, h5py.Group):
                visit(obj)
            else:
                require(not obj.is_virtual and not obj.external, 'external/virtual HDF5 dataset')
    seen.add(h5py.h5o.get_info(handle.id).addr)
    visit(handle)


def finite(dataset):
    if dataset.size == 0:
        return
    if dataset.ndim == 0:
        require(np.isfinite(dataset[()]).all(), f'{dataset.name}: nonfinite')
        return
    for start in range(0, len(dataset), 65536):
        require(np.isfinite(dataset[start:start + 65536]).all(), f'{dataset.name}: nonfinite')


def time_ns(steps, numerator: int, denominator: int):
    require(type(numerator) is int and type(denominator) is int and numerator > 0 and denominator > 0, 'invalid period')
    scale = Fraction(numerator * 1_000_000_000, denominator)
    values = []
    for step in steps:
        require(isinstance(step, (int, np.integer)) and not isinstance(step, (bool, np.bool_)), 'noninteger step')
        value = int(step) * scale
        sign = 1 if value >= 0 else -1
        value = abs(value)
        result = sign * ((2 * value.numerator + value.denominator) // (2 * value.denominator))
        require(np.iinfo(np.int64).min <= result <= np.iinfo(np.int64).max, 'time overflow')
        values.append(result)
    return np.asarray(values, dtype='<i8')


def lower_bound(dataset, target, *, right=False):
    lo, hi = 0, len(dataset)
    while lo < hi:
        mid = (lo + hi) // 2
        value = int(dataset[mid])
        if value < target or (right and value == target):
            lo = mid + 1
        else:
            hi = mid
    return lo


def window_rows(dataset, start_ns, stop_ns):
    require(type(start_ns) is int and type(stop_ns) is int and start_ns <= stop_ns, 'invalid time window')
    require(len(dataset) > 0, 'empty time axis')
    return (max(0, lower_bound(dataset, start_ns) - 1),
            min(len(dataset), lower_bound(dataset, stop_ns, right=True) + 1))


@contextmanager
def atomic_h5(path, validate: Callable[[Path], object]):
    """Close, validate, fsync and atomically publish without overwriting a file."""
    path = Path(path)
    require(not path.exists(), f'output already exists: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.partial', dir=path.parent)
    os.close(fd)
    tmp = Path(temporary)
    try:
        with h5py.File(tmp, 'w', libver=LIBVER) as handle:
            yield handle
        validate(tmp)
        with tmp.open('rb') as stream:
            os.fsync(stream.fileno())
        os.link(tmp, path)  # Atomic no-replace publication on the supported Linux filesystem.
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        tmp.unlink(missing_ok=True)
