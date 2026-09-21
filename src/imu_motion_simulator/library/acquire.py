"""Download an explicit public-file manifest; credentials stay in the browser.

Run on Ubuntu. Partial files are resumable and never treated as completed assets.
No installation, archive extraction, deletion, or simulation is performed.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from urllib.parse import urlsplit
import zipfile


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def hashes(path):
    sha = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            sha.update(block)
            md5.update(block)
    return {'sha256': sha.hexdigest(), 'md5': md5.hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--reserve-gib', type=int, default=100)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    jobs = json.loads(args.manifest.read_text())['files']
    ids, paths = set(), set()
    for job in jobs:
        url = urlsplit(job['url'])
        if url.scheme != 'https' or url.username or url.password or url.query or url.fragment:
            raise ValueError('Only explicitly listed public HTTPS URLs without credentials/query')
        path = Path(job['path'])
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Unsafe destination')
        if job['id'] in ids or str(path) in paths:
            raise ValueError('Duplicate file id or destination')
        ids.add(job['id']); paths.add(str(path))
    logs = root / 'logs'
    logs.mkdir(exist_ok=True)
    lock = threading.Lock()
    status_path = logs / 'acquisition-status.json'
    if status_path.is_file():
        status = json.loads(status_path.read_text())
        if not isinstance(status, dict):
            raise ValueError('Existing acquisition status must be a JSON object')
    else:
        status = {}

    def report(job, **fields):
        with lock:
            record = status.setdefault(job['id'], dict(job))
            record.update(job)
            if fields.get('state') != 'blocked':
                record.pop('error', None)
            record.update(updated_utc=now(), **fields)
            atomic_json(status_path, status)
            with (logs / 'acquisition-events.jsonl').open('a') as stream:
                stream.write(json.dumps({'id': job['id'], 'utc': now(), **fields}) + '\n')

    def acquire(job):
        target = root / job['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + '.partial')
        receipt = target.with_name(target.name + '.receipt.json')
        try:
            if target.exists():
                if not receipt.exists():
                    raise RuntimeError('Existing final file has no receipt; preserved for review')
                old = json.loads(receipt.read_text())
                size = target.stat().st_size
                actual = hashes(target)
                if old.get('url') != job['url'] or old.get('sha256') != actual['sha256']:
                    raise RuntimeError('Existing final file differs from receipt; preserved')
                if job.get('bytes') is not None and size != job['bytes']:
                    raise RuntimeError('Existing final file size differs from explicit manifest')
                for kind in ['sha256', 'md5']:
                    if job.get(kind) and actual[kind] != job[kind]:
                        raise RuntimeError(f'Existing final file fails official {kind}')
                report(job, state='verified', bytes=size, **actual)
                return
            report(job, state='downloading', bytes=partial.stat().st_size if partial.exists() else 0)
            current_size = partial.stat().st_size if partial.exists() else 0
            expected_size = job.get('bytes')
            if expected_size is not None and current_size > expected_size:
                raise RuntimeError('Partial file is larger than explicit manifest; preserved')
            if expected_size is None or current_size < expected_size:
                if shutil.disk_usage(root).free < args.reserve_gib * 1024**3:
                    raise RuntimeError('Free-space reserve reached; partials preserved')
                logpath = logs / (job['id'] + '.curl.log')
                with logpath.open('ab') as log:
                    process = subprocess.Popen([
                        'curl', '--fail', '--location', '--silent', '--show-error',
                        '--proto', '=https', '--proto-redir', '=https',
                        '--connect-timeout', '20', '--retry', '3', '--retry-delay', '5',
                        '--speed-limit', '1024', '--speed-time', '120',
                        '--continue-at', '-', '--output', str(partial), job['url'],
                    ], stdin=subprocess.DEVNULL, stdout=log, stderr=log)
                    while process.poll() is None:
                        if shutil.disk_usage(root).free < args.reserve_gib * 1024**3 or (root / 'STOP_DOWNLOADS').exists():
                            process.terminate()
                            try:
                                process.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                process.kill(); process.wait()
                            raise RuntimeError('Stopped by space protection or STOP_DOWNLOADS; partials preserved')
                        report(job, state='downloading', bytes=partial.stat().st_size if partial.exists() else 0)
                        time.sleep(5)
                if process.returncode:
                    raise RuntimeError(f'curl exited {process.returncode}; see {logpath.name}')
            size = partial.stat().st_size
            if size == 0 or (job.get('bytes') is not None and size != job['bytes']):
                raise RuntimeError('Downloaded size differs from explicit manifest')
            report(job, state='checking', bytes=size)
            actual = hashes(partial)
            for kind in ['sha256', 'md5']:
                if job.get(kind) and actual[kind] != job[kind]:
                    raise RuntimeError(f'Official {kind} mismatch; partial preserved')
            archive = None
            if target.suffix.lower() == '.zip':
                with zipfile.ZipFile(partial) as data:
                    members = data.infolist()
                    archive = {'members': len(members), 'declared_unpacked_bytes': sum(x.file_size for x in members)}
                    if archive['declared_unpacked_bytes'] > 1024**4:
                        raise RuntimeError('Archive declares over one TiB; review before checking')
                    bad = data.testzip()
                    if bad is not None:
                        raise RuntimeError('ZIP CRC failure; partial preserved')
                    archive['crc'] = 'passed'
            record = dict(job, completed_utc=now(), bytes=size, **actual, archive=archive)
            # Another process must not silently replace the final file.
            os.link(partial, target)
            partial.unlink()
            atomic_json(receipt, record)
            report(job, state='verified', bytes=size, **actual, archive=archive)
        except Exception as error:
            report(job, state='blocked', error=f'{type(error).__name__}: {error}',
                   bytes=partial.stat().st_size if partial.exists() else 0)

    lockfile = (logs / 'acquisition.lock').open('w')
    import fcntl
    fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(acquire, jobs))
    result = {job['id']: status[job['id']]['state'] for job in jobs}
    print(json.dumps(result), flush=True)
    return 0 if all(state == 'verified' for state in result.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
