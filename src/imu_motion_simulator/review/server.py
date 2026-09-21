"""Local HTTP service for playback and immutable review decisions."""
from __future__ import annotations

import json
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .bundle import _atomic_json, validate_bundle


def _handler(directory):
    root = Path(directory).resolve()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)

        def do_GET(self):
            if self.path == '/api/capabilities':
                payload = b'{"review_write":true}'
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            super().do_GET()

        def do_POST(self):
            if self.path != '/api/review':
                self.send_error(404); return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                request = json.loads(self.rfile.read(length))
                if set(request) != {'decision', 'reviewer', 'reason', 'labels'} \
                        or request['decision'] not in ('accepted', 'rejected') \
                        or not request['reviewer'] or not request['reason'] \
                        or not isinstance(request['labels'], list):
                    raise ValueError('Invalid review decision')
                report = validate_bundle(root)
                previous = json.loads((root / f"review-r{report['latest_revision']}.json").read_text())
                revision = report['latest_revision'] + 1
                value = dict(previous, revision=revision,
                             decision=request['decision'], reviewer=request['reviewer'],
                             reason=request['reason'], labels=request['labels'])
                _atomic_json(root / f'review-r{revision}.json', value)
                payload = json.dumps({'saved': True, 'revision': revision}).encode()
                self.send_response(200); self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload))); self.end_headers()
                self.wfile.write(payload)
            except (ValueError, OSError, json.JSONDecodeError) as error:
                payload = json.dumps({'saved': False, 'error': str(error)}).encode()
                self.send_response(400); self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload))); self.end_headers()
                self.wfile.write(payload)

    return Handler


def serve(directory, *, host='127.0.0.1', port=8765, open_browser=False):
    validate_bundle(directory)
    server = ThreadingHTTPServer((host, port), _handler(directory))
    url = f'http://{host}:{server.server_address[1]}/'
    if open_browser:
        webbrowser.open(url)
    print(url, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
