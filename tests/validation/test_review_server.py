import json
from http.server import ThreadingHTTPServer
from threading import Thread
from urllib.request import urlopen

import pytest

from imu_motion_simulator.review.server import _handler


def test_review_server_advertises_write_capability_only_on_its_endpoint(tmp_path):
    try:
        server = ThreadingHTTPServer(('127.0.0.1', 0), _handler(tmp_path))
    except PermissionError:
        pytest.skip('local sandbox does not allow even loopback sockets')
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        url = f'http://127.0.0.1:{server.server_address[1]}'
        with urlopen(url + '/api/capabilities') as response:
            assert response.status == 200
            assert json.load(response) == {'review_write': True}
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
