"""The cameras live-view HTTP server: serves the page and a frame."""

import urllib.request

import numpy as np

from rammp_curobo_ros.cameras import _ViewServer


def test_view_server_serves_page_and_stream_frame():
    srv = _ViewServer(0)  # ephemeral port
    port = srv.httpd.server_address[1]
    srv.update(np.zeros((24, 32, 3), np.uint8))
    page = urllib.request.urlopen(
        "http://127.0.0.1:%d/" % port, timeout=5
    ).read()
    assert b"/stream" in page
    stream = urllib.request.urlopen(
        "http://127.0.0.1:%d/stream" % port, timeout=5
    )
    head = stream.read(200)
    assert b"image/jpeg" in head
    stream.close()
    srv.httpd.shutdown()
