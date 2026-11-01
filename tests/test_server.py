"""The public server: one test per finding of the 2026-11-01 review."""
import json
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from resonance import llm, server


@pytest.fixture
def srv(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "UPLOADS", tmp_path / "media" / "uploads")
    monkeypatch.setattr(server, "RUNS", tmp_path / "out" / "runs")
    monkeypatch.setattr(server, "SAMPLES", tmp_path / "samples")
    monkeypatch.setattr(server, "SEEDED", {"seed-run"})
    monkeypatch.setattr(server, "JOBS", {})
    monkeypatch.setattr(server, "PUBLIC", True)
    monkeypatch.setattr(server, "_calls", [])
    monkeypatch.setattr(server, "_client_calls", {})
    for d in ("seed-run", "visitor-run"):
        (server.RUNS / d).mkdir(parents=True)
        (server.RUNS / d / "report.json").write_text(json.dumps({"timing": {}, "members": []}))
    (tmp_path / "out" / "secret.json").write_text("{}")
    server.UPLOADS.mkdir(parents=True)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def call(url, data=None, headers=None):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_no_path_traversal(srv):
    assert call(srv + "/runs/../secret.json")[0] == 404
    assert call(srv + "/runs/..%2F/secret.json")[0] == 404
    assert call(srv + "/api/jobs/..")[0] == 404
    assert call(srv + "/api/jobs/../inspect", b"{}")[0] == 404
    assert call(srv + "/api/frame?video=../../etc/passwd")[0] == 400


def test_body_cap_and_bad_json(srv):
    assert call(srv + "/api/review", b"x" * (300 << 10))[0] == 413
    assert call(srv + "/api/review", b"[1, 2]")[0] == 400


def test_upload_must_be_a_video(srv, tmp_path):
    playlist = b"#EXTM3U\n#EXT-X-MEDIA-SEQUENCE:0\n#EXTINF:10.0,\nfile:///etc/passwd\n#EXT-X-ENDLIST\n"
    assert call(srv + "/api/upload?name=evil.mp4", playlist)[0] == 415
    assert not list(server.UPLOADS.glob("*"))
    clip = tmp_path / "c.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=30:duration=1",
                    "-pix_fmt", "yuv420p", str(clip)], check=True)
    code, body = call(srv + "/api/upload?name=a%20b%23c&d.mp4", clip.read_bytes())
    assert code == 200
    name = json.loads(body)["video"]
    assert "#" not in name and "&" not in name and (server.UPLOADS / name).exists()
    shutil.copyfile(clip, server.RUNS / "visitor-run" / "hero.mp4")
    code, part = call(srv + "/runs/visitor-run/hero.mp4", headers={"Range": "bytes=0-9"})
    assert code == 206 and len(part) == 10


def test_recorded_examples_are_read_only_and_unknown_runs_cost_nothing(srv):
    assert call(srv + "/api/jobs/seed-run/inspect", b'{"context": "x"}')[0] == 403
    assert call(srv + "/api/jobs/seed-run/approve", b'{"by": "x"}')[0] == 403
    for _ in range(5):
        assert call(srv + "/api/jobs/nope/inspect", b"{}")[0] == 404
    assert server._calls == []


def test_model_budget_is_charged_per_call(monkeypatch):
    monkeypatch.setattr(server, "PUBLIC", True)
    monkeypatch.setattr(server, "_calls", [])
    monkeypatch.setattr(server, "MODEL_CALLS_PER_HOUR", 3)
    monkeypatch.setattr(llm, "CALL_GATE", server._model_call_allowed)
    monkeypatch.setenv("BASETEN_API_KEY", "test")
    sent = []
    monkeypatch.setattr(llm.urllib.request, "urlopen", lambda *a, **k: sent.append(1) or (_ for _ in ()).throw(OSError("offline")))
    for _ in range(3):
        with pytest.raises(RuntimeError, match="offline"):
            llm.chat([{"role": "user", "content": "hi"}], retries=1)
    with pytest.raises(RuntimeError, match="limit reached"):
        llm.chat([{"role": "user", "content": "hi"}], retries=1)
    assert len(sent) == 3


def test_bad_numbers_are_rejected_not_crashing(srv):
    assert call(srv + "/api/check?video=x.mp4&capture_fps=abc")[0] == 400
    assert call(srv + "/api/analyze", b'{"video": "missing.mp4", "members": [{}]}')[0] == 400


def test_negative_length_suffix_range_and_per_client_limit(srv, monkeypatch):
    import socket
    from urllib.parse import urlparse
    host, port = urlparse(srv).hostname, urlparse(srv).port
    with socket.create_connection((host, port), timeout=10) as sock:
        sock.sendall(b"POST /api/review HTTP/1.1\r\nHost: x\r\nContent-Length: -1\r\n\r\n{}")
        assert b" 413 " in sock.recv(200)
    (server.RUNS / "visitor-run" / "hero.mp4").write_bytes(bytes(range(256)) * 4)
    code, tail = call(srv + "/runs/visitor-run/hero.mp4", headers={"Range": "bytes=-10"})
    assert code == 206 and tail == (bytes(range(256)) * 4)[-10:]
    monkeypatch.setattr(server, "PER_CLIENT_PER_HOUR", 2)
    for _ in range(2):
        assert call(srv + "/api/review", b'{"video": "none.mp4", "members": []}')[0] == 400
    assert call(srv + "/api/review", b'{"video": "none.mp4", "members": []}')[0] == 429
