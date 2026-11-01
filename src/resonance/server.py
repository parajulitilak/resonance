"""Local web app: upload a clip, mark (or auto-detect) the cables, measure, inspect, approve.

Standard library only (http.server). Runs on 127.0.0.1; heavy work happens in background threads
and the page polls for status.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .analyze import analyze_tracking, save
from .displacement import track
from .members import Member
from .video import probe

ROOT = Path.cwd()
UPLOADS = ROOT / "media" / "uploads"
RUNS = ROOT / "out" / "runs"
SAM_CACHE = ROOT / "media" / "sam_cache"
STATIC = Path(__file__).parent / "static"
# Hosted demo: sample clips (with a samples.json and precomputed SAM results) and finished runs to replay
SAMPLES = Path(os.environ.get("RESONANCE_SAMPLES", ROOT / "samples"))
PUBLIC = os.environ.get("RESONANCE_PUBLIC") == "1"   # limits for a public URL with no login
MAX_UPLOAD = 400 << 20
MAX_BODY = 256 << 10                 # JSON request bodies
MODEL_CALLS_PER_HOUR = 150           # every model call (review, inspector turn, verifier batch, SAM API) counts
MAX_MEMBERS = 40
MAX_REVIEW = 25
MAX_RUNNING = 2                      # analyses (and, separately, inspections) at once
UPLOAD_QUOTA = 2 << 30               # oldest uploads are deleted beyond this (public mode)
MAX_RUNS = 150                       # oldest visitor runs are deleted beyond this (public mode)
JOBS: dict[str, dict] = {}
REVIEWS: dict[tuple, dict] = {}      # visual-review verdicts made by this server, never taken from a client
SEEDED: set[str] = set()             # recorded example runs: read-only on a public server
_calls: list[float] = []
_client_calls: dict[tuple, list[float]] = {}
_lock = threading.Lock()
PER_CLIENT_PER_HOUR = 6              # reviews and inspections each, per client address
_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,100}$")


def _model_call_allowed() -> bool:
    """Public deployments share one model key: a budget of model calls per hour, charged per call."""
    if not PUBLIC:
        return True
    with _lock:
        now = time.time()
        _calls[:] = [t for t in _calls if now - t < 3600]
        if len(_calls) >= MODEL_CALLS_PER_HOUR:
            return False
        _calls.append(now)
        return True


def _client_allowed(addr: str, kind: str) -> bool:
    """Public mode: one visitor cannot spend everyone's hourly budget."""
    if not PUBLIC:
        return True
    with _lock:
        now = time.time()
        log = [t for t in _client_calls.get((addr, kind), []) if now - t < 3600]
        if len(log) >= PER_CLIENT_PER_HOUR:
            _client_calls[(addr, kind)] = log
            return False
        _client_calls[(addr, kind)] = log + [now]
        return True


def _safe(name: str | None) -> str | None:
    """A run id or file name that cannot leave its folder."""
    name = str(name or "")
    return name if _SAFE.match(name) and ".." not in name else None


def _video(name) -> Path | None:
    n = _safe(name)
    return UPLOADS / n if n and (UPLOADS / n).is_file() else None


def _looks_like_video(path: Path) -> bool:
    """Only containers phones and cameras write: MP4/MOV, Matroska/WebM, AVI. Checked on the bytes, before ffmpeg
    sees the file (ffmpeg picks its demuxer from the content, and some demuxers read other files)."""
    head = path.open("rb").read(16)
    return (head[4:8] in (b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip") or head[:4] == b"\x1aE\xdf\xa3"
            or (head[:4] == b"RIFF" and head[8:12] == b"AVI "))


def _running(key: str) -> int:
    busy = {"stage": ("queued", "tracking", "analysing", "rendering"), "inspect_stage": ("running",)}[key]
    return sum(j.get(key) in busy for j in JOBS.values())


def _housekeep() -> None:
    """Public mode: keep uploads under a quota and the number of visitor runs bounded (oldest go first). Clips
    being measured or uploaded in the last 30 minutes are never removed."""
    if not PUBLIC:
        return
    busy = {j.get("video") for j in JOBS.values() if j.get("stage") in ("queued", "tracking", "analysing", "rendering")}
    fresh = time.time() - 1800
    ups = sorted((p for p in UPLOADS.glob("*") if p.is_file() and not p.name.startswith("sample-")
                  and p.name not in busy and p.stat().st_mtime < fresh), key=lambda p: p.stat().st_mtime)
    total = sum(p.stat().st_size for p in ups)
    while ups and total > UPLOAD_QUOTA:
        p = ups.pop(0)
        total -= p.stat().st_size
        p.unlink(missing_ok=True)
    runs = sorted((d for d in RUNS.glob("*") if d.is_dir() and d.name not in SEEDED), key=lambda d: d.stat().st_mtime)
    for d in runs[:max(0, len(runs) - MAX_RUNS)]:
        shutil.rmtree(d, ignore_errors=True)
        JOBS.pop(d.name, None)


def _review_key(video: str, p0, p1) -> tuple:
    """Verdicts follow the candidate's position, not its (editable) name."""
    return (video, tuple(round(float(v)) for v in p0), tuple(round(float(v)) for v in p1))


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:24]


def _sam_cached(video: Path, concept: str) -> dict | None:
    key = f"{_sha(video)}_{hashlib.sha256(concept.strip().lower().encode()).hexdigest()[:8]}.json"
    for d in (SAM_CACHE, SAMPLES / "sam_cache"):
        if (d / key).exists():
            return json.loads((d / key).read_text())
    return None


def _sam_store(video: Path, concept: str, result: dict) -> None:
    SAM_CACHE.mkdir(parents=True, exist_ok=True)
    key = f"{_sha(video)}_{hashlib.sha256(concept.strip().lower().encode()).hexdigest()[:8]}.json"
    (SAM_CACHE / key).write_text(json.dumps(result))


def _samples() -> list[dict]:
    f = SAMPLES / "samples.json"
    return [s for s in json.loads(f.read_text()) if (SAMPLES / s["file"]).exists()] if f.exists() else []


def _replays() -> list[dict]:
    f = SAMPLES / "replays.json"
    return [r for r in json.loads(f.read_text()) if (RUNS / r["id"] / "report.json").exists()] if f.exists() else []


def _finite(o):
    """NaN and infinity are not JSON: browsers reject the whole response. Send null instead."""
    if isinstance(o, float):
        return o if o == o and abs(o) != float("inf") else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    return o


def _json(h: BaseHTTPRequestHandler, obj, code: int = 200) -> None:
    body = json.dumps(_finite(json.loads(json.dumps(obj, default=float))), allow_nan=False).encode()
    h.send_response(code)
    h.send_header("Content-Type", "application/json")
    h.send_header("Content-Length", str(len(body)))
    h.end_headers()
    h.wfile.write(body)


def _file(h: BaseHTTPRequestHandler, path: Path, ctype: str) -> None:
    if not path.exists():
        h.send_error(404)
        return
    size = path.stat().st_size
    start, end, code = 0, size - 1, 200
    rng = h.headers.get("Range")
    if rng and ctype.startswith("video/"):  # Safari needs byte ranges for video
        m = re.match(r"bytes=(\d*)-(\d*)$", rng.strip())
        if not m or not (m.group(1) or m.group(2)):
            h.send_error(416)
            return
        if m.group(1):
            start = int(m.group(1))
            end = min(int(m.group(2)) if m.group(2) else size - 1, size - 1)
        else:  # "bytes=-N": the last N bytes
            start, end = max(0, size - int(m.group(2))), size - 1
        if start > end:
            h.send_error(416)
            return
        code = 206
    h.send_response(code)
    if code == 206:
        h.send_header("Content-Range", f"bytes {start}-{end}/{size}")
    h.send_header("Content-Type", ctype)
    h.send_header("Content-Length", str(end - start + 1))
    h.send_header("Accept-Ranges", "bytes")
    h.end_headers()
    with path.open("rb") as f:  # stream only the requested bytes
        f.seek(start)
        left = end - start + 1
        while left > 0:
            chunk = f.read(min(1 << 20, left))
            if not chunk:
                break
            h.wfile.write(chunk)
            left -= len(chunk)


def _run_job(job_id: str, video: Path, spec: dict) -> None:
    job = JOBS[job_id]
    out = RUNS / job_id
    try:
        job.update(stage="tracking")
        members, names = [], set()
        def pos(v):  # lengths and masses: positive and finite, or not given
            return float(v) if v is not None and math.isfinite(float(v)) and float(v) > 0 else None

        def rng(v):  # priors: two positive, ordered numbers, or not given
            if not v or len(v) != 2:
                return None
            lo, hi = float(v[0]), float(v[1])
            return (lo, hi) if 0 < lo < hi and math.isfinite(hi) else None

        def pair(v):
            return (float(v[0]), float(v[1])) if v and len(v) == 2 else None

        for k, m in enumerate(spec["members"][:MAX_MEMBERS]):
            tr_range, f_range = rng(m.get("tension_range")), rng(m.get("f0_range"))
            name = str(m.get("name") or f"m{k + 1}")[:40]
            while name in names:  # names key every result: keep them unique
                name += "'"
            names.add(name)
            members.append(Member(name, pair(m["p0"]), pair(m["p1"]),
                                  half_width=min(max(float(m.get("half_width") or 10), 3.0), 80.0),
                                  length_m=pos(m.get("length_m")), mass_per_m=pos(m.get("mass_per_m")),
                                  tension_range=tr_range, f0_range=f_range))
        anchor = tuple(spec["anchor"]) if spec.get("anchor") else None
        tr = track(str(video), members, anchor=anchor, capture_fps=spec.get("capture_fps"))
        job.update(stage="analysing")
        from .analyze import picture_motion
        rep = analyze_tracking(tr, picture_motion_px=None if anchor else picture_motion(str(video), members))
        rep["video"] = str(video)
        for mem, src, e in zip(members, spec["members"], rep["members"]):  # the server's own verdicts only
            r = REVIEWS.get(_review_key(video.name, mem.p0, mem.p1))
            if r:
                e["visual_review"] = {**r, "name": mem.name}
            elif src.get("source") == "sam":  # proposed by SAM but never reviewed by this server
                e["visual_review"] = {"name": mem.name, "verdict": "unreviewed", "reason": "not reviewed by this server"}
        if spec.get("site_notes"):
            rep["site_notes"] = str(spec["site_notes"])[:1000]
        save(rep, out)
        job.update(stage="rendering", report=rep)
        from .render import render_hero
        render_hero(str(video), tr, rep, out / "hero.mp4", speed=min(max(float(spec.get("speed") or 0.25), 0.01), 1.0))
        job.update(stage="done", hero=f"/runs/{job_id}/hero.mp4")
    except Exception as e:  # noqa: BLE001
        job.update(stage="error", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-2000:])


def _run_inspect(job_id: str, context: str, mode: str, baseline_id: str | None = None) -> None:
    job = JOBS[job_id]
    try:
        job.update(inspect_stage="running")
        from .agent import inspect
        rep = json.loads((RUNS / job_id / "report.json").read_text())
        b = _load_run(baseline_id) if _safe(baseline_id) else None
        base = b["report"] if b else None
        md = inspect(rep, context, RUNS / job_id, mode=mode, log=lambda *_: None, baseline=base)
        if base:  # remembered, so the replay can show what changed
            (RUNS / job_id / "baseline.txt").write_text(Path(baseline_id).name)
            job["baseline"] = Path(baseline_id).name
        else:
            (RUNS / job_id / "baseline.txt").unlink(missing_ok=True)
            job.pop("baseline", None)
        job.update(inspect_stage="done", inspection=md.read_text(),
                   agent=json.loads((RUNS / job_id / "agent.json").read_text()))
    except Exception as e:  # noqa: BLE001
        job.update(inspect_stage="error", inspect_error=f"{type(e).__name__}: {e}")


def _load_run(job_id: str) -> dict | None:
    """A job from memory, or rebuilt from its run folder (so finished runs survive a server restart)."""
    if not _safe(job_id):
        return None
    if job_id in JOBS:
        return JOBS[job_id]
    d = RUNS / job_id
    if not (d / "report.json").exists():
        return None
    job = {"id": job_id, "stage": "done", "report": json.loads((d / "report.json").read_text())}
    if (d / "hero.mp4").exists():
        job["hero"] = f"/runs/{job_id}/hero.mp4"
    if (d / "agent.json").exists():
        job["agent"] = json.loads((d / "agent.json").read_text())
        job["inspect_stage"] = "done"
        if job["agent"].get("error"):
            job["inspect_error"] = job["agent"]["error"]
        job["inspection"] = (d / "inspection.md").read_text() if (d / "inspection.md").exists() else ""
    if (d / "approval.json").exists():
        job["approved"] = json.loads((d / "approval.json").read_text())
    if (d / "baseline.txt").exists():
        job["baseline"] = (d / "baseline.txt").read_text().strip()
    JOBS[job_id] = job
    return job


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet
        pass

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            return _file(self, STATIC / "index.html", "text/html; charset=utf-8")
        if u.path == "/api/frame":
            vid = _video(q.get("video", [""])[0])
            try:
                t = min(max(float(q.get("t", ["0.2"])[0]), 0.0), 3600.0)
            except ValueError:
                t = None
            if not vid or t is None:
                return _json(self, {"error": "unknown clip or bad time"}, 400)
            png = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(vid),
                                  "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
                                 capture_output=True).stdout
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(png)))
            self.end_headers()
            self.wfile.write(png)
            return
        if u.path == "/api/check":
            from .clipcheck import check
            vid = _video(q.get("video", [""])[0])
            try:
                fps = float(q["capture_fps"][0]) if q.get("capture_fps") else None
                if fps is not None and not (1 <= fps <= 100000):
                    raise ValueError
            except ValueError:
                return _json(self, {"error": "capture_fps must be a number of frames per second"}, 400)
            if not vid:
                return _json(self, {"error": "unknown clip"}, 404)
            try:
                return _json(self, check(str(vid), capture_fps=fps))
            except Exception as e:  # noqa: BLE001
                return _json(self, {"error": f"{type(e).__name__}: {e}"}, 500)
        if u.path == "/api/samples":
            return _json(self, {"samples": _samples(), "replays": _replays(), "public": PUBLIC})
        if u.path == "/api/runs":
            runs = sorted((p.name for p in RUNS.glob("*") if (p / "report.json").exists()), reverse=True)
            return _json(self, {"runs": runs[:50]})
        if u.path.startswith("/api/jobs/") and u.path.endswith("/diff"):
            job_id = u.path.split("/")[3]
            base_id = q.get("baseline", [""])[0]
            a, b = _load_run(base_id), _load_run(job_id)  # _load_run rejects unsafe ids
            if not a or not b:
                return _json(self, {"error": "unknown run"}, 404)
            from .fingerprint import describe, diff
            res = diff(a["report"], b["report"])
            for r in res["members"]:
                r["text"] = describe(r)
            return _json(self, res)
        if u.path.startswith("/api/jobs/"):
            job = _load_run(u.path.rsplit("/", 1)[-1])
            return _json(self, job or {"error": "unknown job"}, 200 if job else 404)
        if u.path.startswith("/runs/"):
            parts = u.path.split("/")  # "", runs, id, file
            if len(parts) == 4 and _safe(parts[2]) and _safe(parts[3]):
                p = RUNS / parts[2] / parts[3]
                ctype = {".mp4": "video/mp4", ".json": "application/json", ".jpg": "image/jpeg"}.get(
                    p.suffix, "text/markdown; charset=utf-8")
                return _file(self, p, ctype)
        self.send_error(404)

    def do_POST(self):
        u = urlparse(self.path)
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return _json(self, {"error": "bad Content-Length"}, 400)
        if u.path == "/api/upload":
            if n <= 0 or (PUBLIC and n > MAX_UPLOAD):
                return _json(self, {"error": f"send a clip of at most {MAX_UPLOAD >> 20} MB (trim it to 10 s)"}, 413)
            UPLOADS.mkdir(parents=True, exist_ok=True)
            stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(parse_qs(u.query).get("name", ["clip.mp4"])[0]).stem)[:40]
            name = f"{uuid.uuid4().hex[:6]}-{stem or 'clip'}.mp4"
            dest = UPLOADS / name
            with open(dest, "wb") as f:
                left = n
                while left > 0:
                    chunk = self.rfile.read(min(1 << 20, left))
                    if not chunk:
                        break
                    f.write(chunk)
                    left -= len(chunk)
            if not _looks_like_video(dest):
                dest.unlink(missing_ok=True)
                return _json(self, {"error": "not a video file this app reads (MP4/MOV, MKV/WebM or AVI)"}, 415)
            try:
                info = probe(str(dest))
            except Exception:  # noqa: BLE001
                dest.unlink(missing_ok=True)
                return _json(self, {"error": "the video could not be read"}, 415)
            _housekeep()
            return _json(self, {"video": name, "width": info.width, "height": info.height, "fps": info.fps})
        if not 0 <= n <= MAX_BODY:  # a negative length would read until the socket closes
            return _json(self, {"error": "request too large"}, 413)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(body, dict):
                raise ValueError
        except ValueError:
            return _json(self, {"error": "send a JSON object"}, 400)
        if u.path == "/api/sample":
            s = next((x for x in _samples() if x["id"] == body.get("id")), None)
            if not s:
                return _json(self, {"error": "unknown sample"}, 404)
            UPLOADS.mkdir(parents=True, exist_ok=True)
            dest = UPLOADS / f"sample-{s['id']}.mp4"
            if not dest.exists():
                shutil.copyfile(SAMPLES / s["file"], dest)
            info = probe(str(dest))
            return _json(self, {"video": dest.name, "width": info.width, "height": info.height, "fps": info.fps,
                                "sample": s})
        if u.path == "/api/analyze":
            video = _video(body.get("video"))
            if not video or not isinstance(body.get("members"), list) or not body["members"]:
                return _json(self, {"error": "unknown clip or no members"}, 400)
            if PUBLIC and _running("stage") >= MAX_RUNNING:
                return _json(self, {"error": "the demo server is busy measuring other clips: try again in a minute"}, 429)
            _housekeep()
            job_id = time.strftime("%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
            JOBS[job_id] = {"id": job_id, "stage": "queued", "video": body["video"]}
            threading.Thread(target=_run_job, args=(job_id, video, body), daemon=True).start()
            return _json(self, {"job": job_id})
        if u.path == "/api/detect":
            try:
                from . import sam_local
                from .video import read_frames
                vid = _video(body.get("video"))
                if not vid:
                    return _json(self, {"error": "unknown clip"}, 404)
                concept = str(body.get("concept", "rope"))[:60]
                cached = _sam_cached(vid, concept)
                if cached:
                    was = cached.get("backend", "SAM 3.1").replace("on this computer", "on-device")
                    cached["backend"] = f"{was}, result saved from an earlier run"
                    return _json(self, cached)
                frame = next(iter(read_frames(str(vid))))
                if sam_local.available():  # on-device SAM 3.1 (MLX): real masks
                    ms, extra = sam_local.find_members(frame, concept)
                    backend = "SAM 3.1 on this computer (MLX)"
                elif os.environ.get("MODEL_API_KEY"):  # Meta Model API: boxes
                    if not _model_call_allowed():
                        return _json(self, {"error": "model call limit reached for this hour on the public demo"}, 429)
                    from .sam import find_members
                    ms, extra, backend = find_members(frame, concept), [], "SAM 3.1 via Meta Model API"
                else:
                    return _json(self, {"error": "SAM 3.1 runs on-device (Apple silicon) in the field kit and is not "
                                                 "available on this server; it was run in advance for the sample "
                                                 "clips. Click the two ends of each cable instead."}, 503)
                info = {e["name"]: e for e in extra}
                result = {"backend": backend, "members": [
                    {"name": m.name, "p0": [float(v) for v in m.p0], "p1": [float(v) for v in m.p1],
                     "half_width": m.half_width, **{k: v for k, v in info.get(m.name, {}).items() if k != "name"}}
                    for m in ms]}
                _sam_store(vid, concept, result)
                return _json(self, result)
            except Exception as e:  # noqa: BLE001
                return _json(self, {"error": f"{type(e).__name__}: {e}"}, 502)
        if u.path == "/api/review":
            if not _client_allowed(self.client_address[0], "review"):
                return _json(self, {"error": "review limit for this hour reached on the public demo"}, 429)
            vid = _video(body.get("video"))
            if not vid or not isinstance(body.get("members"), list):
                return _json(self, {"error": "unknown clip or no members"}, 400)
            if len(body["members"]) > MAX_REVIEW:
                return _json(self, {"error": f"at most {MAX_REVIEW} candidates per review"}, 400)
            try:
                from .review import review
                from .video import read_frames
                frame = next(iter(read_frames(str(vid))))
                ms = [Member(str(m["name"])[:40], tuple(float(v) for v in m["p0"])[:2],
                             tuple(float(v) for v in m["p1"])[:2]) for m in body["members"]]
                res = review(frame, ms, context=str(body.get("context", ""))[:1000])
                for m, r in zip(ms, res):  # kept on the server: a measurement only trusts verdicts it made itself
                    REVIEWS[_review_key(vid.name, m.p0, m.p1)] = r
                return _json(self, {"reviews": res})
            except Exception as e:  # noqa: BLE001
                return _json(self, {"error": f"{type(e).__name__}: {e}"}, 502)
        if u.path.startswith("/api/jobs/") and u.path.endswith("/inspect"):
            job_id = u.path.split("/")[3]
            job = _load_run(job_id)
            if not job or not (RUNS / job_id / "report.json").exists():
                return _json(self, {"error": "unknown run"}, 404)
            if PUBLIC and job_id in SEEDED:
                return _json(self, {"error": "this is a recorded example: measure your own clip to run an inspection"}, 403)
            if job.get("inspect_stage") == "running":
                return _json(self, {"error": "an inspection of this run is already running"}, 409)
            if PUBLIC and _running("inspect_stage") >= MAX_RUNNING:
                return _json(self, {"error": "the demo server is busy: try again in a minute"}, 429)
            if not _client_allowed(self.client_address[0], "inspect"):
                return _json(self, {"error": "inspection limit for this hour reached on the public demo"}, 429)
            mode = "field" if body.get("mode") == "field" else "bench"
            job["inspect_stage"] = "running"
            threading.Thread(target=_run_inspect, args=(job_id, str(body.get("context", ""))[:2000], mode,
                                                        body.get("baseline")), daemon=True).start()
            return _json(self, {"ok": True})
        if u.path.startswith("/api/jobs/") and u.path.endswith("/approve"):
            job_id = u.path.split("/")[3]
            if not _load_run(job_id):
                return _json(self, {"error": "unknown run"}, 404)
            if PUBLIC and job_id in SEEDED:
                return _json(self, {"error": "this is a recorded example and cannot be signed"}, 403)
            by = str(body.get("by", "")).strip()[:80]
            if not by:
                return _json(self, {"error": "enter the engineer's name"}, 400)
            try:
                from .agent import approve
                md = approve(RUNS / job_id, by)
                JOBS[job_id]["inspection"] = md.read_text()
                JOBS[job_id]["approved"] = json.loads((RUNS / job_id / "approval.json").read_text())
                return _json(self, {"ok": True, "approval": JOBS[job_id]["approved"]})
            except Exception as e:  # noqa: BLE001
                return _json(self, {"error": str(e)}, 400)
        self.send_error(404)


def serve(port: int = 8765, host: str = "127.0.0.1") -> None:
    from . import llm
    llm.load_env()
    llm.CALL_GATE = _model_call_allowed  # every model call is charged to the hourly budget
    RUNS.mkdir(parents=True, exist_ok=True)
    for d in (SAMPLES / "runs").glob("*/") if (SAMPLES / "runs").exists() else []:
        if (d / "report.json").exists():  # recorded example runs: restored on every start, read-only in public mode
            SEEDED.add(d.name)
            if PUBLIC or not (RUNS / d.name).exists():
                shutil.rmtree(RUNS / d.name, ignore_errors=True)
                shutil.copytree(d, RUNS / d.name)
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"Resonance on http://{host}:{port}")
    if host != "127.0.0.1":
        print("  no login on this server: open it only on a network you trust")
    srv.serve_forever()
