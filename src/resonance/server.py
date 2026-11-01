"""Local web app: upload a clip, mark (or auto-detect) the cables, measure, inspect, approve.

Standard library only (http.server). Runs on 127.0.0.1; heavy work happens in background threads
and the page polls for status.
"""

from __future__ import annotations

import json
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
STATIC = Path(__file__).parent / "static"
JOBS: dict[str, dict] = {}


def _json(h: BaseHTTPRequestHandler, obj, code: int = 200) -> None:
    body = json.dumps(obj, default=float).encode()
    h.send_response(code)
    h.send_header("Content-Type", "application/json")
    h.send_header("Content-Length", str(len(body)))
    h.end_headers()
    h.wfile.write(body)


def _file(h: BaseHTTPRequestHandler, path: Path, ctype: str) -> None:
    if not path.exists():
        h.send_error(404)
        return
    data = path.read_bytes()
    rng = h.headers.get("Range")
    if rng and ctype.startswith("video/"):  # Safari needs byte ranges for video
        start, _, end = rng.replace("bytes=", "").partition("-")
        start = int(start or 0)
        end = int(end) if end else len(data) - 1
        chunk = data[start:end + 1]
        h.send_response(206)
        h.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        h.send_header("Content-Type", ctype)
        h.send_header("Content-Length", str(len(chunk)))
        h.send_header("Accept-Ranges", "bytes")
        h.end_headers()
        h.wfile.write(chunk)
        return
    h.send_response(200)
    h.send_header("Content-Type", ctype)
    h.send_header("Content-Length", str(len(data)))
    h.send_header("Accept-Ranges", "bytes")
    h.end_headers()
    h.wfile.write(data)


def _run_job(job_id: str, video: Path, spec: dict) -> None:
    job = JOBS[job_id]
    out = RUNS / job_id
    try:
        job.update(stage="tracking")
        members = []
        for k, m in enumerate(spec["members"]):
            tr_range = tuple(m["tension_range"]) if m.get("tension_range") else None
            members.append(Member(m.get("name") or f"m{k + 1}", tuple(m["p0"]), tuple(m["p1"]),
                                  half_width=float(m.get("half_width", 10)), length_m=m.get("length_m"),
                                  mass_per_m=m.get("mass_per_m"), tension_range=tr_range))
        anchor = tuple(spec["anchor"]) if spec.get("anchor") else None
        tr = track(str(video), members, anchor=anchor, capture_fps=spec.get("capture_fps"))
        job.update(stage="analysing")
        rep = analyze_tracking(tr)
        rep["video"] = str(video)
        reviews = {m.get("name"): m["review"] for m in spec["members"] if m.get("review")}
        for e in rep["members"]:  # the visual review travels with the member, so the inspector sees it
            if e["name"] in reviews:
                e["visual_review"] = reviews[e["name"]]
        if spec.get("site_notes"):
            rep["site_notes"] = spec["site_notes"][:1000]
        save(rep, out)
        job.update(stage="rendering", report=rep)
        from .render import render_hero
        render_hero(str(video), tr, rep, out / "hero.mp4", speed=float(spec.get("speed", 0.25)))
        job.update(stage="done", hero=f"/runs/{job_id}/hero.mp4")
    except Exception as e:  # noqa: BLE001
        job.update(stage="error", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-2000:])


def _run_inspect(job_id: str, context: str, mode: str, baseline_id: str | None = None) -> None:
    job = JOBS[job_id]
    try:
        job.update(inspect_stage="running")
        from .agent import inspect
        rep = json.loads((RUNS / job_id / "report.json").read_text())
        base = _load_run(baseline_id)["report"] if baseline_id and _load_run(baseline_id) else None
        md = inspect(rep, context, RUNS / job_id, mode=mode, log=lambda *_: None, baseline=base)
        job.update(inspect_stage="done", inspection=md.read_text(),
                   agent=json.loads((RUNS / job_id / "agent.json").read_text()))
    except Exception as e:  # noqa: BLE001
        job.update(inspect_stage="error", inspect_error=f"{type(e).__name__}: {e}")


def _load_run(job_id: str) -> dict | None:
    """A job from memory, or rebuilt from its run folder (so finished runs survive a server restart)."""
    if job_id in JOBS:
        return JOBS[job_id]
    d = RUNS / Path(job_id).name
    if not (d / "report.json").exists():
        return None
    job = {"id": job_id, "stage": "done", "report": json.loads((d / "report.json").read_text())}
    if (d / "hero.mp4").exists():
        job["hero"] = f"/runs/{job_id}/hero.mp4"
    if (d / "agent.json").exists():
        job["agent"] = json.loads((d / "agent.json").read_text())
        job["inspect_stage"] = "done"
        job["inspection"] = (d / "inspection.md").read_text() if (d / "inspection.md").exists() else ""
    if (d / "approval.json").exists():
        job["approved"] = json.loads((d / "approval.json").read_text())
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
            vid = UPLOADS / Path(q["video"][0]).name
            png = subprocess.run(["ffmpeg", "-v", "error", "-ss", q.get("t", ["0.2"])[0], "-i", str(vid),
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
            vid = UPLOADS / Path(q["video"][0]).name
            fps = float(q["capture_fps"][0]) if q.get("capture_fps") else None
            try:
                return _json(self, check(str(vid), capture_fps=fps))
            except Exception as e:  # noqa: BLE001
                return _json(self, {"error": f"{type(e).__name__}: {e}"}, 500)
        if u.path == "/api/runs":
            runs = sorted((p.name for p in RUNS.glob("*") if (p / "report.json").exists()), reverse=True)
            return _json(self, {"runs": runs[:50]})
        if u.path.startswith("/api/jobs/") and u.path.endswith("/diff"):
            job_id = u.path.split("/")[3]
            base_id = q.get("baseline", [""])[0]
            a, b = _load_run(base_id), _load_run(job_id)
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
            parts = Path(u.path).parts  # /, runs, id, file
            if len(parts) == 4:
                p = RUNS / Path(parts[2]).name / Path(parts[3]).name
                ctype = "video/mp4" if p.suffix == ".mp4" else "application/json" if p.suffix == ".json" \
                    else "text/markdown; charset=utf-8"
                return _file(self, p, ctype)
        self.send_error(404)

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length", "0"))
        if u.path == "/api/upload":
            UPLOADS.mkdir(parents=True, exist_ok=True)
            name = f"{uuid.uuid4().hex[:10]}.mp4" if not u.query else \
                f"{uuid.uuid4().hex[:6]}-{Path(parse_qs(u.query).get('name', ['clip.mp4'])[0]).stem[:40]}.mp4"
            dest = UPLOADS / name
            with open(dest, "wb") as f:
                left = n
                while left > 0:
                    chunk = self.rfile.read(min(1 << 20, left))
                    if not chunk:
                        break
                    f.write(chunk)
                    left -= len(chunk)
            info = probe(str(dest))
            return _json(self, {"video": name, "width": info.width, "height": info.height, "fps": info.fps})
        body = json.loads(self.rfile.read(n) or b"{}")
        if u.path == "/api/analyze":
            video = UPLOADS / Path(body["video"]).name
            job_id = time.strftime("%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
            JOBS[job_id] = {"id": job_id, "stage": "queued", "video": body["video"]}
            threading.Thread(target=_run_job, args=(job_id, video, body), daemon=True).start()
            return _json(self, {"job": job_id})
        if u.path == "/api/detect":
            try:
                from . import sam_local
                from .video import read_frames
                vid = UPLOADS / Path(body["video"]).name
                frame = next(iter(read_frames(str(vid))))
                concept = body.get("concept", "rope")
                if sam_local.available():  # on-device SAM 3.1 (MLX): real masks
                    ms, extra = sam_local.find_members(frame, concept)
                    backend = "SAM 3.1 on this computer (MLX)"
                else:  # Meta Model API: boxes
                    from .sam import find_members
                    ms, extra, backend = find_members(frame, concept), [], "SAM 3.1 via Meta Model API"
                info = {e["name"]: e for e in extra}
                return _json(self, {"backend": backend, "members": [
                    {"name": m.name, "p0": [float(v) for v in m.p0], "p1": [float(v) for v in m.p1],
                     "half_width": m.half_width, **{k: v for k, v in info.get(m.name, {}).items() if k != "name"}}
                    for m in ms]})
            except Exception as e:  # noqa: BLE001
                return _json(self, {"error": f"{type(e).__name__}: {e}"}, 502)
        if u.path == "/api/review":
            try:
                from .review import review
                from .video import read_frames
                frame = next(iter(read_frames(str(UPLOADS / Path(body["video"]).name))))
                ms = [Member(m["name"], tuple(m["p0"]), tuple(m["p1"])) for m in body["members"]]
                return _json(self, {"reviews": review(frame, ms, context=body.get("context", ""))})
            except Exception as e:  # noqa: BLE001
                return _json(self, {"error": f"{type(e).__name__}: {e}"}, 502)
        if u.path.startswith("/api/jobs/") and u.path.endswith("/inspect"):
            job_id = u.path.split("/")[3]
            _load_run(job_id)
            threading.Thread(target=_run_inspect, args=(job_id, body.get("context", ""), body.get("mode", "bench"),
                                                        body.get("baseline")), daemon=True).start()
            return _json(self, {"ok": True})
        if u.path.startswith("/api/jobs/") and u.path.endswith("/approve"):
            job_id = u.path.split("/")[3]
            try:
                from .agent import approve
                _load_run(job_id)
                md = approve(RUNS / job_id, body["by"])
                JOBS[job_id]["inspection"] = md.read_text()
                JOBS[job_id]["approved"] = json.loads((RUNS / job_id / "approval.json").read_text())
                return _json(self, {"ok": True, "approval": JOBS[job_id]["approved"]})
            except Exception as e:  # noqa: BLE001
                return _json(self, {"error": str(e)}, 400)
        self.send_error(404)


def serve(port: int = 8765, host: str = "127.0.0.1") -> None:
    from .llm import load_env
    load_env()
    RUNS.mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"Resonance on http://{host}:{port}")
    if host != "127.0.0.1":
        print("  no login on this server: open it only on a network you trust")
    srv.serve_forever()
