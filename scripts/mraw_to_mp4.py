"""Convert a Photron .mraw (with its .cihx header) to an H.264 MP4 that Resonance can read.

Usage: python scripts/mraw_to_mp4.py <file.cihx> <file.mraw> <out.mp4> [--max-frames N]
Prints the record rate, which must be passed to `--capture-fps` (the MP4 stores it, but say it explicitly).
"""
import re
import subprocess
import sys

import numpy as np


def header(cihx: str) -> dict:
    text = open(cihx, "rb").read().decode("utf-8", "ignore")
    xml = text[text.find("<cih>"):] if "<cih>" in text else text

    def tag(name):
        m = re.search(rf"<{name}>\s*([^<]+?)\s*</{name}>", xml)
        return m.group(1) if m else None

    return {"fps": float(tag("recordRate")), "width": int(tag("width")), "height": int(tag("height")),
            "bits": int(tag("bit") or tag("effectiveBit") or 12), "frames": int(tag("totalFrame") or 0)}


def unpack12(buf: bytes) -> np.ndarray:
    b = np.frombuffer(buf, np.uint8).reshape(-1, 3).astype(np.uint16)
    p0 = (b[:, 0] << 4) | (b[:, 1] >> 4)
    p1 = ((b[:, 1] & 0x0F) << 8) | b[:, 2]
    return np.stack([p0, p1], 1).reshape(-1)


def main():
    cihx, mraw, out = sys.argv[1:4]
    max_frames = int(sys.argv[sys.argv.index("--max-frames") + 1]) if "--max-frames" in sys.argv else None
    h = header(cihx)
    w, ht, bits = h["width"], h["height"], h["bits"]
    print(f"record rate {h['fps']:g} fps, {w}x{ht}, {bits}-bit, {h['frames']} frames in header")
    nbytes = w * ht * bits // 8
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{ht}",
                          "-r", f"{h['fps']:g}", "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "8",
                          "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    lo, hi = None, None
    n = 0
    with open(mraw, "rb") as f:
        while max_frames is None or n < max_frames:
            buf = f.read(nbytes)
            if len(buf) < nbytes:
                break
            img = unpack12(buf) if bits == 12 else np.frombuffer(buf, np.uint16 if bits == 16 else np.uint8)
            img = img.reshape(ht, w).astype(np.float32)
            if lo is None:  # fix the contrast stretch on the first frame so brightness doesn't pump
                lo, hi = np.percentile(img, 0.5), np.percentile(img, 99.5)
            p.stdin.write(np.clip((img - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8).tobytes())
            n += 1
    p.stdin.close()
    p.wait()
    print(f"wrote {n} frames to {out}")


if __name__ == "__main__":
    main()
