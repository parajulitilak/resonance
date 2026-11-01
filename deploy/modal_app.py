"""Hosted demo of Resonance on Modal.

    RESONANCE_SAMPLES_DIR=/path/to/samples modal deploy deploy/modal_app.py

The samples folder holds samples.json, the clips it lists, sam_cache/ (SAM 3.1 results computed on-device for
those clips) and runs/ (finished runs to replay). Model keys come from the Modal secret `resonance-keys`
(BASETEN_API_KEY, and MODEL_API_KEY if the Meta Model API is used). One container serves everything, so the
in-memory job table stays consistent; uploads and runs live on a Modal volume.
"""

import os
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = os.environ.get("RESONANCE_SAMPLES_DIR")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("numpy>=1.26", "scipy>=1.11", "pillow>=10")
    .env({"PYTHONPATH": "/app/src", "RESONANCE_PUBLIC": "1", "RESONANCE_SAMPLES": "/app/samples"})
    .add_local_dir(ROOT / "src", "/app/src")
)
if SAMPLES:
    image = image.add_local_dir(SAMPLES, "/app/samples")

app = modal.App("resonance", image=image)
data = modal.Volume.from_name("resonance-data", create_if_missing=True)


@app.function(secrets=[modal.Secret.from_name("resonance-keys")], volumes={"/data": data},
              cpu=4.0, memory=8192, timeout=24 * 3600, max_containers=1, scaledown_window=15 * 60)
@modal.concurrent(max_inputs=100)
@modal.web_server(8000, startup_timeout=120)
def web():
    subprocess.Popen(["python", "-m", "resonance.cli", "serve", "--host", "0.0.0.0", "--port", "8000"], cwd="/data")
