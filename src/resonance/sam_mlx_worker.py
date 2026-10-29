"""SAM 3.1 on Apple silicon (MLX), run in its own Python environment.

Usage: python sam_mlx_worker.py <image.png> <noun phrase> <model path or hub id> <out.npz> [score_threshold]
Writes boxes (N, 4 xyxy), scores (N,) and masks (N, H, W, bool) at the image's own size.
Kept free of Resonance imports so it runs in the environment that has mlx-vlm and nothing else.
"""
import sys

import numpy as np
from PIL import Image


def main():
    image, prompt, model_id, out = sys.argv[1:5]
    threshold = float(sys.argv[5]) if len(sys.argv) > 5 else 0.3
    from mlx_vlm.models.sam3.generate import Sam3Predictor
    from mlx_vlm.models.sam3_1.processing_sam3_1 import Sam31Processor
    from mlx_vlm.utils import get_model_path, load_model

    path = get_model_path(model_id)
    model = load_model(path)
    predictor = Sam3Predictor(model, Sam31Processor.from_pretrained(str(path)), score_threshold=threshold)
    img = Image.open(image).convert("RGB")
    r = predictor.predict(img, text_prompt=prompt)
    w, h = img.size
    masks = []
    for m in np.asarray(r.masks):
        m = np.asarray(m) > 0.5 if m.dtype != bool else m
        if m.shape != (h, w):  # masks can come back at the model's resolution
            m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 127
        masks.append(m)
    np.savez_compressed(out, boxes=np.asarray(r.boxes, dtype=np.float32).reshape(-1, 4),
                        scores=np.asarray(r.scores, dtype=np.float32).reshape(-1),
                        masks=np.array(masks, dtype=bool).reshape(-1, h, w))
    print(f"{len(masks)} masks for '{prompt}'")


if __name__ == "__main__":
    main()
