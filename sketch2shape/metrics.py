"""Timing, GPU memory and sketch-fidelity helpers."""
from __future__ import annotations

import time
from contextlib import contextmanager

import cv2
import numpy as np
from PIL import Image


class StageTimer:
    """Collects per-stage wall-clock time: `with timer.stage("generate"): ...`"""

    def __init__(self) -> None:
        self.times: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.times[name] = round(time.perf_counter() - start, 2)

    @property
    def total(self) -> float:
        return round(sum(self.times.values()), 2)


def gpu_info() -> dict:
    """GPU name and peak memory so far. Safe to call without torch or CUDA."""
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {
                "gpu": props.name,
                "vram_gb": round(props.total_memory / 1024**3, 1),
                "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
            }
    except Exception:
        pass
    return {"gpu": None, "vram_gb": None, "peak_mem_gb": None}


def fidelity_score(sketch_map: Image.Image, generated: Image.Image) -> float:
    """How well the generated image's edges line up with the sketch lines (0 to 1).

    Share of the sketch's line pixels that have a generated-image edge within a
    few pixels. Simple and fast; used to decide whether to retry generation.
    """
    sketch = np.array(sketch_map.convert("L").resize(generated.size))
    sketch_lines = (sketch > 127).astype(np.uint8)
    if sketch_lines.sum() == 0:
        return 0.0

    gen_gray = cv2.cvtColor(np.array(generated.convert("RGB")), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gen_gray, 60, 140)
    edges = cv2.dilate(edges, np.ones((7, 7), np.uint8), iterations=1)

    hit = np.logical_and(sketch_lines > 0, edges > 0).sum()
    return round(float(hit) / float(sketch_lines.sum()), 4)
