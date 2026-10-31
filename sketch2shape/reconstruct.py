"""Stage 5: image-to-3D reconstruction.

TripoSR is the safe baseline (light, installs easily, fits a free Colab T4).
InstantMesh is the planned upgrade; add it as a second class with the same
`.reconstruct(rgba) -> trimesh.Trimesh` interface and switch with config.yaml.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


class TripoSRReconstructor:
    def __init__(self, cfg: dict) -> None:
        import torch

        # TripoSR code is cloned into third_party/TripoSR by setup.sh
        repo = ROOT / "third_party" / "TripoSR"
        if not repo.exists():
            raise RuntimeError("third_party/TripoSR is missing. Run `bash setup.sh` first.")
        sys.path.insert(0, str(repo))
        from tsr.system import TSR  # noqa: E402

        r = cfg["reconstruct"]
        self.cfg = r
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = TSR.from_pretrained(
            r["triposr_repo"], config_name="config.yaml", weight_name="model.ckpt"
        )
        self.model.renderer.set_chunk_size(int(r["chunk_size"]))
        self.model.to(self.device)

    def reconstruct(self, rgba: Image.Image):
        """Return a trimesh.Trimesh with vertex colours."""
        # TripoSR expects the object composited on a mid-grey background.
        arr = np.array(rgba.convert("RGBA")).astype(np.float32) / 255.0
        rgb = arr[..., :3] * arr[..., 3:4] + (1 - arr[..., 3:4]) * 0.5
        image = Image.fromarray((rgb * 255).astype(np.uint8))

        with self.torch.no_grad():
            codes = self.model([image], device=self.device)
            meshes = self.model.extract_mesh(
                codes, has_vertex_color=True, resolution=int(self.cfg["mc_resolution"])
            )
        return meshes[0]


def get_reconstructor(cfg: dict):
    name = cfg["reconstruct"]["model"].lower()
    if name == "triposr":
        return TripoSRReconstructor(cfg)
    raise ValueError(f"Unknown reconstruction model '{name}'. Available: triposr")
