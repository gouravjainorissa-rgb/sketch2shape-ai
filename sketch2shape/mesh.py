"""Stage 6/7: mesh clean-up, normalisation, validation and export."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


def clean_mesh(mesh: trimesh.Trimesh, keep_share: float = 0.9) -> trimesh.Trimesh:
    """Drop floating fragments and degenerate faces, fix normals, centre and unit-scale."""
    mesh = mesh.copy()
    try:
        parts = mesh.split(only_watertight=False)
        if len(parts) > 1:
            parts = sorted(parts, key=lambda m: len(m.faces), reverse=True)
            total = sum(len(p.faces) for p in parts)
            kept, running = [], 0
            for p in parts:
                kept.append(p)
                running += len(p.faces)
                if running / total >= keep_share:
                    break
            mesh = trimesh.util.concatenate(kept)
    except Exception:
        pass  # splitting is a nice-to-have, never fatal

    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(mesh)

    # Centre at the origin and scale into a unit bounding box.
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    extent = float(mesh.extents.max())
    if extent > 0:
        mesh.apply_scale(1.0 / extent)
    return mesh


def _has_colour(mesh: trimesh.Trimesh) -> bool:
    visual = mesh.visual
    kind = getattr(visual, "kind", None)
    if kind == "texture":
        return getattr(visual.material, "image", None) is not None or getattr(
            visual.material, "baseColorTexture", None
        ) is not None
    if kind == "vertex":
        vc = np.asarray(visual.vertex_colors)
        return vc.size > 0 and float(vc[:, :3].std()) > 1.0
    return False


def validate_mesh(mesh: trimesh.Trimesh, cfg: dict) -> dict:
    """Run the SRS section 10.1 checks. Returns a report with an overall `passed`."""
    m = cfg["mesh"]
    verts = np.asarray(mesh.vertices)
    extents = np.sort(np.asarray(mesh.extents))
    flatness = float(extents[0] / extents[-1]) if extents[-1] > 0 else 0.0

    checks = {
        "non_empty": len(mesh.faces) >= m["min_faces"],
        "finite_values": bool(np.isfinite(verts).all()),
        "has_volume": flatness >= m["min_flatness_ratio"],
        "has_colour_or_texture": _has_colour(mesh),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "bbox_extents": [round(float(x), 4) for x in mesh.extents],
        "flatness_ratio": round(flatness, 4),
        "watertight": bool(mesh.is_watertight),
    }


def export_and_verify(mesh: trimesh.Trimesh, out_dir: str | Path, cfg: dict) -> dict:
    """Export .glb (primary) and .obj, then reload both to confirm they open."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    glb_path, obj_path = out / "model.glb", out / "model.obj"

    mesh.export(glb_path)
    mesh.export(obj_path)

    report = {"files": {}, "reload_ok": True}
    for path in (glb_path, obj_path):
        try:
            loaded = trimesh.load(path, force="mesh")
            ok = len(loaded.faces) > 0 and len(loaded.vertices) > 0
            report["files"][path.name] = {
                "size_mb": round(path.stat().st_size / 1024**2, 3),
                "faces": int(len(loaded.faces)),
                "opens": bool(ok),
            }
            report["reload_ok"] &= bool(ok)
        except Exception as exc:
            report["files"][path.name] = {"opens": False, "error": str(exc)}
            report["reload_ok"] = False

    size_ok = report["files"].get("model.glb", {}).get("size_mb", 1e9) <= cfg["mesh"]["max_file_mb"]
    report["size_ok"] = bool(size_ok)
    return report
