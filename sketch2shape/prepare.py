"""Stage 4: background removal, crop, centre and pad for reconstruction."""
from __future__ import annotations

import numpy as np
from PIL import Image

_session = None


def _get_session():
    global _session
    if _session is None:
        from rembg import new_session

        _session = new_session("isnet-general-use")
    return _session


def remove_background(img: Image.Image) -> Image.Image:
    """Return an RGBA cut-out. Falls back to a white-background threshold if rembg fails."""
    try:
        from rembg import remove

        out = remove(img.convert("RGB"), session=_get_session())
        if np.array(out)[..., 3].max() > 0 and (np.array(out)[..., 3] > 0).mean() > 0.01:
            return out.convert("RGBA")
    except Exception:
        pass
    # Fallback: treat near-white pixels as background (we prompt for a white background).
    rgb = np.array(img.convert("RGB"))
    mask = (rgb.min(axis=2) < 235).astype(np.uint8) * 255
    return Image.fromarray(np.dstack([rgb, mask]), "RGBA")


def frame_object(rgba: Image.Image, ratio: float = 0.85, size: int = 512) -> Image.Image:
    """Crop to the object, centre it, pad so it fills `ratio` of the frame."""
    arr = np.array(rgba)
    ys, xs = np.where(arr[..., 3] > 10)
    if len(ys) == 0:
        raise ValueError("No object found after background removal.")
    crop = Image.fromarray(arr[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1])
    scale = (size * ratio) / max(crop.size)
    crop = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))), Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(crop, ((size - crop.width) // 2, (size - crop.height) // 2), crop)
    return canvas


def prepare(img: Image.Image, ratio: float = 0.85, size: int = 512) -> Image.Image:
    return frame_object(remove_background(img), ratio=ratio, size=size)
