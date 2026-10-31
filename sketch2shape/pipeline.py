"""Chains all stages. The CLI and the Gradio app both call `Pipeline.run`."""
from __future__ import annotations

import json
import time
from pathlib import Path

from PIL import Image

from .config import load_config
from .metrics import StageTimer, fidelity_score, gpu_info
from .preprocess import SketchError, preprocess


class PipelineError(RuntimeError):
    """A failure with a message that is safe to show to the user."""


class Pipeline:
    def __init__(self, cfg: dict | None = None, low_vram: bool = False) -> None:
        self.cfg = cfg or load_config()
        self.low_vram = low_vram
        self._gen = None
        self._rec = None

    # Models load lazily and stay resident, so only the first run pays the cost.
    def _generator(self):
        if self._gen is None:
            from .generate import Generator

            self._gen = Generator(self.cfg, low_vram=self.low_vram)
        return self._gen

    def _reconstructor(self):
        if self._rec is None:
            from .reconstruct import get_reconstructor

            self._rec = get_reconstructor(self.cfg)
        return self._rec

    def run(
        self,
        sketch,
        prompt: str | None = None,
        out_dir: str | Path | None = None,
        seed: int | None = None,
        sketch_strength: float | None = None,
        steps: int | None = None,
        progress=None,
    ) -> dict:
        say = progress or (lambda msg: None)
        cfg, timer = self.cfg, StageTimer()
        out = Path(out_dir or Path(cfg["paths"]["outputs"]) / time.strftime("run-%Y%m%d-%H%M%S"))
        out.mkdir(parents=True, exist_ok=True)

        try:
            say("1/5 Cleaning sketch")
            with timer.stage("preprocess"):
                scribble, pre_info = preprocess(sketch, size=cfg["image_size"])
            try:
                original = Image.open(sketch) if isinstance(sketch, (str, Path)) else sketch
                original.save(out / "input_sketch.png")
            except Exception:
                pass  # saving the original is a convenience, never fatal
            scribble.save(out / "sketch_processed.png")

            say("2/5 Generating image from sketch")
            with timer.stage("generate"):
                gen = self._generator()
                image = gen.generate(scribble, prompt, seed, sketch_strength, steps)
                fidelity = fidelity_score(scribble, image)
                retried = False
                if fidelity < cfg["generation"]["min_fidelity"]:
                    retried = True
                    image = gen.generate(scribble, prompt, seed, cfg["generation"]["retry_strength"], steps)
                    fidelity = max(fidelity, fidelity_score(scribble, image))
            image.save(out / "generated.png")

            say("3/5 Removing background")
            from .prepare import prepare

            with timer.stage("prepare"):
                cutout = prepare(image, cfg["prepare"]["foreground_ratio"], cfg["image_size"])
            cutout.save(out / "cutout.png")

            say("4/5 Building 3D model")
            with timer.stage("reconstruct"):
                mesh = self._reconstructor().reconstruct(cutout)

            say("5/5 Cleaning, validating and exporting")
            from .mesh import clean_mesh, export_and_verify, validate_mesh

            with timer.stage("mesh"):
                mesh = clean_mesh(mesh, cfg["mesh"]["keep_largest_component_share"])
                validation = validate_mesh(mesh, cfg)
                export = export_and_verify(mesh, out, cfg)
        except SketchError as err:
            raise PipelineError(str(err)) from err
        except Exception as err:
            msg = str(err)
            if "out of memory" in msg.lower():
                msg = (
                    "The GPU ran out of memory. Restart the runtime, lower reconstruct.mc_resolution "
                    "or chunk_size in config.yaml, or enable low-VRAM mode."
                )
            raise PipelineError(msg) from err

        ok = bool(validation["passed"] and export["reload_ok"] and export["size_ok"])
        result = {
            "success": ok,
            "glb": str(out / "model.glb"),
            "obj": str(out / "model.obj"),
            "generated_image": str(out / "generated.png"),
            "prompt": prompt or "",
            "seed": cfg["seed"] if seed is None else seed,
            "fidelity": fidelity,
            "fidelity_retry_used": retried,
            "validation": validation,
            "export": export,
            "timings_sec": timer.times,
            "total_sec": timer.total,
            "preprocess": pre_info,
            "hardware": gpu_info(),
        }
        (out / "metadata.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (out / "validation_report.json").write_text(
            json.dumps({"validation": validation, "export": export}, indent=2), encoding="utf-8"
        )
        return result
