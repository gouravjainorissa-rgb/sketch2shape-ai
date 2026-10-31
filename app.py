"""Gradio web app: draw or upload a sketch, add an optional prompt, get a .glb.

Run: python app.py   (in Colab, share=True prints a public link)
"""
import argparse

import gradio as gr
import numpy as np
from PIL import Image

from sketch2shape.config import load_config
from sketch2shape.pipeline import Pipeline, PipelineError

cfg = load_config()
pipe = Pipeline(cfg)


def _canvas_to_image(value):
    """Gradio's sketch component returns a dict with 'composite' (or a plain array)."""
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("composite")
    if value is None:
        return None
    return Image.fromarray(np.asarray(value).astype("uint8"))


def generate(drawn, uploaded, prompt, strength, seed, steps, progress=gr.Progress()):
    sketch = uploaded if uploaded is not None else _canvas_to_image(drawn)
    if sketch is None:
        raise gr.Error("Draw a sketch or upload one first.")
    try:
        r = pipe.run(
            sketch,
            prompt=prompt,
            seed=int(seed),
            sketch_strength=float(strength),
            steps=int(steps),
            progress=lambda m: progress(0, desc=m),
        )
    except PipelineError as err:
        raise gr.Error(str(err))
    report = {
        "success": r["success"],
        "fidelity": r["fidelity"],
        "validation": r["validation"],
        "timings_sec": r["timings_sec"],
        "total_sec": r["total_sec"],
        "hardware": r["hardware"],
    }
    return r["generated_image"], r["glb"], [r["glb"], r["obj"]], report


with gr.Blocks(title="Sketch2Shape AI") as demo:
    gr.Markdown(
        "# Sketch2Shape AI\n"
        "**1.** Draw or upload a sketch &nbsp; **2.** Add an optional prompt &nbsp; **3.** Generate and download"
    )
    with gr.Row():
        with gr.Column():
            with gr.Tab("Draw"):
                drawn = gr.Sketchpad(label="Draw here", type="numpy", canvas_size=(512, 512))
            with gr.Tab("Upload"):
                uploaded = gr.Image(label="Upload a sketch", type="pil")
            prompt = gr.Textbox(
                label="Prompt (optional)",
                placeholder="e.g. a red ceramic mug, glossy. Leave empty to let the model decide.",
            )
            with gr.Accordion("Advanced", open=False):
                strength = gr.Slider(0.5, 2.0, value=cfg["generation"]["sketch_strength"], step=0.1, label="Sketch strength")
                seed = gr.Number(value=cfg["seed"], precision=0, label="Seed")
                steps = gr.Slider(10, 40, value=cfg["generation"]["steps"], step=1, label="Steps")
            go = gr.Button("Generate 3D model", variant="primary")
        with gr.Column():
            ref_img = gr.Image(label="Generated reference image")
            viewer = gr.Model3D(label="3D preview (drag to rotate)")
            files = gr.File(label="Download", file_count="multiple")
            report = gr.JSON(label="Validation and timing report")
    go.click(generate, [drawn, uploaded, prompt, strength, seed, steps], [ref_img, viewer, files, report])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="Create a public link (needed in Colab)")
    args = ap.parse_args()
    demo.queue().launch(share=args.share)
