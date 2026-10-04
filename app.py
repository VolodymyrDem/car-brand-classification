"""Gradio demo: upload a car photo and see what ResNet18 and ViT-Small predict.

Usage: python app.py   (then open http://127.0.0.1:7860)
"""
import gradio as gr

from src.predict import Predictor

predictor = Predictor()


def classify(image):
    if image is None:
        return [None] * 3
    result = predictor.predict(image)
    return [result.get(name) for name in ("ResNet18", "ViT-Small", "Ensemble")]


with gr.Blocks(title="Car make classifier") as demo:
    gr.Markdown(
        "# Car make classifier\n"
        "Upload a photo of a car. The models know 20 makes: "
        + ", ".join(predictor.classes) + "."
    )
    with gr.Row():
        image = gr.Image(type="pil", label="Photo", height=360)
        with gr.Column():
            outputs = [gr.Label(num_top_classes=5, label=name) for name in ("ResNet18", "ViT-Small", "Ensemble (average)")]
    image.change(classify, image, outputs)

if __name__ == "__main__":
    demo.launch()
