"""Garment classifier served by Cog. One file, two models:

    cog build -f cog.baseline.yaml  -> BaselineRunner  (Quickstart hyperparameters)
    cog build -f cog.optimized.yaml -> OptimizedRunner (hyperparameters found by tune.py)
"""
import base64
import io
import json
import logging
import os
import sys
from pathlib import Path as FilePath

import torch
from cog import BaseRunner, Input, Path
from PIL import Image, ImageOps
from torchvision.transforms import ToTensor

from networks import CLASSES, IMAGE_SIZE, NeuralNetwork

WEIGHTS_DIR = FilePath(__file__).parent / "weights"

# Every prediction is written to stdout (docker logs) and to this file (docker exec ... tail -f)
LOG_FILE = os.getenv("PREDICTION_LOG", "/tmp/predictions.log")


def create_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_FILE)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def to_fashion_mnist(img):
    """Makes a photo look like a FashionMNIST image: 28x28 grayscale, light garment on a black background,
    filling the frame. Without this, a photo of a white shirt on a white table means nothing to the model."""
    img = ImageOps.exif_transpose(img)  # phone photos store their rotation separately
    if img.mode in ("RGBA", "LA", "P"):
        # transparent background (e.g. product PNGs) -> white
        img = img.convert("RGBA")
        background = Image.new("RGBA", img.size, "white")
        img = Image.alpha_composite(background, img)
    gray = img.convert("L")

    if gray.size != (IMAGE_SIZE, IMAGE_SIZE):  # a real FashionMNIST image is used as it is
        # FashionMNIST garments are light on black: invert photos taken on a light background
        pixels = list(gray.getdata())
        w, h = gray.size
        border = [pixels[x] for x in range(w)] + [pixels[(h - 1) * w + x] for x in range(w)] + \
                 [pixels[y * w] for y in range(h)] + [pixels[y * w + w - 1] for y in range(h)]
        if sum(border) / len(border) > 127:
            gray = ImageOps.invert(gray)

        # Crop to the garment, then scale its longer side to 28 px and center it on black
        box = gray.point(lambda p: 255 if p > 40 else 0).getbbox()
        if box:
            gray = gray.crop(box)
        scale = IMAGE_SIZE / max(gray.size)
        size = (max(1, round(gray.width * scale)), max(1, round(gray.height * scale)))
        gray = gray.resize(size, Image.LANCZOS)
        canvas = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)
        canvas.paste(gray, ((IMAGE_SIZE - size[0]) // 2, (IMAGE_SIZE - size[1]) // 2))
        gray = canvas
    return gray


def png_data_uri(img, scale=4):
    """The 28x28 model input, enlarged, so the website can show what the model actually saw."""
    buffer = io.BytesIO()
    img.resize((img.width * scale, img.height * scale), Image.NEAREST).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


class GarmentRunner(BaseRunner):
    variant = None  # "baseline" or "optimized", set by the subclasses below

    def setup(self):
        """Load the model once when the container starts, not on every prediction"""
        self.log = create_logger(self.variant)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # The checkpoint holds the hyperparameters and the learned weights (state_dict), see training.save_model
        checkpoint = torch.load(WEIGHTS_DIR / f"{self.variant}.pth", map_location=self.device, weights_only=True)
        config = checkpoint["config"]
        self.model = NeuralNetwork(hidden_sizes=config["hidden_sizes"], dropout=config["dropout"]).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()  # inference mode: dropout off

        metrics = json.loads((WEIGHTS_DIR / f"{self.variant}.json").read_text())
        self.info = {
            "name": self.variant,
            "test_accuracy": metrics["test_accuracy"],
            "parameters": metrics["parameters"],
            "config": config,
        }
        self.to_tensor = ToTensor()  # PIL image -> 1x28x28 tensor with values 0..1, as in training
        self.log.info("Model loaded on %s, test accuracy %.2f%%", self.device, metrics["test_accuracy"] * 100)

    def run(self, image: Path = Input(description="Photo of a garment")) -> dict:
        """Return the garment class, its probability and the probability of every class"""
        original = Image.open(image)
        model_input = to_fashion_mnist(original)

        # no_grad: inference only, so PyTorch doesn't keep the data needed for training
        with torch.no_grad():
            logits = self.model(self.to_tensor(model_input).unsqueeze(0).to(self.device))  # add batch dimension
        probabilities = logits[0].softmax(0)  # logits -> probabilities that add up to 1

        # A list instead of a dict: Cog sorts JSON object keys alphabetically, a list keeps the ranking
        ranking = sorted(
            ({"label": label, "probability": p.item()} for label, p in zip(CLASSES, probabilities)),
            key=lambda item: item["probability"],
            reverse=True,
        )
        best = ranking[0]

        self.log.info("size=%dx%d prediction=%s confidence=%.3f second=%s (%.3f)",
                      original.width, original.height, best["label"], best["probability"],
                      ranking[1]["label"], ranking[1]["probability"])

        return {
            "label": best["label"],
            "confidence": best["probability"],
            "probabilities": ranking,
            "model_input": png_data_uri(model_input),
            "model": self.info,
        }


class BaselineRunner(GarmentRunner):
    variant = "baseline"


class OptimizedRunner(GarmentRunner):
    variant = "optimized"
