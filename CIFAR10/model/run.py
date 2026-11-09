"""CIFAR10 classifier served by Cog. One file, two models:

    cog build -f cog.baseline.yaml  -> BaselineRunner  (tutorial hyperparameters)
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
from torchvision import transforms

from networks import CLASSES, IMAGE_SIZE, MEAN, STD, build_network

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


def to_cifar10(img):
    """Makes a photo look like a CIFAR10 image: RGB, square, 32x32 (the object roughly fills the frame)."""
    img = ImageOps.exif_transpose(img).convert("RGB")  # phone photos store their rotation separately
    side = min(img.size)
    img = ImageOps.fit(img, (side, side), method=Image.LANCZOS)  # center crop to a square
    return img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)


def png_data_uri(img, scale=4):
    """The 32x32 model input, enlarged, so the website can show what the model actually saw."""
    buffer = io.BytesIO()
    img.resize((img.width * scale, img.height * scale), Image.NEAREST).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


class CifarRunner(BaseRunner):
    variant = None  # "baseline" or "optimized", set by the subclasses below

    def setup(self):
        """Load the model once when the container starts, not on every prediction"""
        self.log = create_logger(self.variant)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # The checkpoint holds the hyperparameters and the learned weights (state_dict), see training.save_model
        checkpoint = torch.load(WEIGHTS_DIR / f"{self.variant}.pth", map_location=self.device, weights_only=True)
        config = checkpoint["config"]
        self.model = build_network(config).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()  # inference mode: dropout off

        metrics = json.loads((WEIGHTS_DIR / f"{self.variant}.json").read_text())
        self.info = {
            "name": self.variant,
            "test_accuracy": metrics["test_accuracy"],
            "parameters": metrics["parameters"],
            "config": config,
        }
        # Same transform as in training: tensor 0..1, then normalized to -1..1
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
        self.log.info("Model loaded on %s, test accuracy %.2f%%", self.device, metrics["test_accuracy"] * 100)

    def run(self, image: Path = Input(description="Photo of a plane, car, bird, cat, deer, dog, frog, horse, ship or truck")) -> dict:
        """Return the class, its probability and the probability of every class"""
        original = Image.open(image)
        model_input = to_cifar10(original)

        # no_grad: inference only, so PyTorch doesn't keep the data needed for training
        with torch.no_grad():
            logits = self.model(self.transform(model_input).unsqueeze(0).to(self.device))  # add batch dimension
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


class BaselineRunner(CifarRunner):
    variant = "baseline"


class OptimizedRunner(CifarRunner):
    variant = "optimized"
