"""Saves one test-set image per class to ../test-images/ (e.g. sneaker.png), enlarged 8x so they are easy
to see, for trying out the website and Postman. These images were never seen during training.

Usage:  python export_samples.py
"""
from pathlib import Path

from PIL import Image
from torchvision import datasets

from networks import CLASSES
from training import DATA_DIR

OUT_DIR = Path(__file__).parent.parent / "test-images"


def file_name(label):
    return label.lower().replace("/", "-").replace(" ", "-") + ".png"  # "T-shirt/top" -> t-shirt-top.png


def main():
    OUT_DIR.mkdir(exist_ok=True)
    test_set = datasets.FashionMNIST(root=DATA_DIR, train=False, download=True)
    saved = set()
    for image, target in test_set:  # image is a 28x28 PIL image
        if target not in saved:
            image.resize((224, 224), Image.NEAREST).save(OUT_DIR / file_name(CLASSES[target]))
            saved.add(target)
        if len(saved) == len(CLASSES):
            break
    print(f"Saved {len(saved)} images to {OUT_DIR}")


if __name__ == "__main__":
    main()
