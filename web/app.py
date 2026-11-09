"""Classifier website: takes an uploaded image and asks two Cog model containers, the baseline and the
optimized model, to classify it, so their answers can be compared side by side.

The same website serves the garment (FashionMNIST) and the CIFAR10 classifier; CLASSIFIER picks one.
"""
import base64
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import requests
from flask import Flask, jsonify, render_template, request

SITES = {
    "fashion": {
        "title": "Garment Classifier",
        "icon": "👕",
        "subtitle": "Upload a photo of a garment. Two networks trained on FashionMNIST classify it: "
                    "the PyTorch Quickstart baseline and the hyperparameter-optimized model.",
        "subject": "garment",
        "examples": "T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot",
    },
    "cifar10": {
        "title": "CIFAR10 Classifier",
        "icon": "🚗",
        "subtitle": "Upload a photo. Two networks trained on CIFAR10 classify it: "
                    "the PyTorch tutorial baseline and the hyperparameter-optimized model.",
        "subject": "photo",
        "examples": "plane, car, bird, cat, deer, dog, frog, horse, ship, truck",
    },
}
SITE = SITES[os.getenv("CLASSIFIER", "fashion")]

# Addresses of the two Cog containers. In Docker Compose these are the service names,
# when the website runs directly on the computer they are the published ports.
MODELS = {
    "baseline": os.getenv("BASELINE_URL", "http://localhost:5002").rstrip("/"),
    "optimized": os.getenv("OPTIMIZED_URL", "http://localhost:5003").rstrip("/"),
}
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp"}
PREDICTION_TIMEOUT = 60  # seconds; a prediction takes a few milliseconds once the model is loaded

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_BYTES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("classifier-web")


@app.get("/")
def index():
    return render_template("index.html", site=SITE)


def predict(name, data_uri):
    """Sends the image to one Cog container. Returns the model output, or {"error": ...}."""
    try:
        response = requests.post(f"{MODELS[name]}/predictions", json={"input": {"image": data_uri}},
                                 timeout=PREDICTION_TIMEOUT)
        prediction = response.json()
    except (requests.RequestException, ValueError):
        log.exception("%s model at %s is unreachable", name, MODELS[name])
        return {"error": f"The {name} model is not reachable. Is its Cog container running?"}

    if response.status_code != 200 or prediction.get("status") != "succeeded":
        log.warning("%s prediction failed: %s", name, prediction.get("error") or prediction)
        return {"error": "The model could not read this image. Is it a valid image file?"}

    output = prediction["output"]
    output["predict_ms"] = round(prediction.get("metrics", {}).get("predict_time", 0) * 1000, 1)
    return output


@app.post("/api/classify")
def classify():
    image = request.files.get("image")

    if image is None or not image.filename:
        return jsonify(error=f"Please choose a {SITE['subject']} image."), 400
    if image.mimetype not in ALLOWED_TYPES:
        return jsonify(error="Please use a JPEG, PNG, WebP, GIF or BMP image."), 400

    data = image.read()
    if not data:
        return jsonify(error="The file is empty."), 400

    # Cog accepts files inside JSON as data URIs: data:<type>;base64,<content>
    data_uri = f"data:{image.mimetype};base64,{base64.b64encode(data).decode()}"

    # Both models at the same time, so the page waits for the slower one only
    with ThreadPoolExecutor(max_workers=len(MODELS)) as pool:
        futures = {name: pool.submit(predict, name, data_uri) for name in MODELS}
        results = {name: future.result() for name, future in futures.items()}

    if all("error" in result for result in results.values()):
        return jsonify(error=results["baseline"]["error"]), 502

    log.info("%s (%d KB) -> baseline: %s, optimized: %s", image.filename, len(data) // 1024,
             *(r.get("label", "error") for r in results.values()))
    return jsonify(results)


@app.get("/api/health")
def health():
    """Tells the page whether each model container is ready (Cog reports STARTING while setup() runs)."""
    statuses = {}
    for name, url in MODELS.items():
        try:
            statuses[name] = requests.get(f"{url}/health-check", timeout=3).json().get("status", "UNKNOWN")
        except (requests.RequestException, ValueError):
            statuses[name] = "UNREACHABLE"
    return jsonify(statuses)


@app.errorhandler(413)
def file_too_large(_):
    return jsonify(error=f"The image is too large (max {MAX_IMAGE_BYTES // (1024 * 1024)} MB)."), 413


if __name__ == "__main__":
    app.run(debug=True)
