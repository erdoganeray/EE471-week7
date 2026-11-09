# EE471 – Week 7 In-Class Exercise

EE 471 Modern Software Development Practices and Technologies, Izmir Institute of Technology.
Training image classifiers with PyTorch, optimizing their hyperparameters, and serving them with Cog.

| Path | Content |
|---|---|
| `FashionMNIST/` | Garment classifier: training, hyperparameter optimization, two Cog model containers, Compose file, Postman collection |
| `CIFAR10/` | CIFAR10 classifier, same structure |
| `REPORT.md` | One-page report on the performance of both classifiers |
| `web/` | The website (Week 6's cat classifier, reworked): classifies an image with the baseline and the optimized model side by side. Shared by both classifiers |
| `postman/make_request.py` | Turns an image into the JSON body that Cog expects, for sending from Postman |

```
                                    ┌──▶ baseline container  (Cog, :5002)
Browser ──(image)──▶ website (:8000)┤
                                    └──▶ optimized container (Cog, :5003)
```

## Setup

1. Install [Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/) (on Windows with the WSL 2 backend)
   and [Cog](https://github.com/replicate/cog#install). Cog runs on Linux and macOS; on Windows install it inside WSL (Ubuntu)
   and turn on **Docker Desktop → Settings → Resources → WSL integration** for that distro:
   ```bash
   mkdir -p ~/.local/bin && curl -fsSL -o ~/.local/bin/cog https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m) && chmod +x ~/.local/bin/cog
   ```
2. Clone the repository:
   ```bash
   git clone https://github.com/erdoganeray/EE471-week7.git
   ```
3. Only needed to train the models again (the trained weights are in the repository).
   Python 3.12 or 3.13 and PyTorch with CUDA if there is an NVIDIA GPU
   ([other platforms](https://pytorch.org/get-started/locally/)):
   ```bash
   python -m venv .venv
   ```
   ```bash
   .venv\Scripts\activate
   ```
   ```bash
   pip install -r requirements-train.txt --extra-index-url https://download.pytorch.org/whl/cu124
   ```

## FashionMNIST: garment classifier

### 1. Train the baseline (Task 1)

The network and the hyperparameters of the [PyTorch Quickstart](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html):
a fully connected network (784 → 512 → 512 → 10), SGD with learning rate 0.001, batch size 64, 5 epochs.

```bash
cd FashionMNIST/model
python train.py
```

The first run downloads FashionMNIST into `FashionMNIST/model/data/`. The result is `weights/baseline.pth`
(hyperparameters + `state_dict`) and `weights/baseline.json` (test accuracy, per-class accuracy, accuracy after every epoch).

### 2. Optimize the hyperparameters (Task 4)

```bash
python tune.py 40
```

[Optuna](https://optuna.org/) runs 40 trials. Each trial trains a model with hyperparameters chosen by the TPE sampler
on 50,000 training images and scores it on the 10,000 held-out validation images; the test set is never used for choosing.
Trials that are worse than the median of the earlier ones after 3 epochs are stopped early (median pruner).
The best hyperparameters are then used to train the final model on all 60,000 training images → `weights/optimized.pth` / `.json`.

| Hyperparameter | Search range |
|---|---|
| Hidden layers | 1–3 layers of 128, 256, 512 or 1024 neurons |
| Dropout | 0–0.5 |
| Optimizer | SGD (learning rate 0.001–0.3, momentum 0–0.95) or Adam (learning rate 0.0001–0.01) |
| Weight decay | 1e-6–1e-3 |
| Batch size | 32, 64, 128, 256 |
| Epochs | up to 15; the epoch with the best validation accuracy is kept |

### 3. Build the model containers with Cog (Tasks 3 and 5)

One Cog project, two config files: they differ only in the class that runs predictions, which picks the weights.

```bash
cd FashionMNIST/model
cog build -f cog.baseline.yaml -t fashion-baseline
cog build -f cog.optimized.yaml -t fashion-optimized
```

### 4. Run the website (Tasks 2 and 6)

From the repository root (PowerShell or WSL):

```bash
docker compose -f FashionMNIST/compose.yaml up --build
```

Open http://localhost:8000, choose an image (`FashionMNIST/test-images/` has one test image per class) and press
**Classify with both models**. Each side shows the predicted class, the top 5 probabilities and the 28×28 image the model
actually received; a table below compares the hyperparameters of the two models.

The model expects a FashionMNIST-style image: a light garment on a black background, filling the frame.
`run.py` converts photos to that: grayscale, inverted if the background is light, cropped to the garment and scaled to 28×28.
Product photos on a plain background work best.

| File | Purpose |
|---|---|
| `model/networks.py` | The network (`NeuralNetwork`) and the class names, used by training and by Cog |
| `model/training.py` | Dataset loading, train loop, test loop, saving; shared by `train.py` and `tune.py` |
| `model/train.py` | Baseline training (Quickstart hyperparameters) |
| `model/tune.py` | Hyperparameter optimization with Optuna, then training of the optimized model |
| `model/export_samples.py` | Saves one test image per class to `test-images/` |
| `model/run.py` | Cog runner: `setup()` loads the weights once, `run()` preprocesses the image, classifies it and logs the prediction |
| `model/cog.baseline.yaml`, `model/cog.optimized.yaml` | Cog build recipes (Python 3.13, PyTorch 2.6) |
| `model/.dockerignore` | Keeps the training code, dataset and test images out of the images |
| `compose.yaml` | Runs both models and the website, starts the website once both models are READY |
| `postman/garment-classifier.postman_collection.json` | Postman requests for both models |

### Model API

| Model | Health check | Prediction |
|---|---|---|
| Baseline | `GET` http://localhost:5002/health-check | `POST` http://localhost:5002/predictions |
| Optimized | `GET` http://localhost:5003/health-check | `POST` http://localhost:5003/predictions |

Body: `{"input": {"image": "<URL or data URI>"}}`. Example `output`:

```json
{
  "label": "Sandal",
  "confidence": 0.9999,
  "probabilities": [{"label": "Sandal", "probability": 0.9999}, {"label": "Sneaker", "probability": 0.0001}, "..."],
  "model_input": "data:image/png;base64,...",
  "model": {"name": "optimized", "test_accuracy": 0.8955, "parameters": 937482, "config": {"...": "..."}}
}
```

### Testing with Postman

Cog only accepts JSON, with files inside as data URIs. `postman/make_request.py` turns an image into that JSON:

```bash
python postman/make_request.py FashionMNIST/test-images/sneaker.png
```

Import `FashionMNIST/postman/garment-classifier.postman_collection.json` into Postman, open
**optimized: classify (JSON body file)**, choose the generated `sneaker.json` under **Body → binary**, and press **Send**.
The *image URL* requests need the repository to be public on GitHub.

### Prediction logs inside the container

Every prediction is written to the container's output and to `/tmp/predictions.log` inside it:

```bash
docker exec -it fashion-optimized-1 tail -f /tmp/predictions.log   # from inside the container
docker compose -f FashionMNIST/compose.yaml logs -f                # all three containers
```

```
2026-11-09 11:16:43,647 INFO [optimized] size=224x224 prediction=Sandal confidence=1.000 second=Sneaker (0.000)
2026-11-09 11:16:43,647 INFO [baseline] size=224x224 prediction=Sneaker confidence=0.212 second=Sandal (0.164)
```

### Results

| | Baseline | Optimized |
|---|---|---|
| Hidden layers | 512 → 512 | 512 → 1024 |
| Dropout | 0 | 0.07 |
| Optimizer, learning rate | SGD, 0.001 | Adam, 0.000385 |
| Weight decay | 0 | 1.0e-6 |
| Batch size, epochs | 64, 5 | 64, 15 |
| **Test accuracy** | **65.04%** | **89.55%** |
| Weakest class | Sandal 3.1%, Shirt 6.6% | Shirt 72.5% |

40 Optuna trials (18 completed, 22 pruned) took about 10 minutes on an RTX 3050 Ti; the final model trains in 30 seconds.

## CIFAR10 classifier

Same steps and the same file layout as FashionMNIST, in `CIFAR10/`. The website is the same one, started with
`CLASSIFIER=cifar10`, on other ports so both classifiers can run at the same time.

### Train, optimize, build, run

```bash
cd CIFAR10/model
python train.py          # baseline: the tutorial network and hyperparameters
python tune.py 40        # Optuna, then the optimized model
python export_samples.py # one test image per class -> CIFAR10/test-images/
cog build -f cog.baseline.yaml -t cifar10-baseline
cog build -f cog.optimized.yaml -t cifar10-optimized
cd ../..
docker compose -f CIFAR10/compose.yaml up --build
```

Open http://localhost:8001.

The baseline is the network of the PyTorch [Training a classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
tutorial: two convolutional layers (6 and 16 filters, 5×5) with max pooling and three fully connected layers (120, 84, 10),
inputs normalized to −1..1, SGD with learning rate 0.001 and momentum 0.9, batch size 4, 2 epochs.

`tune.py` searches like the tutorial's [Ray Tune version](https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html)
(which tunes `l1`, `l2`, the learning rate and the batch size), with a few more hyperparameters:

| Hyperparameter | Search range |
|---|---|
| Convolution filters | conv1: 16, 32, 64 · conv2: 32, 64, 128 |
| Fully connected layers | fc1: 128, 256, 512 · fc2: 64, 128, 256 |
| Dropout | 0–0.5 |
| Optimizer | SGD (learning rate 0.001–0.1, momentum 0.5–0.95) or Adam (learning rate 0.0001–0.01) |
| Weight decay | 1e-6–1e-3 |
| Batch size | 32, 64, 128, 256 |
| Data augmentation | none, or random horizontal flip + random crop (4 px padding) |
| Epochs | up to 30; the epoch with the best validation accuracy is kept |

Photos are center-cropped to a square and resized to 32×32 before classification.

### Model API, Postman and logs

| Model | Health check | Prediction |
|---|---|---|
| Baseline | `GET` http://localhost:5004/health-check | `POST` http://localhost:5004/predictions |
| Optimized | `GET` http://localhost:5005/health-check | `POST` http://localhost:5005/predictions |

```bash
python postman/make_request.py CIFAR10/test-images/car.png
```

Import `CIFAR10/postman/cifar10-classifier.postman_collection.json` and send `car.json` as in the FashionMNIST section.

```bash
docker exec -it cifar10-optimized-1 tail -f /tmp/predictions.log
docker compose -f CIFAR10/compose.yaml logs -f
```

### Results

| | Baseline | Optimized |
|---|---|---|
| Network | conv 6/16, fc 120/84 | conv 32/128, fc 256/256 |
| Dropout, augmentation | 0, none | 0.07, flip + crop |
| Optimizer, learning rate | SGD (momentum 0.9), 0.001 | SGD (momentum 0.55), 0.029 |
| Batch size, epochs | 4, 2 | 32, 24 |
| **Test accuracy** | **54.06%** | **80.89%** |
| Weakest class | cat 17.8% | cat 54.7% |

40 Optuna trials (18 completed, 22 pruned) took about 35 minutes; the final model trains in 93 seconds.
See [`REPORT.md`](REPORT.md) for the comparison of both classifiers.
