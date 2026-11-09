"""Training helpers shared by train.py (baseline) and tune.py (hyperparameter optimization).

Follows the PyTorch "Training a classifier" tutorial: CIFAR10 with ToTensor + Normalize, a train loop
(forward, loss, backward, optimizer step) and a test loop (no_grad, accuracy).
"""
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Subset, TensorDataset, random_split
from torchvision import datasets, transforms

from networks import CLASSES, MEAN, STD, build_network

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
WEIGHTS_DIR = HERE / "weights"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The tutorial's transform: PIL image -> 3x32x32 tensor (0..1) -> normalized to -1..1
TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def seed_everything(seed=471):
    """Same seed -> same weight initialization and data order, so runs are reproducible."""
    random.seed(seed)
    torch.manual_seed(seed)  # also seeds the GPU


def load_datasets():
    """CIFAR10 as tensors: 50,000 training and 10,000 test images (3x32x32, normalized).

    The torchvision Dataset applies the transform to every image each time it is read.
    Doing that once and keeping the tensors (TensorDataset) makes every epoch several times faster,
    which matters when hyperparameter optimization trains dozens of models.
    """
    def as_tensors(train):
        dataset = datasets.CIFAR10(root=DATA_DIR, train=train, download=True, transform=TRANSFORM)
        images = torch.stack([image for image, _ in dataset])
        return TensorDataset(images, torch.tensor(dataset.targets))

    return as_tensors(train=True), as_tensors(train=False)


def split_train_validation(train_data, validation_size=10_000, seed=471):
    """Hold out part of the training set for choosing hyperparameters, so the test set stays unseen."""
    generator = torch.Generator().manual_seed(seed)
    return random_split(train_data, [len(train_data) - validation_size, validation_size], generator=generator)


def augment(images):
    """Data augmentation on a GPU batch: random horizontal flip and random crop (4 px padding).
    Every epoch the network sees slightly different versions of the images, so it memorizes less."""
    flip = torch.rand(len(images), device=images.device) < 0.5
    images = torch.where(flip[:, None, None, None], images.flip(3), images)
    padded = F.pad(images, (4, 4, 4, 4), mode="reflect")
    dx, dy = random.randint(0, 8), random.randint(0, 8)  # one shift per batch keeps it fast
    return padded[:, :, dy:dy + images.shape[2], dx:dx + images.shape[3]]


class GPUDataLoader:
    """Does what DataLoader(dataset, batch_size, shuffle) does, for a dataset that fits in GPU memory.

    DataLoader collects every batch sample by sample on the CPU and copies it to the GPU. CIFAR10 is only
    ~600 MB as float tensors, so here it is copied to the GPU once and every batch is a slice of it.
    """

    def __init__(self, dataset, batch_size, shuffle=False):
        if isinstance(dataset, Subset):  # e.g. the training part of random_split
            images, labels = dataset.dataset.tensors
            images, labels = images[dataset.indices], labels[dataset.indices]
        else:
            images, labels = dataset.tensors
        self.images, self.labels = images.to(device), labels.to(device)
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        n = len(self.labels)
        order = torch.randperm(n, device=device) if self.shuffle else torch.arange(n, device=device)
        for start in range(0, n, self.batch_size):
            batch = order[start:start + self.batch_size]
            yield self.images[batch], self.labels[batch]

    def __len__(self):
        return (len(self.labels) + self.batch_size - 1) // self.batch_size


def build_optimizer(config, model):
    if config["optimizer"] == "SGD":
        return torch.optim.SGD(model.parameters(), lr=config["learning_rate"],
                               momentum=config.get("momentum", 0.0), weight_decay=config["weight_decay"])
    if config["optimizer"] == "Adam":
        return torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    raise ValueError(f"Unknown optimizer {config['optimizer']}")


def train_loop(dataloader, model, loss_fn, optimizer, use_augmentation=False):
    model.train()  # enables dropout
    for X, y in dataloader:
        if use_augmentation:
            X = augment(X)
        loss = loss_fn(model(X), y)  # forward pass + loss
        loss.backward()              # backpropagation: gradient of the loss for every parameter
        optimizer.step()             # update the parameters using the gradients
        optimizer.zero_grad()        # gradients add up by default, reset them for the next batch


def test_loop(dataloader, model, loss_fn):
    """Returns (average loss, accuracy, number of correct predictions per class)."""
    model.eval()  # disables dropout
    total_loss, correct = 0.0, 0
    correct_per_class = torch.zeros(len(CLASSES), dtype=torch.long)
    with torch.no_grad():  # no gradients needed for evaluation
        for X, y in dataloader:
            pred = model(X)
            total_loss += loss_fn(pred, y).item() * len(y)
            hits = pred.argmax(1) == y
            correct += hits.sum().item()
            correct_per_class += torch.bincount(y[hits], minlength=len(CLASSES)).cpu()
    size = len(dataloader.labels)
    return total_loss / size, correct / size, correct_per_class


def fit(config, train_data, eval_data, on_epoch_end=None):
    """Trains a model with the given hyperparameters and evaluates it after every epoch.

    on_epoch_end(epoch, loss, accuracy) is called after each epoch; tune.py uses it to stop bad trials early.
    Returns the trained model and the per-epoch history.
    """
    seed_everything(config.get("seed", 471))
    train_loader = GPUDataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    eval_loader = GPUDataLoader(eval_data, batch_size=1000)

    model = build_network(config).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = build_optimizer(config, model)

    history = []
    for epoch in range(1, config["epochs"] + 1):
        train_loop(train_loader, model, loss_fn, optimizer, use_augmentation=config["augment"])
        loss, accuracy, _ = test_loop(eval_loader, model, loss_fn)
        history.append({"epoch": epoch, "loss": round(loss, 4), "accuracy": round(accuracy, 4)})
        if on_epoch_end:
            on_epoch_end(epoch, loss, accuracy)
    return model, history


def save_model(name, model, config, test_data, history, extra=None):
    """Saves the weights + hyperparameters (weights/<name>.pth) and the test metrics (weights/<name>.json)."""
    WEIGHTS_DIR.mkdir(exist_ok=True)
    loss, accuracy, correct_per_class = test_loop(GPUDataLoader(test_data, batch_size=1000), model, nn.CrossEntropyLoss())
    samples_per_class = torch.bincount(test_data.tensors[1], minlength=len(CLASSES))

    # state_dict = the learned parameters only; the config is stored next to it so run.py can rebuild the network
    torch.save({"config": config, "state_dict": model.state_dict()}, WEIGHTS_DIR / f"{name}.pth")

    metrics = {
        "name": name,
        "test_accuracy": round(accuracy, 4),
        "test_loss": round(loss, 4),
        "parameters": sum(p.numel() for p in model.parameters()),
        "config": config,
        "per_class_accuracy": {
            label: round(c / n, 4) for label, c, n in zip(CLASSES, correct_per_class.tolist(), samples_per_class.tolist())
        },
        "history": history,
        **(extra or {}),
    }
    (WEIGHTS_DIR / f"{name}.json").write_text(json.dumps(metrics, indent=2))
    print(f"[{name}] test accuracy {accuracy:.2%}, test loss {loss:.4f} -> {WEIGHTS_DIR / (name + '.pth')}")
    return metrics


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.seconds = round(time.perf_counter() - self.start, 1)
