"""The garment classifier network, shared by training (train.py, tune.py) and serving (run.py)."""
from torch import nn

CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
IMAGE_SIZE = 28  # FashionMNIST images are 28x28 grayscale


class NeuralNetwork(nn.Module):
    """Fully connected network from the PyTorch Quickstart, with a configurable shape.

    The defaults (two hidden layers of 512, no dropout) are exactly the Quickstart model.
    Hyperparameter optimization changes the number and size of the hidden layers and the dropout.
    """

    def __init__(self, hidden_sizes=(512, 512), dropout=0.0):
        super().__init__()
        self.flatten = nn.Flatten()  # 1x28x28 image -> 784 values

        layers = []
        in_features = IMAGE_SIZE * IMAGE_SIZE
        for size in hidden_sizes:
            layers += [nn.Linear(in_features, size), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))  # randomly zeroes activations while training
            in_features = size
        layers.append(nn.Linear(in_features, len(CLASSES)))  # one logit per class
        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        return self.linear_relu_stack(self.flatten(x))
