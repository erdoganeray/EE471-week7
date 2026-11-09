"""The CIFAR10 classifier network, shared by training (train.py, tune.py) and serving (run.py)."""
import torch
import torch.nn.functional as F
from torch import nn

CLASSES = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
IMAGE_SIZE = 32  # CIFAR10 images are 32x32 RGB

# The tutorial's normalization: every channel from 0..1 to -1..1
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


class Net(nn.Module):
    """Convolutional network from the PyTorch "Training a classifier" tutorial, with a configurable shape.

    The defaults (6 and 16 filters, fully connected 120 and 84, no dropout) are exactly the tutorial model.
    Hyperparameter optimization changes the widths and the dropout, like the tutorial's Ray Tune
    version changes l1 and l2.
    """

    def __init__(self, conv1=6, conv2=16, fc1=120, fc2=84, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, conv1, 5)    # 3x32x32 -> conv1 x 28x28, pooled to 14x14
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(conv1, conv2, 5)  # -> conv2 x 10x10, pooled to 5x5
        self.fc1 = nn.Linear(conv2 * 5 * 5, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, len(CLASSES))  # one logit per class
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.fc3(x)


def build_network(config):
    return Net(**config["architecture"], dropout=config["dropout"])
