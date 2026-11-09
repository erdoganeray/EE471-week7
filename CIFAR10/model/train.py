"""The baseline CIFAR10 classifier, trained exactly as in the PyTorch "Training a classifier" tutorial.

Usage:  python train.py        -> weights/baseline.pth + weights/baseline.json
"""
from training import Timer, device, fit, load_datasets, save_model

# Network and hyperparameters of the tutorial
BASELINE = {
    "architecture": {"conv1": 6, "conv2": 16, "fc1": 120, "fc2": 84},
    "dropout": 0.0,
    "optimizer": "SGD",
    "learning_rate": 1e-3,
    "momentum": 0.9,
    "weight_decay": 0.0,
    "batch_size": 4,
    "epochs": 2,
    "augment": False,
}


def main():
    print(f"Training the baseline on {device}")
    train_data, test_data = load_datasets()

    def report(epoch, loss, accuracy):
        print(f"Epoch {epoch}: test loss {loss:.4f}, test accuracy {accuracy:.2%}")

    with Timer() as timer:
        model, history = fit(BASELINE, train_data, test_data, on_epoch_end=report)
    save_model("baseline", model, BASELINE, test_data, history, extra={"training_seconds": timer.seconds})


if __name__ == "__main__":
    main()
