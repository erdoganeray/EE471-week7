"""Task 1: the baseline garment classifier, trained exactly as in the PyTorch Quickstart.

Usage:  python train.py        -> weights/baseline.pth + weights/baseline.json
"""
from training import Timer, device, fit, load_datasets, save_model

# Hyperparameters of the Quickstart tutorial
BASELINE = {
    "hidden_sizes": [512, 512],
    "dropout": 0.0,
    "optimizer": "SGD",
    "learning_rate": 1e-3,
    "momentum": 0.0,
    "weight_decay": 0.0,
    "batch_size": 64,
    "epochs": 5,
}


def main():
    print(f"Training the baseline on {device}")
    train_data, test_data = load_datasets()

    def report(epoch, loss, accuracy):
        print(f"Epoch {epoch}: test loss {loss:.4f}, test accuracy {accuracy:.2%}")

    # Like the Quickstart, the baseline is evaluated on the test set after each epoch
    with Timer() as timer:
        model, history = fit(BASELINE, train_data, test_data, on_epoch_end=report)
    save_model("baseline", model, BASELINE, test_data, history, extra={"training_seconds": timer.seconds})


if __name__ == "__main__":
    main()
