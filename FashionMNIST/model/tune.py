"""Task 4: hyperparameter optimization of the garment classifier with Optuna.

1. Split the 60,000 training images into 50,000 for training and 10,000 for validation.
2. Optuna proposes hyperparameters (TPE sampler), trains a model and scores it on the validation set.
   Trials that are clearly worse than the others after a few epochs are pruned (stopped early).
3. The best hyperparameters are used to train the final model on all 60,000 training images,
   which is evaluated once on the 10,000 test images, like the baseline.

Usage:  python tune.py [number of trials, default 40]   -> weights/optimized.pth + weights/optimized.json
"""
import sys

import optuna

from training import Timer, device, fit, load_datasets, save_model, split_train_validation

TUNING_EPOCHS = 15  # every trial trains this long at most; the best epoch count is kept per trial


def suggest_config(trial):
    """The search space: which hyperparameters Optuna may change, and in which range."""
    n_layers = trial.suggest_int("n_layers", 1, 3)
    optimizer = trial.suggest_categorical("optimizer", ["SGD", "Adam"])
    config = {
        "hidden_sizes": [trial.suggest_categorical(f"hidden_size_{i + 1}", [128, 256, 512, 1024]) for i in range(n_layers)],
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "optimizer": optimizer,
        # SGD and Adam work best with very different learning rates, so they get their own range
        "learning_rate": (trial.suggest_float("sgd_learning_rate", 1e-3, 3e-1, log=True) if optimizer == "SGD"
                          else trial.suggest_float("adam_learning_rate", 1e-4, 1e-2, log=True)),
        "momentum": trial.suggest_float("momentum", 0.0, 0.95) if optimizer == "SGD" else 0.0,
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "epochs": TUNING_EPOCHS,
    }
    return config


def make_objective(train_data, validation_data):
    def objective(trial):
        config = suggest_config(trial)
        best = {"accuracy": 0.0, "epoch": 0}

        def on_epoch_end(epoch, loss, accuracy):
            if accuracy > best["accuracy"]:
                best.update(accuracy=accuracy, epoch=epoch)
            trial.report(accuracy, epoch)  # lets the pruner compare this trial with earlier ones
            if trial.should_prune():
                raise optuna.TrialPruned()

        fit(config, train_data, validation_data, on_epoch_end=on_epoch_end)
        # Early stopping: the final model trains for the epoch count that gave the best validation accuracy
        trial.set_user_attr("config", {**config, "epochs": best["epoch"]})
        return best["accuracy"]

    return objective


def main():
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    print(f"Tuning on {device}, {n_trials} trials")
    train_data, test_data = load_datasets()
    tuning_train, validation = split_train_validation(train_data)

    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = optuna.create_study(
        study_name="fashion-mnist",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=471),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )
    with Timer() as tuning_timer:
        study.optimize(make_objective(tuning_train, validation), n_trials=n_trials)

    best_config = study.best_trial.user_attrs["config"]
    print(f"Best validation accuracy {study.best_value:.2%} with {best_config}")

    # Final model: best hyperparameters, all 60,000 training images
    with Timer() as training_timer:
        model, history = fit(best_config, train_data, test_data)

    states = [t.state.name for t in study.trials]
    save_model("optimized", model, best_config, test_data, history, extra={
        "training_seconds": training_timer.seconds,
        "tuning": {
            "trials": len(study.trials),
            "completed": states.count("COMPLETE"),
            "pruned": states.count("PRUNED"),
            "seconds": tuning_timer.seconds,
            "best_validation_accuracy": round(study.best_value, 4),
            "trial_results": [
                {"number": t.number, "state": t.state.name, "validation_accuracy": round(t.value, 4) if t.value else None,
                 "params": t.params}
                for t in study.trials
            ],
        },
    })


if __name__ == "__main__":
    main()
