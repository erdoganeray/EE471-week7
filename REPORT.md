# Classifier Performance Report: FashionMNIST and CIFAR10

**Setup.** Each dataset has a *baseline*, trained with the network and hyperparameters of the official PyTorch tutorial
(Quickstart for FashionMNIST, "Training a classifier" for CIFAR10), and an *optimized* model whose hyperparameters were
chosen by Optuna (TPE sampler, 40 trials, median pruner). Hyperparameters were scored on a 10,000-image validation split
of the training set; the test set (10,000 images) was used once per model, for the numbers below. Hardware: RTX 3050 Ti Laptop GPU.

| | FashionMNIST baseline | FashionMNIST optimized | CIFAR10 baseline | CIFAR10 optimized |
|---|---|---|---|---|
| Network | MLP 512 → 512 | MLP 512 → 1024 | CNN conv 6/16, fc 120/84 | CNN conv 32/128, fc 256/256 |
| Optimizer, learning rate | SGD, 0.001 | Adam, 0.00038 | SGD + momentum 0.9, 0.001 | SGD + momentum 0.55, 0.029 |
| Batch size, epochs | 64, 5 | 64, 15 | 4, 2 | 32, 24 |
| Regularization | none | dropout 0.07 | none | dropout 0.07, flip + crop augmentation |
| Parameters | 669,706 | 937,482 | 62,006 | 992,778 |
| **Test accuracy** | **65.04%** | **89.55%** (+24.5 pts) | **54.06%** | **80.89%** (+26.8 pts) |
| Test loss | 1.057 | 0.326 | 1.283 | 0.575 |
| Weakest class | Sandal 3.1%, Shirt 6.6% | Shirt 72.5% | Cat 17.8% | Cat 54.7% |
| Training time | 9 s | 30 s (+10 min search) | 56 s | 93 s (+35 min search) |

**FashionMNIST.** The baseline is not limited by its network but by its training: plain SGD with a learning rate of 0.001
for 5 epochs has barely started to converge (test loss still 1.06, falling every epoch). It almost never predicts Sandal
or Shirt, so their accuracy is near zero. The search shows that the architecture hardly matters: the three best trials use
512→1024, 512→512 (the baseline's own shape) and a single 512 layer, all within 0.2 points (89.65–89.88% validation).
What they share is Adam with a learning rate around 0.0004–0.0005, batch size 64 and light dropout. Trials with a too high
learning rate (0.0047) or heavy dropout (0.39–0.48) stayed at 82–88%. The improvement is therefore almost entirely due to
the training hyperparameters (optimizer, learning rate, number of epochs).

**CIFAR10.** Natural images are much harder, and here capacity and regularization matter too. The tutorial network
(62k parameters, 2 epochs) reaches 54% and confuses animals (cat 18%, bird 33%). The best trial is 16× larger but would
overfit without regularization. Data augmentation was the single most important choice: the six best trials all used it,
and the best trial without augmentation reached only 70.8% validation, against 80.95% with it. A higher learning rate (0.029)
with a larger batch (32) also trains far more efficiently than the tutorial's 0.001 with batch size 4.
Cats remain the weakest class (55%); 183 of the 453 misclassified cats were predicted as dogs.

**Why hyperparameter optimization matters.** With the *same model families*, choosing the hyperparameters systematically
added about 25 accuracy points on both datasets, without any change to the code of the networks.
Hyperparameters interact (a good learning rate depends on the optimizer and the batch size), so changing them one at a time
does not find good combinations; a sampler like TPE learns from earlier trials which regions are promising, and pruning
stopped 22 of 40 trials early on both datasets, so the search cost 10 and 35 minutes instead of hours.
Using a separate validation split keeps the test accuracy an honest estimate: the hyperparameters were never chosen by
looking at the test set. The cost of the optimized models is small at inference time: in the Cog containers every model,
baseline or optimized, answers in about 8 ms on a CPU.
