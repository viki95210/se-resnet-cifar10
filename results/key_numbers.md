# Key numbers used in the report (24CS3062, SE-ResNet-18 vs ResNet-18, CIFAR-10)

| Model | Params | Best acc (%) | Epoch-100 acc (%) | Best ep. | Median s/epoch |
|---|---|---|---|---|---|
| ResNet-18 | 11,173,962 | 88.18 | 88.10 | 99 | 28.9 |
| SE-ResNet-18 (r=16) | 11,263,042 | 88.53 | 88.41 | 66 | 30.6 |
| Difference | +89,080 (+0.80%) | +0.35 | +0.31 | - | +1.7 (+5.9%) |

Setup: no augmentation, no normalisation, SGD(0.9, wd 5e-4), bs 128, LR 0.1 -> x0.1 at epochs 50 and 75, 100 epochs, seed 42.

## Ablation (reduction ratio) - AS LOGGED, see caveats
| r | Params | Logged best (%) | Epochs run | Best in first 19 ep (%) |
|---|---|---|---|---|
| Base | 11,173,962 | 88.18 | 100 | 79.39 |
| 4 | 11,524,522 | 82.44 | 19 | 82.44 |
| 8 (INVALID, contaminated) | 11,350,202 | 88.53 | 44 | 88.06 |
| 16 | 11,263,042 | 80.63 | 30 | 80.46 |
| 32 | 11,219,462 | 81.02 | 31 | 80.51 |
| 16 (main) | 11,263,042 | 88.53 | 100 | 81.48 |

Caveats: ablation runs used early stopping (patience 10) at LR 0.1 (stopped before the LR decay);
the r=8 run was not freshly initialised (94.4% train acc in epoch 1).
Confusion matrix / ROC / classification report come from the r=16 ABLATION checkpoint (80.63%), not the main model.
