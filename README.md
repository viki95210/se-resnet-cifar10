
# Squeeze-and-Excitation Networks
## SE-ResNet-18 Implementation and Ablation on CIFAR-10

**Vikram Kumar Meena (24CS3062)**  
Deep Learning 2026 – Paper Implementation Project  
Computer Science and Engineering (CSE)  
Rajiv Gandhi Institute of Petroleum Technology (RGIPT)

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-green.svg)](https://www.cs.toronto.edu/~kriz/cifar.html)
[![Paper](https://img.shields.io/badge/Paper-Squeeze--and--Excitation-red.svg)](https://arxiv.org/abs/1709.01507)

---

## 📌 Project Overview

This project presents a from-scratch PyTorch implementation of
**Squeeze-and-Excitation (SE) Networks** and an SE-ResNet-18 model
for image classification on the CIFAR-10 dataset.

The primary objective is to investigate whether channel-wise attention
improves the performance of a residual neural network and to study the
effect of the SE reduction ratio on model parameters and classification
accuracy.

The implementation compares a standard ResNet-18 baseline against
an SE-ResNet-18 model under the same main experimental training setup.

## 🎯 Objectives

- Implement the SE block from scratch using PyTorch.
- Integrate channel attention into a ResNet-18 architecture.
- Compare the baseline ResNet-18 with SE-ResNet-18.
- Evaluate the parameter overhead and classification performance.
- Investigate reduction ratios `r ∈ {4, 8, 16, 32}`.
- Document experimental limitations and reproducibility issues.

## 🧠 Squeeze-and-Excitation Block

The SE block recalibrates feature channels by learning their
relative importance using global contextual information.

It consists of three main operations:

1. **Squeeze:** Global average pooling converts each feature map
   into a single channel descriptor.
2. **Excitation:** A two-layer bottleneck network learns channel-wise
   weights using ReLU and sigmoid activations.
3. **Scale:** The learned weights rescale the original feature maps.

### Mathematical Formulation

For a feature map \(U\), the squeeze operation computes:

\[
z_c = \frac{1}{HW}\sum_{i=1}^{H}\sum_{j=1}^{W}u_c(i,j)
\]

The excitation operation generates channel-wise gates:

\[
s = \sigma(W_2\delta(W_1z))
\]

The recalibrated feature map is:

\[
\widetilde{u}_c = s_c \cdot u_c
\]

Here, \(H\) and \(W\) represent spatial dimensions, \(C\) is the
number of channels, and \(r\) is the reduction ratio.

## 🏗️ Model Architecture

### Baseline ResNet-18

- CIFAR-style 3 × 3 convolutional stem
- Four residual stages
- Two basic residual blocks per stage
- Channel dimensions: 64, 128, 256 and 512
- Global average pooling
- Fully connected classifier for 10 classes

### SE-ResNet-18

The SE block is inserted after the second batch-normalization layer
of the residual branch and before residual addition.

The baseline and SE-enhanced models use the same backbone and main
training configuration, allowing their results to be compared.

## 📂 Dataset

**CIFAR-10** contains 60,000 colour images of size 32 × 32 pixels,
divided into:

- Training set: 50,000 images
- Test set: 10,000 images
- Number of classes: 10

The main experiment scales pixel values to `[0, 1]` without
data augmentation or mean/std normalization.

## ⚙️ Experimental Setup

| Parameter | Configuration |
|---|---|
| Framework | PyTorch |
| Dataset | CIFAR-10 |
| Baseline model | ResNet-18 |
| Proposed model | SE-ResNet-18 |
| Optimizer | SGD |
| Momentum | 0.9 |
| Weight decay | 5 × 10⁻⁴ |
| Batch size | 128 |
| Initial learning rate | 0.1 |
| Learning-rate schedule | ×0.1 at epochs 50 and 75 |
| Training epochs | 100 |
| Random seed | 42 |
| Loss function | Cross-entropy |
| Reduction ratio | 16 for the main comparison |

## 📊 Main Experimental Results

The following results are taken from the project's IEEE report.

| Metric | ResNet-18 | SE-ResNet-18 |
|---|---:|---:|
| Parameters | 11,173,962 | 11,263,042 |
| Best test accuracy | 88.18% | 88.53% |
| Epoch-100 accuracy | 88.10% | 88.41% |
| Best epoch | 99 | 66 |
| Median epoch time | 28.9 s | 30.6 s |

### Key Observations

- SE-ResNet-18 achieved a best test accuracy of **88.53%**,
  compared with **88.18%** for the baseline.
- The best-accuracy difference was **+0.35 percentage points**.
- The SE module added 89,080 parameters, approximately **0.80%**
  over the baseline.
- Median epoch time increased from 28.9 seconds to 30.6 seconds,
  approximately 5.9%.
- The observed accuracy improvement is small and cannot be
  distinguished confidently from run-to-run variation using
  the available single-seed comparison.

## 🔬 Reduction-Ratio Ablation

The reduction ratio controls the bottleneck size in the SE block.
The report investigates:

`r ∈ {4, 8, 16, 32}`

As the reduction ratio increases, the number of parameters generally
decreases.

However, the ablation results are **inconclusive** because some runs
stopped before the first learning-rate decay, and the `r = 8` run
was contaminated by weights carried over from an earlier execution.

Therefore, the experiments do not establish which reduction ratio
provides the best accuracy or parameter-efficiency trade-off.

| Reduction ratio | Parameters | Logged best accuracy | Status |
|---|---:|---:|---|
| 4 | 11,524,522 | 82.44% | Early-stopped run |
| 8 | 11,350,202 | 88.53% | Invalid; contaminated run |
| 16 | 11,263,042 | 80.63% | Early-stopped ablation run |
| 32 | 11,219,462 | 81.02% | Early-stopped run |

**Note:** The ablation accuracies above are not directly comparable
with the 100-epoch main experiment because the training protocols
differ.

## ⚠️ Limitations

1. No data augmentation was used in the main experiment, resulting
   in evidence of overfitting.
2. The best checkpoint was selected using test accuracy rather than
   a separate validation set.
3. Only one main training seed was used per model.
4. The reduction-ratio ablation requires clean reruns with fresh
   initialization and a matched learning-rate schedule.
5. The reported accuracy improvement is too small to establish a
   reliable performance advantage without further experiments.

## 🚀 Future Work

- Add standard random crop and horizontal-flip augmentation.
- Introduce a separate validation set for model selection.
- Repeat experiments using multiple random seeds.
- Retrain every reduction-ratio configuration from fresh
  initialization for the complete learning-rate schedule.
- Compare accuracy, parameter count and training time under
  a consistent experimental protocol.
- Correct the SE visualization hook to capture the sigmoid gate
  values rather than the recalibrated feature maps.

## 📁 Repository Contents

The repository contains the implementation, experiment notebooks,
saved model checkpoints and CSV result files used in the project.

Key artifacts reported in the project include:

- `baseline_resnet_best.pth`
- `se_resnet_best.pth`
- CSV experiment results

Refer to the repository files for the exact source-code and notebook
organization.

## 📚 References

1. J. Hu, L. Shen, S. Albanie, G. Sun and E. Wu,
   "Squeeze-and-Excitation Networks," IEEE Transactions on Pattern
   Analysis and Machine Intelligence.
   [Paper](https://arxiv.org/abs/1709.01507)

2. K. He, X. Zhang, S. Ren and J. Sun,
   "Deep Residual Learning for Image Recognition," CVPR, 2016.

3. K. He, X. Zhang, S. Ren and J. Sun,
   "Identity Mappings in Deep Residual Networks," ECCV, 2016.

4. A. Krizhevsky and G. Hinton,
   "Learning Multiple Layers of Features from Tiny Images," 2009.

## 👨‍💻 Author

**Vikram Kumar Meena**  
Roll Number: 24CS3062  
Department: Computer Science and Engineering  
Institution: Rajiv Gandhi Institute of Petroleum Technology (RGIPT)  
Email: 24cs3062@rgipt.ac.in

**GitHub Repository:**  
https://github.com/viki95210/se-resnet-cifar10.git

---

*Deep Learning 2026 – Paper Implementation Project*
