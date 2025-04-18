# 🧠 MNIST Digit Classifier from Scratch (NumPy Only)

This repository contains a complete **from-scratch implementation** of a digit classifier using the **MNIST dataset** from [Kaggle](https://www.kaggle.com/competitions/digit-recognizer). It achieves a validation accuracy of **83%** using a basic feedforward neural network implemented only with **NumPy**, without using high-level ML libraries.

---

## 📁 Project Overview

- **Framework:** Pure Python + NumPy
- **Dataset:** MNIST (digit recognizer from Kaggle)
- **Accuracy Achieved:** ~83%
- **File:** [`mnist-from-scratch.ipynb`](https://github.com/vaibhavsss/MNIST-from-scratch/blob/main/mnist-from-scratch.ipynb)

---

## 📊 Dataset

- **Input Features:** 28x28 grayscale images → flattened to 784-dimensional vectors
- **Target Labels:** 0 through 9 (10 classes)
- **File Used:** `train.csv` from the Kaggle competition

---

## 🧠 Model Architecture

This project implements a shallow neural network with:

| Layer         | Details                          |
|---------------|----------------------------------|
| Input Layer   | 784 neurons (one for each pixel) |
| Hidden Layer  | 64 neurons                       |
| Output Layer  | 10 neurons (Softmax activation)  |

### ⚙️ Activation Functions
- **Hidden Layer:** Sigmoid
- **Output Layer:** Softmax (for multi-class classification)

---

## 🔢 Mathematical Formulations

### 1. Forward Propagation

For layer \( l \):

\[
Z^{[l]} = W^{[l]} \cdot A^{[l-1]} + b^{[l]}
\]
\[
A^{[l]} = \text{activation}(Z^{[l]})
\]

- **Sigmoid:**  
  \[
  \sigma(z) = \frac{1}{1 + e^{-z}}
  \]

- **Softmax:**  
  \[
  \text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
  \]

---

### 2. Loss Function (Cross Entropy)

\[
\mathcal{L} = - \sum y_i \log(\hat{y}_i)
\]

Where:
- \( y_i \) is the true label (one-hot encoded)
- \( \hat{y}_i \) is the predicted probability

---

### 3. Backpropagation

Gradients computed using the chain rule:
- Compute \( dZ \), \( dW \), \( db \) for each layer
- Update weights:
  \[
  W = W - \alpha \cdot dW
  \]
  \[
  b = b - \alpha \cdot db
  \]
  where \( \alpha \) is the learning rate

---

## 🧪 Training Details

- **Learning Rate:** 0.01
- **Epochs:** 100
- **Loss:** Categorical Cross-Entropy
- **Optimizer:** Vanilla Gradient Descent
- **Initialization:** Random normal
- **Train/Validation Split:** 80/20

---

## 📈 Results

After training for 100 epochs:

- **Training Accuracy:** ~83%
- **Validation Accuracy:** ~82%
- **Loss Curve:** Shows stable decrease across epochs

---

## 📂 File Breakdown

| File                        | Description                               |
|-----------------------------|-------------------------------------------|
| `mnist-from-scratch.ipynb` | Complete notebook with code + results     |
| `train.csv`                | Dataset (Kaggle MNIST) - not uploaded     |
| `README.md`                | Project documentation                     |

---

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/vaibhavsss/MNIST-from-scratch.git
cd MNIST-from-scratch
