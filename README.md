# 🧠 MNIST Digit Classifier from Scratch (NumPy Only)

This project is a from-scratch implementation of a **handwritten digit classifier** using only **NumPy**, trained on the **MNIST dataset** from [Kaggle's Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer/data). It achieves an accuracy of **83%** on the validation set using basic feedforward neural network principles.

---

## 📊 Dataset Overview

- **Images:** 28x28 grayscale pixels
- **Classes:** Digits 0–9
- **Format:** Flattened into 784-dimensional feature vectors (CSV format)

---

## 🧠 Model Architecture

A simple feedforward neural network with:
- **Input layer:** 784 neurons
- **Hidden layer:** 128 neurons (with ReLU or Sigmoid)
- **Output layer:** 10 neurons (with Softmax)

---

## 🧮 Math Behind the Model

**1. Forward Propagation**
- \( Z = W \cdot X + b \)
- \( A = \text{activation}(Z) \)

**Activations:**
- Sigmoid: \( \sigma(z) = \frac{1}{1 + e^{-z}} \)
- ReLU: \( \max(0, z) \)
- Softmax (output layer):  
  \[
  \text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
  \]

---

**2. Loss Function (Cross Entropy):**  
\[
L = -\sum y_i \log(\hat{y}_i)
\]

---

**3. Backpropagation & Gradient Descent:**
- Use chain rule to compute gradients
- Update weights:  
  \[
  W = W - \alpha \cdot \frac{\partial L}{\partial W}
  \]
  where \( \alpha \) is the learning rate

---

**4. Accuracy:**
\[
\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Samples}} \times 100\%
\]

---

## ✅ Results

- **Validation Accuracy:** 83%
- **Training Method:** Full-batch gradient descent
- **Activation:** ReLU or Sigmoid
- **Output:** Softmax with cross-entropy loss

---

## 🚀 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/mnist-from-scratch.git
   cd mnist-from-scratch
# 🧠 MNIST Digit Classifier from Scratch (NumPy Only)

This project is a from-scratch implementation of a **handwritten digit classifier** using only **NumPy**, trained on the **MNIST dataset** from [Kaggle's Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer/data). It achieves an accuracy of **83%** on the validation set using basic feedforward neural network principles.

---

## 📊 Dataset Overview

- **Images:** 28x28 grayscale pixels
- **Classes:** Digits 0–9
- **Format:** Flattened into 784-dimensional feature vectors (CSV format)

---

## 🧠 Model Architecture

A simple feedforward neural network with:
- **Input layer:** 784 neurons
- **Hidden layer:** 128 neurons (with ReLU or Sigmoid)
- **Output layer:** 10 neurons (with Softmax)

---

## 🧮 Math Behind the Model

**1. Forward Propagation**
- \( Z = W \cdot X + b \)
- \( A = \text{activation}(Z) \)

**Activations:**
- Sigmoid: \( \sigma(z) = \frac{1}{1 + e^{-z}} \)
- ReLU: \( \max(0, z) \)
- Softmax (output layer):  
  \[
  \text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
  \]

---

**2. Loss Function (Cross Entropy):**  
\[
L = -\sum y_i \log(\hat{y}_i)
\]

---

**3. Backpropagation & Gradient Descent:**
- Use chain rule to compute gradients
- Update weights:  
  \[
  W = W - \alpha \cdot \frac{\partial L}{\partial W}
  \]
  where \( \alpha \) is the learning rate

---

**4. Accuracy:**
\[
\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Samples}} \times 100\%
\]

---

## ✅ Results

- **Validation Accuracy:** 83%
- **Training Method:** Full-batch gradient descent
- **Activation:** ReLU or Sigmoid
- **Output:** Softmax with cross-entropy loss

---

## 🚀 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/mnist-from-scratch.git
   cd mnist-from-scratch

