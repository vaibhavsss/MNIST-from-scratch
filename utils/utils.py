# utils.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(train_csv_path, test_csv_path=None):
    """
        - X_train: features
        - y_train: labels (as integers)
        - X_val: validation features
        - y_val: validation labels
    """
    df = pd.read_csv(train_csv_path)
    X = df.drop("label", axis=1).values
    y = df["label"].values

    # Normalize pixel values to range [0, 1]
    X = X / 255.0

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if test_csv_path:
        test_df = pd.read_csv(test_csv_path)
        X_test = test_df.values / 255.0
        return X_train, y_train, X_val, y_val, X_test

    return X_train, y_train, X_val, y_val

def one_hot_encode(y, num_classes=10):
    """
    Converts labels to one-hot encoded vectors.
    """
    encoded = np.zeros((y.shape[0], num_classes))
    encoded[np.arange(y.shape[0]), y] = 1
    return encoded

def accuracy(predictions, labels):
    """
    Calculates accuracy: percentage of correct predictions.
    """
    preds = np.argmax(predictions, axis=1)
    return np.mean(preds == labels) * 100

def softmax(z):
    """
    Computes softmax for a batch of logits.
    """
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # stability trick
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
