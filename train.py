"""
train.py
========
Train a gesture classifier on collected landmark data.

Workflow:
    1. Run `collect_data.py` to gather landmark CSVs for each gesture.
    2. Run this script to train and save the model.
    3. The saved model is loaded automatically by main.py.

Usage:
    python train.py                          # Train with default settings
    python train.py --data data/landmarks   # Custom data directory
    python train.py --model svm             # Choose classifier
    python train.py --test-size 0.2         # Train/test split ratio
"""

import argparse
import os
import pickle
import glob

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder


CLASSIFIERS = {
    "rf": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "svm": SVC(kernel="rbf", probability=True, C=10, gamma="scale"),
    "mlp": MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42),
}


def load_dataset(data_dir: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load landmark CSVs from data_dir.
    Each CSV file should be named <gesture_label>.csv with rows of 63 floats.

    Returns:
        X: Feature matrix (N, 63)
        y: Encoded integer labels (N,)
        label_names: List of class names
    """
    X_list, y_list = [], []
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in '{data_dir}'.\n"
            "Run collect_data.py first to gather training data."
        )

    label_encoder = LabelEncoder()
    labels = []

    for csv_path in csv_files:
        label = os.path.splitext(os.path.basename(csv_path))[0]
        data = np.loadtxt(csv_path, delimiter=",")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        X_list.append(data)
        labels.extend([label] * len(data))

    X = np.vstack(X_list)
    y_encoded = label_encoder.fit_transform(labels)
    label_names = list(label_encoder.classes_)

    print(f"[INFO] Loaded {len(X)} samples across {len(label_names)} classes.")
    print(f"[INFO] Classes: {label_names}")
    return X, y_encoded, label_names


def train(args):
    X, y, label_names = load_dataset(args.data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )

    clf = CLASSIFIERS[args.model]
    print(f"\n[INFO] Training {args.model.upper()} classifier...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n[RESULTS] Test Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_names))

    # Save model
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump({"model": clf, "labels": label_names}, f)
    print(f"[INFO] Model saved to: {args.output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train gesture classifier")
    parser.add_argument("--data", default="data/landmarks", help="Landmark CSV directory")
    parser.add_argument(
        "--model",
        choices=["rf", "svm", "mlp"],
        default="rf",
        help="Classifier type: rf=RandomForest, svm=SVM, mlp=MLP (default: rf)",
    )
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--output", default="models/gesture_model.pkl")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
