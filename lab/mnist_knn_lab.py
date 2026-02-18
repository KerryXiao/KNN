
"""
Script Execution: python mnist_knn_lab.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from keras.datasets import mnist

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Setup project paths using pathlib
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def reshape_and_scale(X: np.ndarray) -> np.ndarray:
    X_flat = X.reshape(X.shape[0], -1).astype(np.float32)
    X_flat /= 255.0
    return X_flat


def pick_best_k(
    X_train_flat: np.ndarray,
    y_train: np.ndarray,
    k_values: list[int],
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[int, dict[int, float]]:

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_flat,
        y_train,
        test_size=test_size,
        random_state=random_state,
        stratify=y_train,
    )

    scores: dict[int, float] = {}

    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k, weights="distance")
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        scores[k] = acc
        print(f"k={k:2d} validation accuracy={acc:.4f}")

    best_k = max(scores, key=scores.get)
    print(f"\nBest k: {best_k} (validation accuracy={scores[best_k]:.4f})\n")

    # Save validation results
    results_path = OUTPUT_DIR / "validation_results.txt"
    with results_path.open("w") as f:
        for k, acc in scores.items():
            f.write(f"k={k}, accuracy={acc:.4f}\n")

    return best_k, scores


def plot_confusion_matrix(cm: np.ndarray, filename: Path) -> None:
    """Save confusion matrix plot to file."""
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (MNIST kNN)")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def main() -> None:
    # 1. Load MNIST
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # 2. Shapes of X_train, X_test, X_train[i], X_test[i], y_train, and y_test
    print("=== Shapes ===")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("Single image shape:", X_train[0].shape)
    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)
    print()

    # 3. Reshaping
    X_train_flat = reshape_and_scale(X_train)
    X_test_flat = reshape_and_scale(X_test)

    print("After reshape:")
    print("X_train_flat:", X_train_flat.shape)
    print("X_test_flat :", X_test_flat.shape)
    print()

    # 4. Select best k
    k_values = list(range(1, 22, 2))
    best_k, scores = pick_best_k(X_train_flat, y_train, k_values)

    # Optimal Number of Neighbors
    model = KNeighborsClassifier(n_neighbors=best_k, weights="distance")
    model.fit(X_train_flat, y_train)

    y_pred = model.predict(X_test_flat)
    acc = accuracy_score(y_test, y_pred)

    # 5. Validation Accuracy
    print("Test Accuracy:", acc)

    # 6. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix:")
    print(cm)

    # Save confusion matrix plot
    cm_plot_path = OUTPUT_DIR / "confusion_matrix.png"
    plot_confusion_matrix(cm, cm_plot_path)

    # Save classification report
    report_path = OUTPUT_DIR / "classification_report.txt"
    with report_path.open("w") as f:
        f.write(classification_report(y_test, y_pred))

    print("\nOutputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()