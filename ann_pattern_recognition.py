import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


DEFAULT_DATASET_PATH = "patterns.json"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(sigmoid_output: np.ndarray) -> np.ndarray:
    return sigmoid_output * (1.0 - sigmoid_output)


@dataclass
class TrainingHistory:
    losses: List[float]
    accuracies: List[float]


class SimpleANN:
    """
    A tiny fully-connected ANN:
    - Input  : 25 neurons (5x5 binary image)
    - Hidden : 5 neurons
    - Output : 1 neuron (binary classification)
    """

    def __init__(self, input_size: int = 25, hidden_size: int = 5, learning_rate: float = 0.5, seed: int = 42) -> None:
        self.learning_rate = learning_rate
        rng = np.random.default_rng(seed)

        # Small random initialization improves stability.
        self.w1 = rng.normal(0, 0.5, size=(input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = rng.normal(0, 0.5, size=(hidden_size, 1))
        self.b2 = np.zeros((1, 1))

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        z1 = np.dot(x, self.w1) + self.b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, self.w2) + self.b2
        a2 = sigmoid(z2)
        return a1, a2

    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 5000,
        verbose_every: int = 500,
        verbose: bool = True,
    ) -> TrainingHistory:
        history = TrainingHistory(losses=[], accuracies=[])
        n = x.shape[0]

        for epoch in range(1, epochs + 1):
            # Forward pass
            a1, y_hat = self.forward(x)

            # Mean Squared Error
            loss = np.mean((y_hat - y) ** 2)

            # Accuracy for quick interpretation
            predictions = (y_hat >= 0.5).astype(int)
            accuracy = np.mean(predictions == y)

            history.losses.append(float(loss))
            history.accuracies.append(float(accuracy))

            # Backpropagation
            d_loss_yhat = 2.0 * (y_hat - y) / n
            d_yhat_z2 = sigmoid_derivative(y_hat)
            d_loss_z2 = d_loss_yhat * d_yhat_z2

            d_loss_w2 = np.dot(a1.T, d_loss_z2)
            d_loss_b2 = np.sum(d_loss_z2, axis=0, keepdims=True)

            d_loss_a1 = np.dot(d_loss_z2, self.w2.T)
            d_a1_z1 = sigmoid_derivative(a1)
            d_loss_z1 = d_loss_a1 * d_a1_z1

            d_loss_w1 = np.dot(x.T, d_loss_z1)
            d_loss_b1 = np.sum(d_loss_z1, axis=0, keepdims=True)

            # Gradient descent update
            self.w2 -= self.learning_rate * d_loss_w2
            self.b2 -= self.learning_rate * d_loss_b2
            self.w1 -= self.learning_rate * d_loss_w1
            self.b1 -= self.learning_rate * d_loss_b1

            if verbose and (epoch % verbose_every == 0 or epoch == 1 or epoch == epochs):
                print(f"Epoch {epoch:4d}/{epochs} | Loss: {loss:.6f} | Accuracy: {accuracy * 100:.2f}%")

        return history

    def predict(self, x: np.ndarray) -> np.ndarray:
        _, y_hat = self.forward(x)
        return y_hat


def flatten_pattern(pattern_5x5: List[List[int]]) -> np.ndarray:
    arr = np.array(pattern_5x5, dtype=float)
    return arr.reshape(1, 25)


def print_pattern(pattern_5x5: List[List[int]]) -> None:
    for row in pattern_5x5:
        print(" ".join("#" if p == 1 else "." for p in row))


def default_patterns() -> Dict[str, Dict[str, List[List[int]] | int]]:
    return {
        "zero_A": {
            "label": 0,
            "pattern": [
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1],
            ],
        },
        "zero_B": {
            "label": 0,
            "pattern": [
                [0, 1, 1, 1, 0],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [0, 1, 1, 1, 0],
            ],
        },
        "zero_C": {
            "label": 0,
            "pattern": [
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 1, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1],
            ],
        },
        "one_A": {
            "label": 1,
            "pattern": [
                [0, 0, 1, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0],
            ],
        },
        "one_B": {
            "label": 1,
            "pattern": [
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
            ],
        },
        "one_C": {
            "label": 1,
            "pattern": [
                [0, 1, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0],
            ],
        },
        "zero_I": {
            "label": 0,
            "pattern": [
                [1, 1, 1, 1, 0],
                [1, 0, 0, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [0, 1, 1, 1, 1],
            ],
        },
        "zero_J": {
            "label": 0,
            "pattern": [
                [1, 1, 1, 1, 1],
                [1, 0, 1, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 1, 0, 1],
                [1, 1, 1, 1, 1],
            ],
        },
        "zero_K": {
            "label": 0,
            "pattern": [
                [0, 1, 1, 1, 0],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1],
            ],
        },
        "zero_L": {
            "label": 0,
            "pattern": [
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1],
            ],
        },
        "one_H": {
            "label": 1,
            "pattern": [
                [0, 0, 1, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [1, 1, 1, 1, 0],
            ],
        },
        "one_I": {
            "label": 1,
            "pattern": [
                [0, 1, 1, 1, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
            ],
        },
        "one_J": {
            "label": 1,
            "pattern": [
                [0, 0, 0, 1, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0],
                [0, 1, 1, 1, 0],
            ],
        },
        "one_K": {
            "label": 1,
            "pattern": [
                [0, 1, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0],
            ],
        },
    }


def save_patterns(patterns: Dict[str, Dict[str, List[List[int]] | int]], dataset_path: str = DEFAULT_DATASET_PATH) -> None:
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(patterns, f, indent=2)


def load_patterns(dataset_path: str = DEFAULT_DATASET_PATH) -> Dict[str, Dict[str, List[List[int]] | int]]:
    if not os.path.exists(dataset_path):
        patterns = default_patterns()
        save_patterns(patterns, dataset_path=dataset_path)
        return patterns
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataset(dataset_path: str = DEFAULT_DATASET_PATH) -> Tuple[np.ndarray, np.ndarray, Dict[str, Dict[str, List[List[int]] | int]]]:
    patterns = load_patterns(dataset_path=dataset_path)

    x_list = []
    y_list = []
    for _, item in patterns.items():
        p = item["pattern"]
        label = int(item["label"])
        x_list.append(np.array(p, dtype=float).reshape(25))
        y_list.append(label)

    x = np.array(x_list, dtype=float)
    y = np.array(y_list, dtype=float).reshape(-1, 1)
    return x, y, patterns


def run_demo(epochs: int, lr: float, seed: int, dataset_path: str = DEFAULT_DATASET_PATH) -> None:
    x, y, patterns = build_dataset(dataset_path=dataset_path)
    model = SimpleANN(learning_rate=lr, seed=seed)

    print("\nTraining model...")
    model.train(x, y, epochs=epochs, verbose_every=max(epochs // 10, 1))

    print("\n=== In-sample Predictions ===")
    for name, item in patterns.items():
        pattern = item["pattern"]
        truth = int(item["label"])
        vec = flatten_pattern(pattern)
        prob = float(model.predict(vec)[0, 0])
        pred = 1 if prob >= 0.5 else 0
        print(f"{name:7s} | truth={truth} | pred={pred} | prob_one={prob:.4f}")

    # Dataset interpretation section for your presentation:
    # Flip one pixel in an existing pattern to see prediction sensitivity.
    print("\n=== Interpretation: Modified Pattern Test ===")
    target_name = next((name for name, item in patterns.items() if int(item["label"]) == 0), None)
    if target_name is None:
        print("No class-0 sample found for interpretation demo.")
        return

    target_pattern = patterns[target_name]["pattern"]
    modified = [row[:] for row in target_pattern]
    modified[2][2] = 1  # Introduce a center pixel change.

    print(f"Original {target_name}:")
    print_pattern(target_pattern)
    original_prob = float(model.predict(flatten_pattern(target_pattern))[0, 0])
    print(f"Model prob_one on original: {original_prob:.4f}")

    print("\nModified zero_A (center pixel flipped to 1):")
    print_pattern(modified)
    modified_prob = float(model.predict(flatten_pattern(modified))[0, 0])
    print(f"Model prob_one on modified: {modified_prob:.4f}")

    print("\nInterpretation:")
    if modified_prob > original_prob:
        print("- The pixel change made the pattern more similar to class '1'.")
    elif modified_prob < original_prob:
        print("- The pixel change made the pattern more similar to class '0'.")
    else:
        print("- The pixel change had negligible effect in this trained model.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple ANN for 5x5 pattern recognition.")
    parser.add_argument("--epochs", type=int, default=5000, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.5, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_PATH, help="Path to dataset JSON.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(epochs=args.epochs, lr=args.lr, seed=args.seed, dataset_path=args.dataset)
