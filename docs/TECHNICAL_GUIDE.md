# Technical Guide - ANN Simple Image Pattern Recognition

This document explains the full technical workflow of the project in presentation-ready language.

## 1) End-to-End Workflow

1. Load dataset from `patterns.json`.
2. Convert each `5x5` binary pattern into a flattened length-25 vector.
3. Initialize ANN parameters (weights and biases).
4. Train with forward pass + MSE + backpropagation + gradient descent.
5. Evaluate using train/test accuracy, precision, recall, F1, and confusion matrix.
6. Use Streamlit UI to draw patterns, predict class, and manage dataset samples.

## 2) Project Architecture

### A) Core ANN Engine (`ann_pattern_recognition.py`)

- Defines constants:
  - `DEFAULT_DATASET_PATH`
  - `GRID_SIZE`
  - `INPUT_SIZE`
  - `DEFAULT_HIDDEN_SIZE`
- Implements ANN math with NumPy only.
- Handles dataset loading/saving and CLI execution.

Main components:

- `sigmoid(x)`:
  - Activation function mapping real numbers to `(0,1)`.
- `sigmoid_derivative(sigmoid_output)`:
  - Derivative used in backpropagation.
- `TrainingHistory`:
  - Stores epoch-wise loss and accuracy.
- `SimpleANN`:
  - `__init__`: initializes `w1, b1, w2, b2`.
  - `forward`: computes hidden and output activations.
  - `train`: performs iterative optimization.
  - `predict`: outputs class-1 probability.
- Data utilities:
  - `flatten_pattern`
  - `print_pattern`
  - `default_patterns`
  - `save_patterns`
  - `load_patterns`
  - `build_dataset`
- CLI:
  - `run_demo`
  - `parse_args`

### B) Frontend Layer (`streamlit_app.py`)

- Streamlit app for training, evaluation, and interactive prediction.
- Session state keeps model and UI state across reruns.

Key frontend blocks:

- Training controls:
  - epochs
  - learning rate
  - test split ratio
  - seed
- Evaluation outputs:
  - loss/accuracy trend
  - train/test accuracy
  - precision, recall, F1
  - confusion matrix
- Sample inspector:
  - collapsible section to browse dataset examples and preview layout.
- Pattern input:
  - drawing canvas -> converted to `5x5` binary grid.
  - prediction panel with confidence.
- Dataset management (non-presentation mode):
  - add sample
  - undo add
  - remove selected sample
  - reset to defaults

### C) Dataset Layer (`patterns.json`)

Each sample has:
- `label`: `0` (zero-like) or `1` (one-like)
- `pattern`: 5 rows x 5 columns of binary values (`0/1`)

This file is the single source of truth for training samples.

## 3) Neural Network Math (What Happens in Training)

### Forward Propagation

- Hidden pre-activation: `z1 = XW1 + b1`
- Hidden activation: `a1 = sigmoid(z1)`
- Output pre-activation: `z2 = a1W2 + b2`
- Output activation: `y_hat = sigmoid(z2)`

### Loss

- Mean Squared Error:
  - `MSE = mean((y_hat - y)^2)`

### Backpropagation

- Output gradient terms:
  - `d_loss_yhat = 2*(y_hat - y)/n`
  - `d_loss_z2 = d_loss_yhat * sigmoid_derivative(y_hat)`
- Parameter gradients:
  - `dW2 = a1^T * d_loss_z2`
  - `db2 = sum(d_loss_z2)`
- Hidden layer propagation:
  - `d_loss_a1 = d_loss_z2 * W2^T`
  - `d_loss_z1 = d_loss_a1 * sigmoid_derivative(a1)`
  - `dW1 = X^T * d_loss_z1`
  - `db1 = sum(d_loss_z1)`

### Gradient Descent Update

- `W = W - lr * dW`
- `b = b - lr * db`

## 4) Evaluation Metrics (How to Explain in Viva)

- Train Accuracy:
  - Correct predictions on training split.
- Test Accuracy:
  - Correct predictions on held-out split.
- Precision:
  - `TP / (TP + FP)`
  - Of predicted class-1 samples, how many are truly class 1.
- Recall:
  - `TP / (TP + FN)`
  - Of true class-1 samples, how many were found by model.
- F1-Score:
  - `2 * (Precision * Recall) / (Precision + Recall)`
  - Balance between precision and recall.
- Confusion Matrix:
  - Rows = true class, columns = predicted class.

## 5) Why You Sometimes See 100% Accuracy

- Small, simple, hand-crafted dataset can be memorized.
- Training accuracy alone is not enough for generalization.
- That is why train/test split was added.

## 6) Hyperparameters (Simple Explanation)

- Epochs:
  - Number of full passes over training data.
- Learning Rate (`eta`):
  - Step size in gradient descent updates.
- Random Seed:
  - Makes initialization/shuffling reproducible.
- Test Split Ratio:
  - Fraction of data used for testing.

## 7) Runtime Flow in Streamlit

1. User clicks **Train Model**.
2. Data is loaded and split by seed and test ratio.
3. ANN trains on training subset.
4. Metrics are computed and displayed.
5. Pattern input section is unlocked.
6. User draws pattern and clicks **Predict**.
7. App shows predicted class + confidence.
8. Optional: user edits dataset and retrains.

## 8) Key Design Decisions

- Manual ANN (no sklearn model APIs, no TensorFlow/PyTorch): educational transparency.
- Binary `5x5` dataset: easy to explain and demo live.
- JSON-based dataset: dynamic and editable during presentation.
- Collapsible sample inspector: cleaner UI while keeping depth.

## 9) Suggested Viva Script (Short)

- "Input is a flattened 5x5 binary image (25 features)."
- "Model is a 25-5-1 feedforward ANN with sigmoid activation."
- "I train using MSE loss and manual backpropagation in NumPy."
- "I evaluate with train/test accuracy, precision, recall, F1, and confusion matrix."
- "In the demo, I draw patterns, predict, and interpret output confidence."
- "I can add/remove patterns from dataset and retrain to analyze behavior changes."

## 10) Files You Should Know by Heart

- `ann_pattern_recognition.py` -> core model and training logic
- `streamlit_app.py` -> UI and experiment workflow
- `patterns.json` -> training samples
- `short_report_template.md` -> report content structure
- `presentation_script.md` -> speaking flow for the video
