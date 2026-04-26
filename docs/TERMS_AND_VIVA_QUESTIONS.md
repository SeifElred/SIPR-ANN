# Terms and Viva Questions Study Sheet

Use this as a pre-presentation revision guide.

## 1) Core ANN Terms

- **Artificial Neural Network (ANN)**  
  A computational model made of connected layers (input, hidden, output) that learns patterns from data.

- **Neuron**  
  A unit that computes weighted sum + bias, then applies an activation function.

- **Input Layer**  
  First layer receiving features. In this project: 25 inputs (flattened 5x5).

- **Hidden Layer**  
  Intermediate layer learning internal representations. In this project: 5 neurons.

- **Output Layer**  
  Final layer producing prediction. Here: 1 neuron for binary classification.

- **Feature Vector**  
  Numerical representation of input data. Here: 5x5 grid flattened to length 25.

- **Flattening**  
  Converting a 2D grid into a 1D vector.

- **Weight (W)**  
  Learnable parameter controlling feature influence.

- **Bias (b)**  
  Learnable offset added before activation.

- **Activation Function (Sigmoid)**  
  `sigmoid(x) = 1/(1+e^-x)` maps values to (0,1), used as probability-like output.

- **Forward Propagation**  
  Compute outputs from input to output through network layers.

- **Prediction Threshold**  
  Rule converting probability to class label (here: threshold 0.5).

- **Loss Function (MSE)**  
  Measures prediction error; lower is better.

- **Mean Squared Error (MSE)**  
  Average squared difference between predicted and true values.

- **Backpropagation**  
  Computes gradients of loss w.r.t. parameters using chain rule.

- **Gradient**  
  Direction and magnitude telling how to adjust parameters to reduce loss.

- **Gradient Descent**  
  Parameter update rule: subtract learning rate × gradient.

- **Learning Rate (eta)**  
  Step size of each parameter update.

- **Epoch**  
  One full pass through training data.

- **Random Seed**  
  Fixed value to make initialization/shuffling reproducible.

- **Train/Test Split**  
  Partition of data into learning subset and evaluation subset.

- **Overfitting**  
  Model memorizes training data but generalizes poorly.

- **Generalization**  
  Model’s ability to perform on unseen data.

- **Confusion Matrix**  
  Table of true vs predicted classes.

- **True Positive (TP)**  
  Correctly predicted class 1.

- **True Negative (TN)**  
  Correctly predicted class 0.

- **False Positive (FP)**  
  Predicted class 1 but true class is 0.

- **False Negative (FN)**  
  Predicted class 0 but true class is 1.

- **Accuracy**  
  `(TP + TN) / Total`

- **Precision**  
  `TP / (TP + FP)`  
  Of predicted class-1 samples, how many are truly class 1.

- **Recall**  
  `TP / (TP + FN)`  
  Of true class-1 samples, how many were detected.

- **F1-Score**  
  Harmonic mean of precision and recall.

- **Binary Classification**  
  Classification into two classes (0 and 1).

## 2) Project-Specific Terms

- **5x5 Binary Pattern**  
  Small image-like grid using only 0/1 values.

- **Class 0 (zero-like)**  
  Patterns shaped like digit 0.

- **Class 1 (one-like)**  
  Patterns shaped like digit 1.

- **Canvas-to-Grid Conversion**  
  Frontend drawing is converted to grayscale, then downsampled to 5x5 and thresholded to binary.

- **`patterns.json`**  
  Dataset storage file for all samples and labels.

- **`default_patterns()`**  
  Built-in fallback dataset used for reset/missing file.

- **Presentation Mode**  
  Cleaner UI mode hiding nonessential controls.

## 3) Likely Viva Questions (With Strong Short Answers)

- **Q1: Why 25 input neurons?**  
  Because each sample is a 5x5 grid, flattened to 25 values.

- **Q2: Why use sigmoid?**  
  It gives smooth differentiable outputs in (0,1), suitable for binary probability.

- **Q3: Why MSE instead of cross-entropy?**  
  This project is educational/manual; MSE keeps implementation simple while still demonstrating learning.

- **Q4: Why does accuracy become 100% sometimes?**  
  Small handcrafted datasets are easy to memorize; that is why train/test split and extra metrics are important.

- **Q5: What is the role of learning rate?**  
  It controls update size; too small is slow, too large can destabilize training.

- **Q6: What does random seed do?**  
  Fixes randomness in weight initialization and shuffling for reproducible runs.

- **Q7: What happens in backpropagation?**  
  We compute error gradients layer by layer and update weights/biases to reduce loss.

- **Q8: How is a drawing converted to model input?**  
  Canvas image -> grayscale -> split into 5x5 cells -> threshold each cell to 0/1.

- **Q9: Why include precision/recall/F1, not only accuracy?**  
  They show class-wise behavior and are more informative when class distribution changes.

- **Q10: What does confusion matrix tell you?**  
  Where the model is correct/incorrect by class (TP/TN/FP/FN).

- **Q11: What is overfitting in your project context?**  
  High train accuracy with weaker test performance on held-out patterns.

- **Q12: How can you improve this project?**  
  Increase dataset diversity, test cross-entropy, add regularization, and extend to multi-class digits.

- **Q13: Why manual NumPy implementation?**  
  To demonstrate understanding of ANN internals rather than using black-box libraries.

- **Q14: What is the exact architecture?**  
  Fully connected 25-5-1 network with sigmoid activation.

- **Q15: What is the prediction rule?**  
  If output probability >= 0.5 => class 1, else class 0.

## 4) Quick Memory Checklist Before Presentation

- Explain data representation (5x5 -> 25).
- Explain forward pass equations conceptually.
- Explain loss and backprop at high level.
- Explain each metric in one sentence.
- Show one live prediction and interpret confidence.
- Show one dataset change, retrain, and interpret behavior.
