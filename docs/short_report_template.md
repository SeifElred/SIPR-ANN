# Short Report - ANN for Simple Image Pattern Recognition

## Student Information
- **Name:** [Your Name]
- **Course / Section:** [Course Name]
- **Submission Date:** [Date]

## YouTube Presentation Link (5-10 minutes)
- **Video URL:** [Paste your YouTube link here]

## 1) Project Definition
This project implements an Artificial Neural Network (ANN) to classify simple 5x5 binary image patterns representing digit-like shapes.  
The goal is to demonstrate the fundamentals of image pattern recognition using a fully manual implementation in NumPy.

## 2) Dataset Used
- Input data consists of manually designed 5x5 patterns with binary pixel values (`0` or `1`).
- Each 5x5 pattern is flattened into a 25-element input vector.
- Two classes are used:
  - Class `0`: zero-like patterns
  - Class `1`: one-like patterns

## 3) ANN Design and Training
- Input layer: 25 neurons
- Hidden layer: 5 neurons
- Output layer: 1 neuron (binary output)
- Sigmoid activation is used in hidden and output layers.
- Loss function: Mean Squared Error (MSE)
- Weights and biases are initialized manually and updated using backpropagation over multiple epochs.

## 4) Results Obtained
- During training, loss decreases and classification accuracy improves.
- The model correctly classifies the provided training patterns after sufficient epochs.
- Output is expressed as a probability value (close to 0 for class `0`, close to 1 for class `1`).

## 5) Interpretation by Changing Dataset
To analyze model behavior, one pixel in a known pattern was modified and the prediction probability was compared before and after change.

- Observation example: Changing a central pixel slightly altered confidence.
- Interpretation: The network is sensitive to local binary features and pattern structure, showing how data representation influences ANN decisions.

## 6) Conclusion
The project successfully demonstrates that even a small manually implemented ANN can perform simple image pattern recognition.  
It provides practical understanding of forward propagation, error calculation, and backpropagation, and highlights how dataset changes affect predictions.

## 7) Future Work
- Increase dataset size and pattern diversity.
- Extend from binary to multi-class classification.
- Compare MSE with cross-entropy loss.
- Visualize learned weights for deeper interpretation.
