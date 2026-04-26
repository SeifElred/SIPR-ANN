# Application of Artificial Neural Networks for Simple Image Pattern Recognition

This project is a clean, manual implementation of a small Artificial Neural Network (ANN) for recognizing simple 5x5 binary image patterns.  
It is designed for coursework demonstration and presentation.

## Project Objective
- Build an ANN that classifies simple handwritten-style patterns.
- Show the full learning pipeline with no ML framework (NumPy only).
- Demonstrate interpretation by changing input pixels and observing output.

## Architecture
- **Input layer:** 25 neurons (flattened 5x5 image)
- **Hidden layer:** 5 neurons
- **Output layer:** 1 neuron (binary class)
- **Activation:** Sigmoid
- **Loss:** Mean Squared Error (MSE)

## Dataset
Manually defined binary patterns for two classes:
- Class `0`: zero-like patterns
- Class `1`: one-like patterns

Each image is a 5x5 grid with pixel values in `{0, 1}` and is flattened to length 25.

## How to Run (CLI)
1. Install Python 3.10+.
2. Install dependency:

```bash
pip install numpy
```

3. Run training/demo:

```bash
python ann_pattern_recognition.py
```

Optional parameters:

```bash
python ann_pattern_recognition.py --epochs 7000 --lr 0.4 --seed 7
```

Use a custom dataset file:

```bash
python ann_pattern_recognition.py --dataset patterns.json
```

## Interactive Frontend (Recommended for Demo)
Install Streamlit:

```bash
pip install streamlit
```

Run:

```bash
streamlit run streamlit_app.py
```

Frontend features:
- Train with adjustable epochs, learning rate, and seed
- Visual canvas input with automatic 5x5 conversion
- Predict drawn pattern after training
- Add newly drawn samples directly to `patterns.json`
- Re-train instantly to interpret dataset changes live
- Show training curves, per-sample outputs, and confusion matrix

## Documentation
- `docs/README.md` -> documentation index
- `docs/TECHNICAL_GUIDE.md` -> full technical explanation
- `docs/LINE_BY_LINE_NOTES.md` -> code walkthrough notes
- `docs/presentation_script.md` -> presentation script
- `docs/short_report_template.md` -> report template

## What the Program Shows
- Training progress with loss and accuracy
- Predictions for each training pattern
- Interpretation demo:
  - modifies one pixel in a sample pattern
  - compares output probability before/after change
  - explains model sensitivity

## Suggested 5-10 Minute Presentation Flow
1. Problem definition and objective
2. Dataset representation (5x5 binary grid)
3. Network architecture
4. Forward pass and backpropagation (manual NumPy code)
5. Streamlit frontend demo: draw, train, predict
6. Add/modify sample in dataset and retrain to interpret changes
7. Conclusion and future improvements
