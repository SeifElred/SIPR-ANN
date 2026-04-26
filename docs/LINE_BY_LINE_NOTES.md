# Line-by-Line Notes (Practical Walkthrough)

This guide explains the code in study order, block by block, so you can confidently explain it in viva.

## 1) `ann_pattern_recognition.py`

### Imports and constants
- `argparse`, `json`, `os`: CLI args and dataset file I/O.
- `dataclass`: clean container for training history.
- `typing`: type hints for readability.
- `numpy`: all math and matrix operations.
- Constants:
  - `DEFAULT_DATASET_PATH` -> default JSON source.
  - `GRID_SIZE`, `INPUT_SIZE`, `DEFAULT_HIDDEN_SIZE` -> avoid magic numbers.

### Activation functions
- `sigmoid(x)`: converts linear output to bounded probability-like values.
- `sigmoid_derivative(sigmoid_output)`: needed for gradient computation in backprop.

### `TrainingHistory`
- Stores `losses` and `accuracies` each epoch.
- Returned by `train` and used for frontend plotting.

### `SimpleANN.__init__`
- Saves learning rate.
- Uses seeded RNG (`np.random.default_rng(seed)`) for reproducibility.
- Initializes:
  - `w1`: input-to-hidden weights (`25 x 5`)
  - `b1`: hidden bias (`1 x 5`)
  - `w2`: hidden-to-output weights (`5 x 1`)
  - `b2`: output bias (`1 x 1`)

### `SimpleANN.forward`
- `z1 = XW1 + b1`
- `a1 = sigmoid(z1)`
- `z2 = a1W2 + b2`
- `a2 = sigmoid(z2)` -> predicted probability of class 1.
- Returns `(a1, a2)`.

### `SimpleANN.train`
- Loops over `epochs`.
- Calls `forward` to get predictions.
- Computes MSE loss.
- Converts probabilities to class labels with threshold `0.5`.
- Computes accuracy.
- Backprop:
  - output layer delta
  - hidden layer delta
  - gradients for all parameters
- Gradient descent update on all weights and biases.
- Optional logging every `verbose_every`.
- Returns `TrainingHistory`.

### `SimpleANN.predict`
- Forward-only inference path.
- Returns output probability.

### Data helpers
- `flatten_pattern`: reshapes one `5x5` to `(1,25)` vector.
- `print_pattern`: console visualization (`#` and `.`).

### `default_patterns`
- Built-in fallback dataset used if JSON file missing.

### `save_patterns` / `load_patterns`
- Save and read dataset JSON.
- If dataset path doesn’t exist:
  - create from `default_patterns`
  - save automatically

### `build_dataset`
- Reads patterns dictionary.
- For each sample:
  - flatten pattern to length 25
  - append label
- Returns:
  - `x`: shape `(N,25)`
  - `y`: shape `(N,1)`
  - raw `patterns` dictionary

### `run_demo`
- Loads data.
- Creates model.
- Trains model.
- Prints per-sample prediction table.
- Performs interpretation test:
  - flips a center pixel in one class-0 sample
  - compares probability before/after
  - prints interpretation sentence

### CLI
- `parse_args` reads:
  - `--epochs`
  - `--lr`
  - `--seed`
  - `--dataset`
- Main block executes `run_demo(...)`.

---

## 2) `streamlit_app.py`

### Imports
- `io` + `PIL.Image`: image conversion for preview.
- `numpy`, `pandas`, `streamlit`.
- `st_canvas` from `streamlit_drawable_canvas` (if installed).
- Imports backend functions and constants from `ann_pattern_recognition.py`.

### App header
- `st.set_page_config(...)` sets title/layout.
- Title + caption rendered.
- `presentation_mode` toggle controls reduced UI.

### Session state initialization
- Creates persistent keys:
  - `grid`, `trained_model`, `last_metrics`, `train_history`, `last_dataset`,
  - `canvas_key`, `last_added_sample_name`, `last_prediction`, `last_eval`
- Initializes per-cell fallback checkbox states (`pixel_r_c`).

### Utility functions
- `update_checkboxes_from_grid`: sync helper for fallback input mode.
- `pattern_preview_image`: converts 5x5 pattern to enlarged PNG buffer.
- `pattern_ascii_layout`: text visualization of selected sample.
- `canvas_to_grid`: converts free-draw image to 5x5 binary using grayscale threshold.

### Layout columns
- `left, right = st.columns([1.2, 1])`

### Left column: Training and evaluation

#### Controls
- `epochs`, `learning rate`, `test ratio`, `seed`.

#### Train button logic
- Loads dataset (`build_dataset`).
- Shuffles using seed.
- Splits into train and test by selected ratio.
- Trains on train split.
- Computes:
  - train accuracy
  - test accuracy
  - confusion components (TP/TN/FP/FN) on full dataset
  - precision/recall/F1
- Stores model/history/metrics in `session_state`.

#### Reset Grid button
- Clears current input grid and canvas state.

#### Metrics panel
- Shows:
  - final loss
  - train accuracy
  - TP/TN
  - test accuracy
  - precision/recall/F1 (with info tooltips)

#### Training curve
- Plots loss + accuracy from `train_history`.

#### Training samples expander
- Displays count summary by class.
- Select a sample from dropdown.
- Shows:
  - enlarged visual layout
  - ID and label
  - binary text grid
  - model prediction for that sample (if model trained)

#### Output quality tables
- If trained:
  - per-sample prediction table
  - confusion matrix
- In presentation mode these are tabbed; otherwise separate sections.

### Right column: Pattern input and prediction

#### Locked before training
- Shows info message until model is trained.

#### Input mode
- Preferred: canvas draw mode.
- Fallback: 5x5 checkbox grid if canvas package unavailable.

#### Canvas controls
- Brush size slider.
- Clear canvas button.
- Draw area (240x240).
- Conversion to 5x5 grid every rerun.
- Shows 5x5 preview image used by model.

#### Predict button
- Flattens current grid to `(1,25)`.
- Calls model `predict`.
- Thresholds at `0.5`.
- Stores prediction result.

#### Prediction panel
- Shows:
  - class label
  - confidence
  - `P(class 1)`
  - progress bar

#### Dataset editing section (non-presentation mode)
- Add sample (name + label + current 5x5 pattern).
- Undo last added sample.
- Remove selected sample.
- Reset dataset to defaults.

---

## 3) Quick Viva Recap

- Input representation: `5x5` binary -> `25` features.
- Model: `25-5-1`, sigmoid activation.
- Loss: MSE.
- Optimization: manual backprop + gradient descent.
- Evaluation: accuracy, precision, recall, F1, confusion matrix.
- Demo: draw pattern, predict, modify dataset, retrain, interpret.
