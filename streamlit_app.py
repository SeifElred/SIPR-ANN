import io

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    from streamlit_drawable_canvas import st_canvas
except Exception:
    st_canvas = None

from ann_pattern_recognition import (SimpleANN, build_dataset, default_patterns,
                                     load_patterns, save_patterns)

st.set_page_config(page_title="ANN Pattern Recognition", layout="wide")
st.title("ANN Simple Image Pattern Recognition (5x5)")
st.caption("Train, draw, and predict with a clean interface.")
presentation_mode = st.toggle("Presentation Mode", value=True, help="Cleaner view for live presentation.")


if "grid" not in st.session_state:
    st.session_state.grid = [[0 for _ in range(5)] for _ in range(5)]
if "trained_model" not in st.session_state:
    st.session_state.trained_model = None
if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = None
if "train_history" not in st.session_state:
    st.session_state.train_history = None
if "last_dataset" not in st.session_state:
    st.session_state.last_dataset = None
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0
if "last_added_sample_name" not in st.session_state:
    st.session_state.last_added_sample_name = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "last_eval" not in st.session_state:
    st.session_state.last_eval = None
for r in range(5):
    for c in range(5):
        key = f"pixel_{r}_{c}"
        if key not in st.session_state:
            st.session_state[key] = bool(st.session_state.grid[r][c])


def update_checkboxes_from_grid() -> None:
    for rr in range(5):
        for cc in range(5):
            st.session_state[f"pixel_{rr}_{cc}"] = bool(st.session_state.grid[rr][cc])


def pattern_preview_image(pattern: list[list[int]], size_px: int = 200) -> io.BytesIO:
    arr = (np.array(pattern, dtype=np.uint8) * 255).astype(np.uint8)
    img = Image.fromarray(arr).resize((size_px, size_px), resample=Image.Resampling.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def pattern_ascii_layout(pattern: list[list[int]]) -> str:
    return "\n".join(" ".join("#" if v else "." for v in row) for row in pattern)


def canvas_to_grid(image_data: np.ndarray) -> list[list[int]]:
    # image_data is RGBA; convert to grayscale then downsample to 5x5.
    gray = np.mean(image_data[:, :, :3], axis=2)
    h, w = gray.shape
    out = np.zeros((5, 5), dtype=int)
    for r in range(5):
        for c in range(5):
            r0 = int(r * h / 5)
            r1 = int((r + 1) * h / 5)
            c0 = int(c * w / 5)
            c1 = int((c + 1) * w / 5)
            cell = gray[r0:r1, c0:c1]
            # Black strokes reduce intensity; threshold for active pixel.
            out[r, c] = 1 if np.mean(cell) < 220 else 0
    return out.tolist()


left, right = st.columns([1.2, 1])

with left:
    st.subheader("Training")
    dataset_path = "patterns.json"
    epochs = st.slider(
        "Epochs",
        min_value=200,
        max_value=3000,
        value=1200,
        step=100,
        help="More epochs = more training passes on the same dataset.",
    )
    lr = st.slider(
        "Learning Rate (eta)",
        min_value=0.05,
        max_value=0.8,
        value=0.3,
        step=0.01,
        help="Step size for weight updates during gradient descent.",
    )
    test_ratio = st.slider(
        "Test Split Ratio",
        min_value=0.1,
        max_value=0.4,
        value=0.2,
        step=0.05,
        help="Fraction of samples held out for test evaluation.",
    )
    seed = 42 if presentation_mode else st.number_input(
        "Random Seed",
        min_value=0,
        max_value=9999,
        value=42,
        step=1,
        help="Controls random weight initialization for reproducible runs.",
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Train Model", use_container_width=True):
            x, y, patterns = build_dataset(dataset_path=dataset_path)
            rng = np.random.default_rng(int(seed))
            idx = np.arange(len(x))
            rng.shuffle(idx)
            x_shuf = x[idx]
            y_shuf = y[idx]

            n_samples = len(x_shuf)
            split_idx = max(1, int(round(n_samples * (1.0 - float(test_ratio)))))
            split_idx = min(split_idx, n_samples - 1) if n_samples > 1 else 1
            x_train = x_shuf[:split_idx]
            y_train = y_shuf[:split_idx]
            x_test = x_shuf[split_idx:]
            y_test = y_shuf[split_idx:]

            model = SimpleANN(learning_rate=float(lr), seed=int(seed))
            history = model.train(
                x_train,
                y_train,
                epochs=int(epochs),
                verbose_every=max(epochs // 10, 1),
                verbose=False,
            )
            y_prob_train = model.predict(x_train)
            y_pred_train = (y_prob_train >= 0.5).astype(int)
            train_acc = float(np.mean(y_pred_train == y_train))

            y_prob_test = model.predict(x_test)
            y_pred_test = (y_prob_test >= 0.5).astype(int)
            test_acc = float(np.mean(y_pred_test == y_test))

            # Keep confusion matrix based on full dataset for readability.
            y_prob_all = model.predict(x)
            y_pred_all = (y_prob_all >= 0.5).astype(int)
            tp = int(np.sum((y_pred_all == 1) & (y == 1)))
            tn = int(np.sum((y_pred_all == 0) & (y == 0)))
            fp = int(np.sum((y_pred_all == 1) & (y == 0)))
            fn = int(np.sum((y_pred_all == 0) & (y == 1)))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            st.session_state.trained_model = model
            st.session_state.train_history = history
            st.session_state.last_dataset = (x, y, patterns)
            st.session_state.last_eval = {
                "train_acc": train_acc,
                "test_acc": test_acc,
                "train_n": int(len(x_train)),
                "test_n": int(len(x_test)),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
            st.session_state.last_metrics = {
                "final_loss": history.losses[-1],
                "final_accuracy": train_acc,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
            }

    with c2:
        if st.button("Reset Grid", use_container_width=True):
            st.session_state.grid = [[0 for _ in range(5)] for _ in range(5)]
            update_checkboxes_from_grid()
            st.session_state.canvas_key += 1

    if st.session_state.last_metrics:
        m = st.session_state.last_metrics
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Final Loss (MSE)", f"{m['final_loss']:.6f}")
        col_b.metric("Train Accuracy", f"{m['final_accuracy'] * 100:.2f}%")
        col_c.metric("True Positives", m["tp"])
        col_d.metric("True Negatives", m["tn"])
        if st.session_state.last_eval is not None:
            ev = st.session_state.last_eval
            st.caption(
                f"Test Accuracy: {ev['test_acc'] * 100:.2f}% "
                f"(train n={ev['train_n']}, test n={ev['test_n']})"
            )
            met1, met2, met3 = st.columns(3)
            met1.metric(
                "Precision",
                f"{ev['precision']:.3f}",
                help="Of all samples predicted as class 1, what fraction were truly class 1? "
                "Formula: TP / (TP + FP). Uses the full-dataset confusion counts after training.",
            )
            met2.metric(
                "Recall",
                f"{ev['recall']:.3f}",
                help="Of all samples that are truly class 1, what fraction did the model predict as class 1? "
                "Formula: TP / (TP + FN). Uses the full-dataset confusion counts after training.",
            )
            met3.metric(
                "F1-Score",
                f"{ev['f1']:.3f}",
                help="Harmonic mean of precision and recall (balances both). "
                "Formula: 2 · (precision · recall) / (precision + recall). "
                "Higher is better when both precision and recall matter.",
            )

    if st.session_state.train_history is not None:
        history_df = pd.DataFrame(
            {
                "epoch": np.arange(1, len(st.session_state.train_history.losses) + 1),
                "loss": st.session_state.train_history.losses,
                "accuracy": st.session_state.train_history.accuracies,
            }
        ).set_index("epoch")
        st.line_chart(history_df[["loss", "accuracy"]], use_container_width=True)

    patterns = load_patterns(dataset_path=dataset_path)
    name_list = sorted(patterns.keys())
    n_total = len(patterns)
    n_class0 = sum(1 for k in patterns if int(patterns[k]["label"]) == 0)
    n_class1 = n_total - n_class0
    with st.expander(
        f"Training Samples ({n_total} · {n_class0} class 0, {n_class1} class 1)",
        expanded=False,
    ):
        st.caption("Choose a sample to inspect its class label, grid layout, and model output.")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total samples", n_total)
        m2.metric("Class 0 (zero-like)", n_class0)
        m3.metric("Class 1 (one-like)", n_class1)
        selected_name = st.selectbox(
            "Sample",
            options=name_list,
            index=0,
            format_func=lambda n: f"{n}  (class {int(patterns[n]['label'])})",
        )
        item = patterns[selected_name]
        label = int(item["label"])
        pat = item["pattern"]

        with st.container(border=True):
            pv_col, meta_col = st.columns([1, 1.2])
            with pv_col:
                st.markdown("**Pixel layout** (5×5, enlarged)")
                st.image(pattern_preview_image(pat, 220), width=220)
            with meta_col:
                st.markdown(f"**Identifier** `{selected_name}`")
                st.markdown(f"**Ground-truth class:** `{label}` — 0 ≈ zero-like, 1 ≈ one-like")
                st.markdown("**Binary grid** (`#` = 1, `.` = 0)")
                st.code(pattern_ascii_layout(pat), language=None)
                if st.session_state.trained_model is not None:
                    vec = np.array(pat, dtype=float).reshape(1, 25)
                    prob = float(st.session_state.trained_model.predict(vec)[0, 0])
                    pred = 1 if prob >= 0.5 else 0
                    match = "match" if pred == label else "mismatch"
                    st.markdown(
                        f"**Model:** predicted class **{pred}** · P(class 1) = **{prob:.4f}** · {match} with label"
                    )
                else:
                    st.info("Train the model to see predicted class for this sample.")

        if not presentation_mode:
            with st.expander("All samples (table)", expanded=False):
                st.dataframe(
                    [{"name": n, "label": int(patterns[n]["label"])} for n in name_list],
                    use_container_width=True,
                    hide_index=True,
                )

    if st.session_state.last_dataset and st.session_state.trained_model is not None:
        x_last, y_last, patterns_last = st.session_state.last_dataset
        probs = st.session_state.trained_model.predict(x_last).flatten()
        preds = (probs >= 0.5).astype(int)
        names = list(patterns_last.keys())
        sample_df = pd.DataFrame(
            {
                "sample": names,
                "true_label": y_last.flatten().astype(int),
                "pred_label": preds,
                "prob_one": probs,
            }
        )
        m = st.session_state.last_metrics
        cm_df = pd.DataFrame(
            [[m["tn"], m["fp"]], [m["fn"], m["tp"]]],
            index=["True 0", "True 1"],
            columns=["Pred 0", "Pred 1"],
        )
        if presentation_mode:
            tab1, tab2 = st.tabs(["Per-Sample Outputs", "Confusion Matrix"])
            with tab1:
                st.dataframe(sample_df, use_container_width=True)
            with tab2:
                st.dataframe(cm_df, use_container_width=True)
        else:
            st.subheader("Per-Sample Output Table")
            st.dataframe(sample_df, use_container_width=True)
            st.subheader("Confusion Matrix")
            st.dataframe(cm_df, use_container_width=True)

with right:
    st.subheader("Pattern Input")
    st.caption("Step 1: Draw. Step 2: Predict.")
    if st.session_state.trained_model is None:
        st.info("Train the model first to unlock drawing and prediction.")
    else:
        if st_canvas is None:
            st.warning("Canvas package missing. Using compact pixel fallback.")
            with st.expander("Fallback 5x5 Pixel Grid", expanded=True):
                for r in range(5):
                    cols = st.columns(5)
                    for c in range(5):
                        key = f"pixel_{r}_{c}"
                        with cols[c]:
                            st.checkbox(f"pixel-{r}-{c}", key=key, label_visibility="collapsed")
                        st.session_state.grid[r][c] = 1 if st.session_state[key] else 0
        else:
            with st.container(border=True):
                control_col1, control_col2 = st.columns([3, 2])
                with control_col1:
                    brush_size = st.slider("Brush Size", min_value=4, max_value=18, value=10, step=1)
                with control_col2:
                    if st.button("Clear Canvas", use_container_width=True):
                        st.session_state.canvas_key += 1
                        st.session_state.grid = [[0 for _ in range(5)] for _ in range(5)]
                        update_checkboxes_from_grid()
                        st.session_state.last_prediction = None

                canvas_result = st_canvas(
                    fill_color="rgba(0, 0, 0, 1)",
                    stroke_width=brush_size,
                    stroke_color="#000000",
                    background_color="#FFFFFF",
                    width=240,
                    height=240,
                    drawing_mode="freedraw",
                    key=f"canvas_{st.session_state.canvas_key}",
                    display_toolbar=False,
                )
                if canvas_result.image_data is not None:
                    st.session_state.grid = canvas_to_grid(canvas_result.image_data)
                    update_checkboxes_from_grid()

            with st.container(border=True):
                # Fresh PNG each run avoids stale Streamlit media URLs after rerun/refresh.
                preview_arr = (np.array(st.session_state.grid, dtype=np.uint8) * 255).astype(np.uint8)
                preview_img = Image.fromarray(preview_arr).resize(
                    (140, 140), resample=Image.Resampling.NEAREST
                )
                buf = io.BytesIO()
                preview_img.save(buf, format="PNG")
                buf.seek(0)
                st.image(buf, caption="Model Input (5x5)", width=140)

        grid_arr = np.array(st.session_state.grid, dtype=float).reshape(1, 25)
        if st.button("Predict", use_container_width=True):
            prob = float(st.session_state.trained_model.predict(grid_arr)[0, 0])
            pred = 1 if prob >= 0.5 else 0
            st.session_state.last_prediction = {"pred": pred, "prob_one": prob}

        if st.session_state.last_prediction is not None:
            pred = st.session_state.last_prediction["pred"]
            prob = st.session_state.last_prediction["prob_one"]
            label = "Class 1 (one-like)" if pred == 1 else "Class 0 (zero-like)"
            confidence = prob if pred == 1 else (1.0 - prob)
            with st.container(border=True):
                st.markdown(f"### Prediction: **{label}**")
                st.caption(f"Confidence: {confidence * 100:.1f}%  |  P(class 1): {prob:.4f}")
                st.progress(int(round(confidence * 100)))

        if not presentation_mode:
            st.subheader("Add Drawn Pattern to Dataset")
            sample_name = st.text_input("Sample name", value="new_sample")
            sample_label = st.selectbox("Sample label", options=[0, 1], index=0)
            if st.button("Add Sample", use_container_width=True):
                patterns = load_patterns(dataset_path=dataset_path)
                patterns[sample_name] = {
                    "label": int(sample_label),
                    "pattern": [row[:] for row in st.session_state.grid],
                }
                save_patterns(patterns, dataset_path=dataset_path)
                st.session_state.last_added_sample_name = sample_name
                st.success(f"Saved sample '{sample_name}' to {dataset_path}")

            if st.button("Undo Last Added Sample", use_container_width=True):
                last_name = st.session_state.last_added_sample_name
                if not last_name:
                    st.info("No recently added sample to undo.")
                else:
                    patterns = load_patterns(dataset_path=dataset_path)
                    if last_name in patterns:
                        del patterns[last_name]
                        save_patterns(patterns, dataset_path=dataset_path)
                        st.session_state.last_added_sample_name = None
                        st.success(f"Undid last added sample '{last_name}'.")
                    else:
                        st.warning("Last added sample is already removed or dataset changed.")

            st.subheader("Manage Dataset")
            current_patterns = load_patterns(dataset_path=dataset_path)
            if current_patterns:
                delete_name = st.selectbox(
                    "Select sample to remove",
                    options=list(current_patterns.keys()),
                    index=0,
                )
                if st.button("Remove Selected Sample", use_container_width=True):
                    patterns = load_patterns(dataset_path=dataset_path)
                    if delete_name in patterns:
                        del patterns[delete_name]
                        save_patterns(patterns, dataset_path=dataset_path)
                        st.success(f"Removed sample '{delete_name}' from {dataset_path}")
                    else:
                        st.warning("Sample was not found. Refresh and try again.")

            if st.button("Reset Dataset to Default Samples", use_container_width=True):
                save_patterns(default_patterns(), dataset_path=dataset_path)
                st.success(f"Dataset reset to default samples in {dataset_path}")
