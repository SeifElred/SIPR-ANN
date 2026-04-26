# 5-10 Minute Presentation Script (Demo-Focused)

## 0:00 - 0:45 | Introduction
"This project is about applying Artificial Neural Networks to simple image pattern recognition.  
I implemented the whole model manually using NumPy, without machine learning libraries."

## 0:45 - 2:00 | Problem and Data
"The problem is binary classification of handwritten-style 5x5 patterns.  
Each pixel is 0 or 1, and each image is flattened into a 25-value vector."

## 2:00 - 3:30 | Network Architecture
"The ANN has:
- 25 input neurons
- 5 hidden neurons
- 1 output neuron
I use sigmoid activation and Mean Squared Error loss."

## 3:30 - 5:30 | Implementation Walkthrough
"The script includes:
- manual weight and bias initialization
- forward propagation
- loss computation
- backpropagation with gradient descent
- iterative training over epochs"

## 5:30 - 7:30 | Live Demo
Run:

```bash
python ann_pattern_recognition.py
```

Explain:
- Loss/accuracy trend
- Prediction probability for each pattern
- Why values near 0 represent class 0 and near 1 represent class 1

## 7:30 - 9:00 | Dataset Change + Interpretation
"Now I modify one pixel in a known pattern and run prediction again."

Explain:
- Whether probability moved toward class 0 or class 1
- What this says about feature sensitivity

## 9:00 - 10:00 | Conclusion
"The ANN successfully learns simple pattern recognition with manual implementation.  
This demonstrates core neural-network learning behavior and how input variation affects output."

---

## Demo Checklist (Before Recording)
- Python + NumPy installed
- Script runs without errors
- One clean terminal demo recorded
- Include dataset-change interpretation segment
- Upload video to YouTube (unlisted is fine unless instructed otherwise)
- Paste link into `docs/short_report_template.md` and submit
