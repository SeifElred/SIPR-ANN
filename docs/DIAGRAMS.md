# Architecture and Workflow Diagrams (Code)

Use these Mermaid blocks in Markdown viewers that support Mermaid (or in tools like mermaid.live).

## 1) Architecture Diagram Code

```mermaid
flowchart LR
    A[patterns.json<br/>5x5 binary samples + labels] --> B[Data Loader<br/>build_dataset()]
    B --> C[Preprocessing<br/>flatten 5x5 to 25 features]
    C --> D[SimpleANN Model]

    subgraph D[SimpleANN: 25-5-1]
        D1[Input Layer<br/>25 neurons]
        D2[Hidden Layer<br/>5 neurons + sigmoid]
        D3[Output Layer<br/>1 neuron + sigmoid]
        D1 --> D2 --> D3
    end

    D --> E[Training Loop<br/>forward -> MSE -> backprop -> update]
    E --> F[Metrics<br/>Accuracy, Precision, Recall, F1, Confusion Matrix]
    D --> G[Prediction API<br/>predict()]

    H[Streamlit UI] --> I[Canvas Drawing]
    I --> J[Canvas-to-Grid<br/>grayscale -> 5x5 -> binary]
    J --> G
    G --> K[Prediction Panel<br/>class + confidence]

    H --> L[Dataset Management<br/>add/remove/reset]
    L --> A
```

## 2) Workflow Diagram Code (Step-by-Step)

```mermaid
flowchart TD
    S1[Start App] --> S2[Load patterns.json]
    S2 --> S3[Set hyperparameters<br/>epochs, lr, test ratio, seed]
    S3 --> S4[Click Train Model]
    S4 --> S5[Shuffle + Train/Test Split]
    S5 --> S6[Train ANN<br/>forward + backprop over epochs]
    S6 --> S7[Compute Metrics<br/>loss, train/test acc, precision, recall, F1]
    S7 --> S8[Show confusion matrix + sample outputs]
    S8 --> S9[Unlock Pattern Input]
    S9 --> S10[Draw on canvas]
    S10 --> S11[Convert drawing to 5x5 binary grid]
    S11 --> S12[Predict class + confidence]
    S12 --> S13{Need dataset update?}
    S13 -- Yes --> S14[Add/Remove/Undo/Reset sample]
    S14 --> S4
    S13 -- No --> S15[End / Present results]
```

## 3) Optional Sequence Diagram Code

```mermaid
sequenceDiagram
    participant User
    participant UI as Streamlit UI
    participant Data as patterns.json
    participant Model as SimpleANN

    User->>UI: Set epochs/lr/test split/seed
    User->>UI: Train Model
    UI->>Data: Load samples
    UI->>Model: Train on train split
    Model-->>UI: History + trained weights
    UI-->>User: Metrics + confusion matrix

    User->>UI: Draw pattern + Predict
    UI->>UI: Convert canvas to 5x5 binary
    UI->>Model: predict(x)
    Model-->>UI: prob_one
    UI-->>User: Class + confidence
```
