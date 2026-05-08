"""
Example 1 — Fundamental: MNIST-Style Ternary Classifier
========================================================

Use case:
    Train a lightweight image classifier on synthetic 28×28 pixel data (simulating
    MNIST) using a mix of standard FP32 Linear layers and Ternary-Zero BitLinear
    layers.  The example demonstrates the full lifecycle: model construction,
    forward/backward training, quantization-aware inference, and accuracy evaluation.

Key concepts demonstrated:
    - tz.nn.Module / tz.nn.Sequential    → building models
    - tz.nn.BitLinear                    → 2-bit quantized linear layer (STE-trained)
    - tz.nn.Linear                       → standard FP32 linear layer
    - tz.optim.Adam                      → optimizer with gradient descent
    - tz.nn.CrossEntropyLoss             → classification loss
    - tz.quantize.ternary_weight_analysis → inspecting quantization statistics
    - tz.no_grad() / model.eval()        → inference mode

Expected output:
    Per-epoch loss values that decrease over time, followed by quantization
    statistics for the BitLinear layers and a final accuracy estimate.
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ternary_zero as tz
import numpy as np


# ──────────────────────────────────────────────────────────────────────
# 1. Synthetic Dataset (simulates flattened 28×28 grayscale images)
# ──────────────────────────────────────────────────────────────────────

def make_synthetic_dataset(
    num_samples: int = 500,
    input_dim: int = 784,
    num_classes: int = 10,
    seed: int = 42,
):
    """Generate a reproducible synthetic classification dataset.

    Each class is centred on a different region of the input space so that
    a simple model can learn to separate them, mimicking digit classification.
    """
    rng = np.random.RandomState(seed)

    samples_per_class = num_samples // num_classes
    X_parts, y_parts = [], []

    for cls in range(num_classes):
        centre = rng.randn(input_dim).astype(np.float32) * 2.0
        noise = rng.randn(samples_per_class, input_dim).astype(np.float32) * 0.5
        X_parts.append(centre + noise)
        y_parts.append(np.full(samples_per_class, cls, dtype=np.int64))

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    # Shuffle
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


# ──────────────────────────────────────────────────────────────────────
# 2. Model Definition
# ──────────────────────────────────────────────────────────────────────

def build_classifier(input_dim: int = 784, hidden_dim: int = 256, num_classes: int = 10):
    """Construct a 3-layer MLP with BitLinear in the hidden and output layers.

    Architecture:
        Linear(784 → 256)   — FP32 first layer for full-precision feature extraction
        ReLU
        Dropout(0.2)
        BitLinear(256 → 128) — ternary-quantized hidden layer (8× compression)
        ReLU
        BitLinear(128 → 10)  — ternary-quantized output layer

    The first layer remains FP32 to preserve input fidelity.  Hidden and
    output layers use BitLinear to demonstrate the 2-bit weight compression
    that Ternary-Zero provides with the STE training trick.
    """
    return tz.nn.Sequential(
        tz.nn.Linear(input_dim, hidden_dim),
        tz.nn.ReLU(),
        tz.nn.Dropout(0.2),
        tz.nn.BitLinear(hidden_dim, hidden_dim // 2, alpha=0.5),
        tz.nn.ReLU(),
        tz.nn.BitLinear(hidden_dim // 2, num_classes, alpha=0.5),
    )


# ──────────────────────────────────────────────────────────────────────
# 3. Training Loop
# ──────────────────────────────────────────────────────────────────────

def train(model, X_train, y_train, epochs: int = 15, batch_size: int = 64, lr: float = 1e-3):
    """Mini-batch training with Adam optimiser and cross-entropy loss.

    Returns a list of per-epoch average losses for monitoring convergence.
    """
    optimizer = tz.optim.Adam(model.parameters(), lr=lr)
    loss_fn = tz.nn.CrossEntropyLoss()

    n = len(y_train)
    epoch_losses = []

    for epoch in range(epochs):
        # Simple sequential mini-batching (shuffle each epoch)
        perm = np.random.permutation(n)
        X_shuf = X_train[perm]
        y_shuf = y_train[perm]

        running_loss = 0.0
        num_batches = 0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = tz.tensor(X_shuf[start:end].tolist(), requires_grad=False)
            yb = tz.tensor(y_shuf[start:end].tolist(), dtype=np.int64, requires_grad=False)

            # Forward
            logits = model(xb)
            loss = loss_fn(logits, yb)

            # Backward + update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        avg_loss = running_loss / num_batches
        epoch_losses.append(avg_loss)
        print(f"  Epoch {epoch + 1:2d}/{epochs}  loss = {avg_loss:.4f}")

    return epoch_losses


# ──────────────────────────────────────────────────────────────────────
# 4. Evaluation
# ──────────────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, batch_size: int = 128) -> float:
    """Compute classification accuracy in inference mode (no gradients)."""
    model.eval()
    correct = 0
    total = len(y_test)

    with tz.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            xb = tz.tensor(X_test[start:end].tolist())

            logits = model(xb)
            preds = logits.data.argmax(axis=-1)
            correct += int(np.sum(preds == y_test[start:end]))

    accuracy = correct / total
    return accuracy


# ──────────────────────────────────────────────────────────────────────
# 5. Quantization Inspection
# ──────────────────────────────────────────────────────────────────────

def inspect_quantization(model):
    """Walk through model layers and report ternary statistics for each BitLinear."""
    print("\n── Quantization Analysis ─────────────────────────────────")

    for idx, layer in enumerate(model):
        if isinstance(layer, tz.nn.BitLinear):
            # Trigger quantization from current trained weights
            layer.quantize_weights()

            stats = tz.quantize.ternary_weight_analysis(
                tz.tensor(layer._ternary_weight.flatten().tolist())
            )
            print(f"  Layer {idx} (BitLinear {layer.in_features}→{layer.out_features}):")
            print(f"    Weights   : {stats['total']:,}")
            print(f"    Zeros     : {stats['zeros']:,}  ({stats['sparsity']:.1%} sparsity)")
            print(f"    +1 / -1   : {stats['positives']:,} / {stats['negatives']:,}")
            print(f"    Compression: {stats['compression_ratio_vs_fp16']:.0f}× vs FP16")
            print(f"                {stats['compression_ratio_vs_fp32']:.0f}× vs FP32")

            # Demonstrate pack → unpack roundtrip
            flat_ternary = tz.tensor(layer._ternary_weight.flatten().tolist())
            n = layer.in_features
            packed = tz.quantize.pack_ternary_to_u32(flat_ternary, n)
            unpacked = tz.quantize.unpack_u32_to_ternary(packed, n)
            match = np.array_equal(flat_ternary.data.flatten(), unpacked.data.flatten())
            print(f"    Pack/unpack integrity: {'PASS' if match else 'FAIL'}")


# ──────────────────────────────────────────────────────────────────────
# 6. Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("═" * 60)
    print(" Ternary-Zero  ·  Example 1: MNIST-Style Ternary Classifier")
    print("═" * 60)
    print(f" Library version: {tz.__version__}\n")

    # Dataset
    print("── Dataset ───────────────────────────────────────────────")
    X, y = make_synthetic_dataset(num_samples=500, input_dim=784, num_classes=10)
    split = 400
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    print(f"  Train: {len(y_train)} samples   Test: {len(y_test)} samples")
    print(f"  Input dim: {X_train.shape[1]}   Classes: {len(np.unique(y))}\n")

    # Model
    model = build_classifier()
    print("── Model Architecture ────────────────────────────────────")
    print(model)
    print(f"  Total parameters: {model.num_parameters():,}\n")

    # Training
    print("── Training ─────────────────────────────────────────────")
    losses = train(model, X_train, y_train, epochs=15, batch_size=64, lr=1e-3)
    print(f"  Final loss: {losses[-1]:.4f}  (started at {losses[0]:.4f})\n")

    # Quantization inspection
    inspect_quantization(model)

    # Evaluation
    print("\n── Evaluation ───────────────────────────────────────────")
    accuracy = evaluate(model, X_test, y_test)
    print(f"  Test accuracy: {accuracy:.1%}")

    # Inference demo
    print("\n── Single-sample Inference ──────────────────────────────")
    with tz.no_grad():
        model.eval()
        sample = tz.tensor(X_test[0].tolist()).unsqueeze(0)  # shape [1, 784]
        logits = model(sample)
        probs = logits.softmax(dim=-1)
        pred = int(probs.data.argmax())
        print(f"  Predicted class: {pred}")
        print(f"  Confidence     : {probs.data[0, pred]:.1%}")
        print(f"  True class     : {y_test[0]}")

    print("\n" + "═" * 60)
    print(" Done.")
    print("═" * 60)


if __name__ == "__main__":
    main()
