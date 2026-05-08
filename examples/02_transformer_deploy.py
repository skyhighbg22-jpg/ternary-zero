"""
Example 2 — Advanced: Transformer-Style Text Classifier with Deployment Pipeline
=================================================================================

Use case:
    Build a small transformer-inspired encoder for text classification, train it,
    then demonstrate a complete deployment pipeline:

      1. Custom Module subclassing (multi-head attention + feed-forward block)
      2. Mixed-precision model  (FP32 Linear in attention, BitLinear in FFN)
      3. Learning-rate scheduling (step-decay)
      4. Full train → save-state-dict → load-state-dict → eval cycle
      5. Post-training ternary quantization & packing for deployment
      6. Binary embedding utilities for similarity search

Key concepts demonstrated:
    - tz.nn.Module subclassing             → custom layers
    - tz.nn.LayerNorm / tz.nn.BitLinear    → mixed-precision architecture
    - model.state_dict() / load_state_dict → checkpoint persistence
    - tz.quantize.ternary_quantize         → post-training weight quantization
    - tz.quantize.pack_ternary_to_u32      → GPU-ready 2-bit packing
    - tz.quantize.ternary_weight_analysis  → compression statistics
    - tz.utils.pack_binary / hamming_distance → binary embedding search

Architecture (simplified transformer encoder block):
    Input tokens  → Embedding lookup (FP32)
                  → [Attention (FP32 Linear Q/K/V) + residual + LayerNorm]
                  → [FFN: BitLinear → ReLU → BitLinear + residual + LayerNorm]
                  → Mean-pool → Classification head (BitLinear)
"""

import sys, os, json, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ternary_zero as tz
import numpy as np


# ══════════════════════════════════════════════════════════════════════
# 1.  CUSTOM MODULES
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(tz.nn.Module):
    """Simplified single-head attention using FP32 Linear layers.

    In a production setting you would split into multiple heads; here we
    keep it to a single head for clarity and to showcase custom Module
    subclassing with the Ternary-Zero API.

    Key methods used:
        tz.nn.Linear       — full-precision Q/K/V projections
        .softmax(dim=-1)   — attention weight normalisation
        .matmul()          — weighted aggregation of values
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        # Q, K, V projections stay FP32 to preserve attention precision
        self.q_proj = tz.nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = tz.nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = tz.nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = tz.nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: tz.Tensor) -> tz.Tensor:
        # x shape: [batch, seq_len, embed_dim]
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Scaled dot-product attention (omitting head split for simplicity)
        scale = float(np.sqrt(self.embed_dim))
        attn_scores = (Q @ K.transpose(-2, -1)) / scale   # [B, S, S]
        attn_weights = attn_scores.softmax(dim=-1)         # [B, S, S]
        out = attn_weights @ V                              # [B, S, D]
        return self.out_proj(out)


class FeedForward(tz.nn.Module):
    """Position-wise feed-forward network using BitLinear for compression.

    The two linear transforms inside the FFN are the largest parameters in
    a transformer.  Replacing them with BitLinear gives 8× compression on
    these layers with minimal accuracy impact thanks to STE training.

    Key methods used:
        tz.nn.BitLinear(alpha=0.5) — ternary-quantized linear
        .relu()                    — element-wise activation
    """

    def __init__(self, embed_dim: int, ff_dim: int, alpha: float = 0.5):
        super().__init__()
        self.fc1 = tz.nn.BitLinear(embed_dim, ff_dim, alpha=alpha)
        self.fc2 = tz.nn.BitLinear(ff_dim, embed_dim, alpha=alpha)

    def forward(self, x: tz.Tensor) -> tz.Tensor:
        return self.fc2(self.fc1(x).relu())


class TransformerBlock(tz.nn.Module):
    """Single transformer encoder block: Attention → Add&Norm → FFN → Add&Norm.

    Demonstrates mixing FP32 (attention) and ternary (FFN) layers in one block,
    plus residual connections via the ``+`` operator on Tensors.
    """

    def __init__(self, embed_dim: int, ff_dim: int, alpha: float = 0.5):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim)
        self.norm1 = tz.nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ff_dim, alpha=alpha)
        self.norm2 = tz.nn.LayerNorm(embed_dim)

    def forward(self, x: tz.Tensor) -> tz.Tensor:
        # Attention sub-layer with pre-norm residual
        h = self.norm1(x)
        x = x + self.attn(h)
        # FFN sub-layer with pre-norm residual
        h = self.norm2(x)
        x = x + self.ffn(h)
        return x


class TernaryTextClassifier(tz.nn.Module):
    """End-to-end classifier: token embedding → N × TransformerBlock → pool → head.

    Architecture:
        TokenEmbedding(vocab_size, embed_dim)       — lookup table (FP32)
        TransformerBlock × num_layers                — encoder stack
        MeanPool                                        — aggregate over seq_len
        BitLinear(embed_dim → num_classes)            — classification head
    """

    def __init__(
        self,
        vocab_size: int = 256,
        embed_dim: int = 64,
        ff_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 4,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        # Embedding stored as a plain Parameter (no dedicated Embedding module)
        self.embedding = tz.nn.Parameter(
            np.random.randn(vocab_size, embed_dim).astype(np.float32) * 0.02
        )
        self.layers = tz.nn.Sequential(
            *[TransformerBlock(embed_dim, ff_dim, alpha=alpha) for _ in range(num_layers)]
        )
        self.classifier = tz.nn.BitLinear(embed_dim, num_classes, alpha=alpha)

    def forward(self, token_ids: tz.Tensor) -> tz.Tensor:
        # token_ids: [batch, seq_len] of integer indices
        # Embedding lookup (manual gather)
        ids = token_ids.data.astype(np.int64)
        x_data = self.embedding.data[ids]          # [B, S, D]
        x = tz.tensor(x_data.tolist(), requires_grad=True)

        # Transformer encoder
        x = self.layers(x)

        # Mean-pool over sequence dimension → [B, D]
        x = tz.tensor(x.data.mean(axis=1).tolist(), requires_grad=True)

        # Classification head → [B, num_classes]
        return self.classifier(x)


# ══════════════════════════════════════════════════════════════════════
# 2.  SYNTHETIC TEXT DATASET
# ══════════════════════════════════════════════════════════════════════

def make_text_dataset(
    num_samples: int = 200,
    seq_len: int = 16,
    vocab_size: int = 256,
    num_classes: int = 4,
    seed: int = 42,
):
    """Generate synthetic "text" data as random byte sequences with class-dependent
    token distributions (so the model has a learnable signal).
    """
    rng = np.random.RandomState(seed)
    X_parts, y_parts = [], []

    samples_per_class = num_samples // num_classes
    for cls in range(num_classes):
        # Each class biases certain token ranges
        base = cls * (vocab_size // num_classes)
        tokens = (rng.randint(0, vocab_size // num_classes, (samples_per_class, seq_len)) + base) % vocab_size
        X_parts.append(tokens.astype(np.int64))
        y_parts.append(np.full(samples_per_class, cls, dtype=np.int64))

    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


# ══════════════════════════════════════════════════════════════════════
# 3.  TRAINING WITH LEARNING-RATE SCHEDULING
# ══════════════════════════════════════════════════════════════════════

class StepDecayScheduler:
    """Simple step-decay LR scheduler.

    Every `decay_every` epochs, multiply the LR by `decay_factor`.
    Demonstrates how to hook into the optimiser's param_groups manually.
    """

    def __init__(self, optimizer, decay_every: int = 5, decay_factor: float = 0.5):
        self.optimizer = optimizer
        self.decay_every = decay_every
        self.decay_factor = decay_factor
        self.base_lr = optimizer.param_groups[0]["lr"]

    def step(self, epoch: int):
        if epoch > 0 and epoch % self.decay_every == 0:
            new_lr = self.base_lr * (self.decay_factor ** (epoch // self.decay_every))
            for group in self.optimizer.param_groups:
                group["lr"] = new_lr
            return new_lr
        return self.optimizer.param_groups[0]["lr"]


def train_model(model, X_train, y_train, epochs: int = 20, batch_size: int = 32, lr: float = 3e-3):
    """Train with Adam + step-decay scheduler."""
    optimizer = tz.optim.Adam(model.parameters(), lr=lr)
    scheduler = StepDecayScheduler(optimizer, decay_every=8, decay_factor=0.5)
    loss_fn = tz.nn.CrossEntropyLoss()

    n = len(y_train)
    history = []

    for epoch in range(epochs):
        current_lr = scheduler.step(epoch)
        perm = np.random.permutation(n)
        X_shuf, y_shuf = X_train[perm], y_train[perm]

        running_loss = 0.0
        batches = 0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = tz.tensor(X_shuf[start:end].tolist(), requires_grad=False)
            yb = tz.tensor(y_shuf[start:end].tolist(), dtype=np.int64, requires_grad=False)

            logits = model(xb)
            loss = loss_fn(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batches += 1

        avg = running_loss / batches
        history.append(avg)
        lr_str = f"  lr={current_lr:.6f}" if current_lr != lr else ""
        print(f"  Epoch {epoch + 1:2d}/{epochs}  loss={avg:.4f}{lr_str}")

    return history


# ══════════════════════════════════════════════════════════════════════
# 4.  CHECKPOINTING (state_dict save / load)
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(model, path: str):
    """Serialize model state_dict to a JSON-compatible file.

    Each parameter is stored as a nested list (flattened shape + data).
    """
    state = model.state_dict()
    serializable = {}
    for name, arr in state.items():
        serializable[name] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "data": arr.flatten().tolist(),
        }
    with open(path, "w") as f:
        json.dump(serializable, f)
    size_kb = os.path.getsize(path) / 1024
    print(f"  Checkpoint saved → {path}  ({size_kb:.1f} KB, {len(state)} params)")


def load_checkpoint(model, path: str):
    """Restore model state_dict from a JSON checkpoint."""
    with open(path, "r") as f:
        raw = json.load(f)
    restored = {}
    for name, blob in raw.items():
        arr = np.array(blob["data"], dtype=np.float32).reshape(blob["shape"])
        restored[name] = arr
    model.load_state_dict(restored)
    print(f"  Checkpoint loaded ← {path}  ({len(restored)} params)")


# ══════════════════════════════════════════════════════════════════════
# 5.  POST-TRAINING QUANTIZATION & DEPLOYMENT PACKING
# ══════════════════════════════════════════════════════════════════════

def quantize_and_pack(model, alpha: float = 0.5):
    """Walk every BitLinear layer, quantize its weights to ternary,
    pack them into uint32 format, and report compression statistics.

    This is the deployment path: after training, you quantize once,
    pack, and ship the packed weights for GPU GEMV kernels.
    """
    print("\n── Post-Training Quantization & Packing ─────────────────")

    for name, module in model.named_modules():
        if not isinstance(module, tz.nn.BitLinear):
            continue

        full_name = name or "root"
        w = module.weight
        flat_w = tz.tensor(w.data.flatten().tolist())

        # Ternary quantize
        ternary, scale = tz.quantize.ternary_quantize(flat_w, alpha=alpha)
        stats = tz.quantize.ternary_weight_analysis(ternary)

        # Pack into uint32 for GPU kernels
        n = module.in_features
        packed = tz.quantize.pack_ternary_to_u32(ternary, n)

        # Verify roundtrip
        unpacked = tz.quantize.unpack_u32_to_ternary(packed, n)
        integrity = np.array_equal(ternary.data.flatten(), unpacked.data.flatten())

        # Compute sizes
        fp32_bytes = w.numel() * 4
        packed_bytes = packed.numel() * 4
        ratio = fp32_bytes / packed_bytes if packed_bytes > 0 else float("inf")

        print(f"  [{full_name}] BitLinear({module.in_features}→{module.out_features}):")
        print(f"    Sparsity     : {stats['sparsity']:.1%}")
        print(f"    FP32 size    : {fp32_bytes:,} bytes")
        print(f"    Packed size  : {packed_bytes:,} bytes")
        print(f"    Compression  : {ratio:.1f}×")
        print(f"    Integrity    : {'PASS' if integrity else 'FAIL'}")
        print(f"    Scale factor : {scale:.6f}")


# ══════════════════════════════════════════════════════════════════════
# 6.  BINARY EMBEDDING UTILITIES
# ══════════════════════════════════════════════════════════════════════

def demo_binary_embeddings(model, X_data, y_data):
    """Use the trained model to extract embeddings, binarize them,
    and perform fast similarity search via Hamming distance.

    This demonstrates how ternary/binary representations can be used
    for efficient nearest-neighbour search on the model's learned features.
    """
    print("\n── Binary Embedding Similarity Search ───────────────────")

    model.eval()
    with tz.no_grad():
        # Get embeddings (mean-pooled hidden states) for a small subset
        sample_ids = tz.tensor(X_data[:8].tolist())
        ids = sample_ids.data.astype(np.int64)
        x_data = model.embedding.data[ids]
        x = tz.tensor(x_data.tolist())
        h = model.layers(x)
        embeddings = h.data.mean(axis=1)    # [8, embed_dim]

    # Binarize: sign(embedding) → {-1, +1}
    binary_vecs = np.sign(embeddings).astype(np.float32)
    binary_vecs[binary_vecs == 0] = 1.0  # handle exact zeros

    # Pack each binary vector
    packed_vecs = []
    for i in range(len(binary_vecs)):
        b = tz.tensor(binary_vecs[i].tolist())
        packed_vecs.append(tz.utils.pack_binary(b))

    # Hamming distance matrix
    n_vecs = len(packed_vecs)
    print(f"  Computing {n_vecs}×{n_vecs} Hamming distance matrix ...")
    print(f"  {'Query':>6}  {'Target':>6}  {'Hamming':>8}  {'Class':>5}")
    print(f"  {'─' * 6}  {'─' * 6}  {'─' * 8}  {'─' * 5}")

    for qi in range(min(4, n_vecs)):
        distances = []
        for ti in range(n_vecs):
            d = tz.utils.hamming_distance(packed_vecs[qi], packed_vecs[ti])
            distances.append(d)
        d_arr = np.array(distances, dtype=np.float64)
        d_arr[qi] = np.inf  # exclude self
        nearest = int(np.argmin(d_arr))
        print(
            f"  q={qi} (cls={y_data[qi]})  "
            f"→ nearest={nearest} (cls={y_data[nearest]})  "
            f"dist={distances[nearest]}"
        )

    print("  (Binary embeddings enable O(1) per-distance XOR+popcount operations)")


# ══════════════════════════════════════════════════════════════════════
# 7.  EVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate_model(model, X_test, y_test, batch_size: int = 64) -> float:
    """Compute classification accuracy in eval mode."""
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

    return correct / total


# ══════════════════════════════════════════════════════════════════════
# 8.  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print("═" * 65)
    print(" Ternary-Zero  ·  Example 2: Transformer Text Classifier")
    print("═" * 65)
    print(f" Library version: {tz.__version__}\n")

    # ── Dataset ──────────────────────────────────────────────────────
    print("── Dataset ───────────────────────────────────────────────")
    X, y = make_text_dataset(num_samples=200, seq_len=16, vocab_size=256, num_classes=4)
    split = 160
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    print(f"  Train: {len(y_train)}   Test: {len(y_test)}")
    print(f"  Seq len: {X.shape[1]}   Vocab: 256   Classes: 4\n")

    # ── Model ────────────────────────────────────────────────────────
    model = TernaryTextClassifier(
        vocab_size=256, embed_dim=64, ff_dim=128, num_layers=2, num_classes=4, alpha=0.5,
    )
    print("── Model ─────────────────────────────────────────────────")
    print(model)
    total_params = model.num_parameters()
    trainable = model.num_parameters(only_trainable=True)
    print(f"  Parameters: {total_params:,}  (trainable: {trainable:,})\n")

    # ── Training ─────────────────────────────────────────────────────
    print("── Training ─────────────────────────────────────────────")
    t0 = time.time()
    history = train_model(model, X_train, y_train, epochs=20, batch_size=32, lr=3e-3)
    train_time = time.time() - t0
    print(f"  Completed in {train_time:.1f}s   Final loss: {history[-1]:.4f}\n")

    # ── Save checkpoint ──────────────────────────────────────────────
    print("── Checkpointing ────────────────────────────────────────")
    ckpt_path = os.path.join(os.path.dirname(__file__), "checkpoint.json")
    save_checkpoint(model, ckpt_path)

    # Create a fresh model and restore weights (verifies state_dict roundtrip)
    model_fresh = TernaryTextClassifier(
        vocab_size=256, embed_dim=64, ff_dim=128, num_layers=2, num_classes=4, alpha=0.5,
    )
    load_checkpoint(model_fresh, ckpt_path)

    # Verify restored model produces same accuracy
    acc_original = evaluate_model(model, X_test, y_test)
    acc_restored = evaluate_model(model_fresh, X_test, y_test)
    print(f"  Original accuracy : {acc_original:.1%}")
    print(f"  Restored accuracy : {acc_restored:.1%}")
    print(f"  State dict roundtrip: {'PASS' if abs(acc_original - acc_restored) < 1e-6 else 'FAIL'}")

    # Clean up checkpoint file
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    # ── Post-training quantization & packing ─────────────────────────
    quantize_and_pack(model, alpha=0.5)

    # ── Binary embedding search ──────────────────────────────────────
    demo_binary_embeddings(model, X_test, y_test)

    # ── Final evaluation ─────────────────────────────────────────────
    print("\n── Final Evaluation ─────────────────────────────────────")
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"  Test accuracy: {accuracy:.1%}")

    # Per-class breakdown
    model.eval()
    class_correct = {c: 0 for c in range(4)}
    class_total = {c: 0 for c in range(4)}
    with tz.no_grad():
        for start in range(0, len(y_test), 64):
            end = min(start + 64, len(y_test))
            xb = tz.tensor(X_test[start:end].tolist())
            preds = model(xb).data.argmax(axis=-1)
            for i, true_cls in enumerate(y_test[start:end]):
                class_total[true_cls] += 1
                if preds[i] == true_cls:
                    class_correct[true_cls] += 1
    print("  Per-class accuracy:")
    for c in range(4):
        pct = class_correct[c] / class_total[c] if class_total[c] > 0 else 0
        print(f"    Class {c}: {pct:.1%}  ({class_correct[c]}/{class_total[c]})")

    print("\n" + "═" * 65)
    print(" Done.  All pipeline stages completed successfully.")
    print("═" * 65)


if __name__ == "__main__":
    main()
