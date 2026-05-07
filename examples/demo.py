import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ternary_zero as tz
import numpy as np


def main():
    print(f"ternary_zero version: {tz.__version__}")
    print()

    x = tz.randn(3, 4, requires_grad=True)
    w = tz.randn(4, 2, requires_grad=True)
    b = tz.zeros(2, requires_grad=True)

    y = x @ w + b
    loss = y.sum()
    loss.backward()
    print(f"Input:  {x.shape}")
    print(f"Weight: {w.shape}")
    print(f"Output: {y.shape}")
    print(f"Loss:   {loss.item():.4f}")
    print()

    model = tz.nn.Sequential(
        tz.nn.Linear(784, 256),
        tz.nn.ReLU(),
        tz.nn.Dropout(0.2),
        tz.nn.Linear(256, 128),
        tz.nn.ReLU(),
        tz.nn.BitLinear(128, 10, alpha=0.5),
    )
    print(model)
    print(f"\nParameters: {model.num_parameters():,}")
    print()

    x = tz.randn(32, 784)
    output = model(x)
    print(f"Batch output shape: {output.shape}")

    model.train()
    x = tz.randn(4, 784, requires_grad=True)
    target = tz.tensor([0, 1, 2, 3], dtype=np.int64)

    logits = model(x)
    loss_fn = tz.nn.CrossEntropyLoss()
    loss = loss_fn(logits, target)
    print(f"Training loss: {loss.item():.4f}")

    model.zero_grad()
    loss.backward()
    print("Backward pass completed successfully")

    optimizer = tz.optim.Adam(model.parameters(), lr=0.001)
    optimizer.step()
    print("Optimizer step completed successfully")

    print("\n--- Ternary Quantization Demo ---")
    w = tz.randn(4, 16)
    ternary, scale = tz.quantize.ternary_quantize(w, alpha=0.5)
    stats = tz.quantize.ternary_weight_analysis(ternary)
    print(f"Weight shape: {w.shape}")
    print(f"Ternary: {np.unique(ternary.data)}")
    print(f"Scale: {scale:.4f}")
    print(f"Sparsity: {stats['sparsity']:.1%}")
    print(f"Compression vs FP32: {stats['compression_ratio_vs_fp32']:.0f}x")
    print(f"Compression vs FP16: {stats['compression_ratio_vs_fp16']:.0f}x")

    packed = tz.quantize.pack_ternary_to_u32(ternary, 16)
    unpacked = tz.quantize.unpack_u32_to_ternary(packed, 16)
    assert np.array_equal(ternary.data.flatten(), unpacked.data), "Pack/unpack roundtrip failed"
    print("Pack/unpack roundtrip: PASS")

    print("\n--- Binary Utilities Demo ---")
    b = tz.utils.pack_binary(tz.tensor([1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0]))
    print(f"Packed binary: {b.data}")

    print("\nAll demos completed successfully!")


if __name__ == "__main__":
    main()
