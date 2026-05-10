"""
Demo: Quantizing a Transformer-Style MLP - FP32 vs Ternary
============================================================

Builds a small 2-layer MLP block (64->128->64) twice - once with standard
Linear layers and once with BitLinear (ternary-quantized) layers - then
compares memory, latency, and output accuracy.

Run with:
    py examples/demo_quantize_transformer.py
"""

import sys, os, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import ternary_zero as tz

if tz.has_torch():
    import torch


# ------------------------------------------------------------------
# 1. Model Builders
# ------------------------------------------------------------------

def build_fp32_mlp(dim_in=64, dim_hidden=128, dim_out=64):
    """Standard FP32 2-layer MLP: Linear -> ReLU -> Linear."""
    return tz.nn.Sequential(
        tz.nn.Linear(dim_in, dim_hidden),
        tz.nn.ReLU(),
        tz.nn.Linear(dim_hidden, dim_out),
    )


def build_ternary_mlp(dim_in=64, dim_hidden=128, dim_out=64, alpha=0.5):
    """Ternary-quantized 2-layer MLP: BitLinear -> ReLU -> BitLinear."""
    return tz.nn.Sequential(
        tz.nn.BitLinear(dim_in, dim_hidden, alpha=alpha),
        tz.nn.ReLU(),
        tz.nn.BitLinear(dim_hidden, dim_out, alpha=alpha),
    )


# ------------------------------------------------------------------
# 2. Weight Copying & Quantization
# ------------------------------------------------------------------

def copy_and_quantize(fp32_model, ternary_model):
    """Copy FP32 weights into BitLinear layers, quantize, and patch for eval.

    After quantize_weights(), the internal _scale and _ternary_weight
    buffers need to be coerced to types the eval forward path expects
    (torch tensors when torch is available).
    """
    fp32_layers = [m for m in fp32_model if isinstance(m, (tz.nn.Linear, tz.nn.BitLinear))]
    ternary_layers = [m for m in ternary_model if isinstance(m, tz.nn.BitLinear)]

    assert len(fp32_layers) == len(ternary_layers), \
        f"Layer count mismatch: {len(fp32_layers)} FP32 vs {len(ternary_layers)} ternary"

    with tz.no_grad():
        for fp32_layer, ternary_layer in zip(fp32_layers, ternary_layers):
            ternary_layer.weight.data.copy_(fp32_layer.weight.data)
            if fp32_layer.bias is not None and ternary_layer.bias is not None:
                ternary_layer.bias.data.copy_(fp32_layer.bias.data)
            ternary_layer.quantize_weights()
            _patch_bitlinear_for_eval(ternary_layer)


def _patch_bitlinear_for_eval(module):
    """Ensure _ternary_weight and _scale are in formats compatible with eval forward.

    quantize_weights() stores _scale as np.float32 scalar and _ternary_weight
    as np.ndarray. The eval forward path checks isinstance(sc, np.ndarray) which
    fails for numpy scalars. This patch converts them to torch tensors when
    torch is active.
    """
    if tz.has_torch():
        device = module.weight.data.device
        tw = module._ternary_weight
        sc = module._scale
        if isinstance(tw, np.ndarray):
            module._ternary_weight = torch.from_numpy(tw.copy()).to(device)
        if isinstance(sc, (np.floating, float, np.float32)):
            module._scale = torch.tensor(float(sc), device=device)


# ------------------------------------------------------------------
# 3. Memory & Parameter Counting
# ------------------------------------------------------------------

def count_parameters(model):
    return model.num_parameters()


def _get_ternary_weight(module):
    """Read the ternary weight array, handling both numpy and torch storage."""
    tw = module._ternary_weight
    if tz.has_torch() and isinstance(tw, torch.Tensor):
        return tw.detach().cpu().numpy()
    return tw


def _get_scale(module):
    """Read the scale factor as a Python float."""
    sc = module._scale
    if tz.has_torch() and isinstance(sc, torch.Tensor):
        return float(sc.item())
    if isinstance(sc, np.ndarray):
        return float(sc.flat[0])
    return float(sc)


def estimate_fp32_bytes(model):
    """FP32 model: 4 bytes per parameter."""
    return count_parameters(model) * 4


def estimate_ternary_inference_bytes(model):
    """Ternary inference footprint: 1 byte per ternary weight + 4 bytes scale + 4 bytes per bias param."""
    total = 0
    for module in model.modules():
        if isinstance(module, tz.nn.BitLinear):
            total += module.in_features * module.out_features  # int8 = 1 byte each
            total += 4  # scale float32
            if module.bias is not None:
                total += module.out_features * 4  # bias float32
    return total


# ------------------------------------------------------------------
# 4. Benchmarking Helpers
# ------------------------------------------------------------------

def benchmark_forward(model, x, warmup=10, runs=100):
    """Measure average forward-pass latency in milliseconds."""
    model.eval()
    with tz.no_grad():
        for _ in range(warmup):
            model(x)

        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            t1 = time.perf_counter()
            times.append(t1 - t0)

    return np.array(times) * 1000.0


def compute_output_metrics(fp32_out, ternary_out):
    """Compute MSE, max absolute error, and cosine similarity."""
    a = to_flat_f64(fp32_out.data)
    b = to_flat_f64(ternary_out.data)

    mse = float(np.mean((a - b) ** 2))
    max_abs_err = float(np.max(np.abs(a - b)))

    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    cosine_sim = float(dot / (norm_a * norm_b)) if norm_a > 0 and norm_b > 0 else 0.0

    return mse, max_abs_err, cosine_sim


def to_flat_f64(data):
    """Convert tensor data to a flat float64 numpy array."""
    if tz.has_torch() and isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy().flatten().astype(np.float64)
    return np.asarray(data).flatten().astype(np.float64)


# ------------------------------------------------------------------
# 5. Ternary Weight Inspection
# ------------------------------------------------------------------

def inspect_ternary_weights(model):
    layer_idx = 0
    for module in model.modules():
        if isinstance(module, tz.nn.BitLinear):
            tw = _get_ternary_weight(module)
            sc = _get_scale(module)
            flat = tw.flatten()
            total = len(flat)
            zeros = int(np.sum(flat == 0))
            pos = int(np.sum(flat == 1))
            neg = int(np.sum(flat == -1))
            sparsity = zeros / total if total > 0 else 0.0

            print(f"  BitLinear {layer_idx} ({module.in_features}->{module.out_features}):")
            print(f"    Scale      : {sc:.6f}")
            print(f"    Weights    : {total:,}")
            print(f"    Zeros      : {zeros:,}  ({sparsity:.1%} sparsity)")
            print(f"    +1 / -1    : {pos:,} / {neg:,}")
            layer_idx += 1


# ------------------------------------------------------------------
# 6. Summary Table Printer
# ------------------------------------------------------------------

def print_table(headers, rows, col_widths=None):
    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            max_w = len(h)
            for row in rows:
                if i < len(row):
                    max_w = max(max_w, len(str(row[i])))
            col_widths.append(max_w + 2)

    def fmt_row(cells):
        parts = []
        for i, cell in enumerate(cells):
            w = col_widths[i] if i < len(col_widths) else 10
            parts.append(str(cell).ljust(w))
        return "| " + " | ".join(parts) + " |"

    sep_h = "+".join("-" * w for w in col_widths)

    print("+" + sep_h + "+")
    print(fmt_row(headers))
    print("+" + sep_h + "+")
    for row in rows:
        print(fmt_row(row))
    print("+" + sep_h + "+")


# ------------------------------------------------------------------
# 7. Main
# ------------------------------------------------------------------

def main():
    DIM_IN = 64
    DIM_HID = 128
    DIM_OUT = 64
    BATCH = 32
    WARMUP = 10
    RUNS = 100
    ALPHA = 0.5

    print("=" * 64)
    print(" Ternary-Zero  |  Transformer Layer Quantization Demo")
    print("=" * 64)
    print(f" Library version : {tz.__version__}")
    print(f" Architecture    : MLP {DIM_IN}->{DIM_HID}->{DIM_OUT}")
    print(f" Batch size      : {BATCH}")
    print(f" Timing          : {WARMUP} warmup + {RUNS} runs")
    print(f" Alpha (threshold): {ALPHA}")
    print()

    fp32_model = build_fp32_mlp(DIM_IN, DIM_HID, DIM_OUT)
    ternary_model = build_ternary_mlp(DIM_IN, DIM_HID, DIM_OUT, alpha=ALPHA)

    copy_and_quantize(fp32_model, ternary_model)

    fp32_model.eval()
    ternary_model.eval()

    fp32_params = count_parameters(fp32_model)
    ternary_params = count_parameters(ternary_model)
    fp32_bytes = estimate_fp32_bytes(fp32_model)
    ternary_inf_bytes = estimate_ternary_inference_bytes(ternary_model)

    print("-- Model Architecture -----------------------------------")
    print(f"  FP32 model:\n{fp32_model}")
    print(f"\n  Ternary model:\n{ternary_model}")
    print()

    print("-- Memory Comparison ------------------------------------")
    savings_pct = (1 - ternary_inf_bytes / fp32_bytes) * 100
    compression = fp32_bytes / ternary_inf_bytes if ternary_inf_bytes > 0 else float("inf")
    print_table(
        ["Metric", "FP32 Model", "Ternary Model", "Savings"],
        [
            ["Parameters", f"{fp32_params:,}", f"{ternary_params:,}", "-"],
            ["FP32 storage (bytes)", f"{fp32_bytes:,}", "-", "-"],
            ["Inference storage (bytes)", "-", f"{ternary_inf_bytes:,}", f"{savings_pct:.1f}%"],
            ["Compression ratio", "-", "-", f"{compression:.1f}x"],
        ],
    )
    print()

    print("-- Ternary Weight Analysis ------------------------------")
    inspect_ternary_weights(ternary_model)
    print()

    np.random.seed(42)
    x_np = np.random.randn(BATCH, DIM_IN).astype(np.float32)
    x = tz.tensor(x_np.tolist(), requires_grad=False)

    with tz.no_grad():
        fp32_out = fp32_model(x)

    with tz.no_grad():
        ternary_out = ternary_model(x)

    mse, max_abs, cos_sim = compute_output_metrics(fp32_out, ternary_out)

    print("-- Output Deviation (FP32 vs Ternary) -------------------")
    print_table(
        ["Metric", "Value"],
        [
            ["MSE", f"{mse:.8f}"],
            ["Max Absolute Error", f"{max_abs:.6f}"],
            ["Cosine Similarity", f"{cos_sim:.8f}"],
        ],
    )
    print()

    print("-- Latency Benchmark ------------------------------------")
    fp32_times = benchmark_forward(fp32_model, x, warmup=WARMUP, runs=RUNS)
    ternary_times = benchmark_forward(ternary_model, x, warmup=WARMUP, runs=RUNS)

    fp32_mean = np.mean(fp32_times)
    fp32_std = np.std(fp32_times)
    fp32_median = np.median(fp32_times)
    ternary_mean = np.mean(ternary_times)
    ternary_std = np.std(ternary_times)
    ternary_median = np.median(ternary_times)

    speedup = fp32_mean / ternary_mean if ternary_mean > 0 else float("inf")

    print_table(
        ["Statistic", "FP32 (ms)", "Ternary (ms)", "Ratio"],
        [
            ["Mean", f"{fp32_mean:.4f}", f"{ternary_mean:.4f}", f"{speedup:.2f}x"],
            ["Std Dev", f"{fp32_std:.4f}", f"{ternary_std:.4f}", "-"],
            ["Median", f"{fp32_median:.4f}", f"{ternary_median:.4f}",
             f"{fp32_median / ternary_median:.2f}x" if ternary_median > 0 else "-"],
            ["Min", f"{np.min(fp32_times):.4f}", f"{np.min(ternary_times):.4f}", "-"],
            ["Max", f"{np.max(fp32_times):.4f}", f"{np.max(ternary_times):.4f}", "-"],
        ],
    )
    print()

    print("-- Summary ----------------------------------------------")
    print(f"  Inference weight compression : {compression:.1f}x")
    print(f"  Memory savings (inference)   : {savings_pct:.1f}%")
    print(f"  Output cosine similarity     : {cos_sim:.6f}")
    print(f"  Output MSE                   : {mse:.8f}")
    print(f"  Latency speedup (mean)       : {speedup:.2f}x")
    print()
    print("=" * 64)
    print(" Done. Ternary quantization trades minor output deviation for")
    print(" significant memory compression and competitive latency.")
    print("=" * 64)


if __name__ == "__main__":
    main()
