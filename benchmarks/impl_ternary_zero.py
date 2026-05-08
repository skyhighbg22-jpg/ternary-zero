"""
Ternary-Zero implementation of Karpathy's microGPT.
Uses the ternary_zero library's Tensor, autograd, nn modules.
Includes both standard Linear and BitLinear (2-bit ternary quantized) variants.
This showcases Ternary-Zero's core value proposition: sub-byte quantized inference.
"""

import os
import sys
import random
import time
import tracemalloc
import psutil
import numpy as np

# Add project root to path for ternary_zero import
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

random.seed(42)

import ternary_zero as tz


# ─── Data ────────────────────────────────────────────────────────────────────

def load_data(path="input.txt"):
    if not os.path.exists(path):
        import urllib.request
        names_url = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
        urllib.request.urlretrieve(names_url, path)
    docs = [line.strip() for line in open(path) if line.strip()]
    random.shuffle(docs)
    return docs


def build_tokenizer(docs):
    uchars = sorted(set("".join(docs)))
    BOS = len(uchars)
    vocab_size = len(uchars) + 1
    return uchars, BOS, vocab_size


def encode(doc, uchars, BOS):
    return [BOS] + [uchars.index(ch) for ch in doc] + [BOS]


# ─── RMSNorm Module ──────────────────────────────────────────────────────────

class RMSNorm(tz.nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = tz.nn.Parameter(np.ones(dim, dtype=np.float32))

    def forward(self, x):
        if isinstance(x.data, np.ndarray):
            data = x.data
        else:
            import torch
            data = x.data.float()
            if isinstance(data, torch.Tensor):
                rms = torch.mean(data * data)
                scale = (rms + self.eps) ** -0.5
                return tz.Tensor(data * scale * self.weight.data)

        ms = np.mean(data * data)
        scale = np.float32((ms + self.eps) ** -0.5)
        return tz.Tensor(data * scale * self.weight.data)


# ─── Attention Module ────────────────────────────────────────────────────────

class MultiHeadAttention(tz.nn.Module):
    def __init__(self, n_embd, n_head, use_bitlinear=False):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        LinearCls = tz.nn.BitLinear if use_bitlinear else tz.nn.Linear
        kw = {"alpha": 0.5} if use_bitlinear else {}
        self.wq = LinearCls(n_embd, n_embd, bias=False, **kw)
        self.wk = LinearCls(n_embd, n_embd, bias=False, **kw)
        self.wv = LinearCls(n_embd, n_embd, bias=False, **kw)
        self.wo = LinearCls(n_embd, n_embd, bias=False, **kw)

    def forward(self, x, key_cache, val_cache):
        import torch
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        key_cache.append(k)
        val_cache.append(v)

        hd = self.head_dim
        x_attn_parts = []
        for h in range(self.n_head):
            hs = h * hd
            q_h = q.data[hs:hs + hd]
            k_all = torch.stack([kc.data[hs:hs + hd] for kc in key_cache])
            v_all = torch.stack([vc.data[hs:hs + hd] for vc in val_cache])

            attn_logits = k_all @ q_h / (hd ** 0.5)
            attn_w = torch.softmax(attn_logits, dim=0)
            head_out = attn_w @ v_all
            x_attn_parts.append(head_out)

        x_attn = torch.cat(x_attn_parts)
        out = self.wo(tz.Tensor(x_attn))
        return out


# ─── MLP Module ──────────────────────────────────────────────────────────────

class MLP(tz.nn.Module):
    def __init__(self, n_embd, use_bitlinear=False):
        super().__init__()
        LinearCls = tz.nn.BitLinear if use_bitlinear else tz.nn.Linear
        kw = {"alpha": 0.5} if use_bitlinear else {}
        self.fc1 = LinearCls(n_embd, 4 * n_embd, bias=False, **kw)
        self.fc2 = LinearCls(4 * n_embd, n_embd, bias=False, **kw)

    def forward(self, x):
        x = tz.nn.F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ─── Transformer Block ──────────────────────────────────────────────────────

class TransformerBlock(tz.nn.Module):
    def __init__(self, n_embd, n_head, use_bitlinear=False):
        super().__init__()
        self.ln1 = RMSNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, use_bitlinear)
        self.ln2 = RMSNorm(n_embd)
        self.mlp = MLP(n_embd, use_bitlinear)

    def forward(self, x, key_cache, val_cache):
        import torch
        x_res = x
        x = self.ln1(x)
        x = self.attn(x, key_cache, val_cache)
        x = tz.Tensor(x.data + x_res.data)

        x_res = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = tz.Tensor(x.data + x_res.data)
        return x


# ─── microGPT Model ─────────────────────────────────────────────────────────

class TernaryZeroMicroGPT(tz.nn.Module):
    def __init__(self, vocab_size, n_embd=16, block_size=16, n_layer=1,
                 n_head=4, use_bitlinear=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.use_bitlinear = use_bitlinear

        std = 0.08
        self.wte = tz.nn.Embedding(vocab_size, n_embd)
        self.wpe = tz.nn.Embedding(block_size, n_embd)
        self.ln_f = RMSNorm(n_embd)

        self.layers = tz.nn.ModuleList()
        for _ in range(n_layer):
            self.layers.append(
                TransformerBlock(n_embd, n_head, use_bitlinear)
            )

        LinearCls = tz.nn.BitLinear if use_bitlinear else tz.nn.Linear
        kw = {"alpha": 0.5} if use_bitlinear else {}
        self.lm_head = LinearCls(n_embd, vocab_size, bias=False, **kw)

    def forward_single(self, token_id, pos_id, key_caches, val_caches):
        import torch
        tok_emb = self.wte(tz.tensor(np.array([token_id], dtype=np.int64)))
        pos_emb = self.wpe(tz.tensor(np.array([pos_id], dtype=np.int64)))
        x = tz.Tensor(tok_emb.data.squeeze(0) + pos_emb.data.squeeze(0))
        x = self.ln_f(x)

        for li, layer in enumerate(self.layers):
            x = layer(x, key_caches[li], val_caches[li])

        logits = self.lm_head(x)
        return logits

    def forward_train(self, token_ids):
        import torch
        n = len(token_ids) - 1
        key_caches = [[] for _ in range(self.n_layer)]
        val_caches = [[] for _ in range(self.n_layer)]

        total_loss_val = None
        for pos in range(n):
            logits = self.forward_single(
                token_ids[pos], pos, key_caches, val_caches
            )
            target = tz.tensor(np.array([token_ids[pos + 1]], dtype=np.int64))
            if logits.ndim == 1:
                logits_2d = tz.Tensor(logits.data.unsqueeze(0))
            else:
                logits_2d = logits
            loss_t = tz.nn.F.cross_entropy(logits_2d, target)
            if total_loss_val is None:
                total_loss_val = loss_t
            else:
                total_loss_val = tz.Tensor(total_loss_val.data + loss_t.data)

        return tz.Tensor(total_loss_val.data / n)

    def num_params(self):
        count = 0
        for p in self.parameters():
            count += int(np.prod(p.shape))
        return count

    def weight_memory_bytes(self):
        """Calculate weight memory footprint."""
        count = 0
        for p in self.parameters():
            count += int(np.prod(p.shape))
        if self.use_bitlinear:
            return count * 2 // 8  # 2-bit ternary
        else:
            return count * 4  # float32


# ─── Quantization Analysis ───────────────────────────────────────────────────

def analyze_quantization(model):
    """Analyze ternary quantization stats for BitLinear layers."""
    stats = {"total_params": 0, "quantized_params": 0, "bitlinear_layers": 0}
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            for subname, submod in module._modules.items() if hasattr(module, "_modules") else []:
                if isinstance(submod, tz.nn.BitLinear):
                    stats["bitlinear_layers"] += 1
                    w = submod.weight
                    stats["quantized_params"] += int(np.prod(w.shape))
                stats["total_params"] += int(np.prod(submod.weight.shape)) if hasattr(submod, "weight") else 0
    return stats


# ─── Benchmark Harness ───────────────────────────────────────────────────────

def run_benchmark(num_train_steps=20, num_inference_samples=5,
                  use_bitlinear=False):
    label = "ternary_zero_bitlinear" if use_bitlinear else "ternary_zero"
    docs = load_data()
    uchars, BOS, vocab_size = build_tokenizer(docs)

    model = TernaryZeroMicroGPT(
        vocab_size, use_bitlinear=use_bitlinear
    )
    model.training = True

    optimizer = tz.optim.Adam(model.parameters(), lr=0.01)

    num_params = model.num_params()
    weight_bytes = model.weight_memory_bytes()

    results = {
        "implementation": label,
        "num_params": num_params,
        "weight_bytes": weight_bytes,
        "quantized": use_bitlinear,
        "vocab_size": vocab_size,
        "train_steps": num_train_steps,
        "inference_samples": num_inference_samples,
    }

    proc = psutil.Process(os.getpid())

    # ── Training ──────────────────────────────────────────────────────────
    mem_before = proc.memory_info().rss / (1024 * 1024)
    tracemalloc.start()
    train_losses = []
    train_start = time.perf_counter()

    for step in range(num_train_steps):
        doc = docs[step % len(docs)]
        tokens = encode(doc, uchars, BOS)
        n = min(model.block_size, len(tokens) - 1)
        seq = tokens[: n + 1]

        optimizer.zero_grad()
        loss = model.forward_train(seq)
        loss.backward()
        optimizer.step()

        train_losses.append(float(loss.data.item()))

    train_end = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mem_after = proc.memory_info().rss / (1024 * 1024)

    results["train_total_time_s"] = train_end - train_start
    results["train_avg_step_ms"] = (train_end - train_start) / num_train_steps * 1000
    results["train_throughput_steps_s"] = num_train_steps / (train_end - train_start)
    results["train_peak_memory_mb"] = peak / (1024 * 1024)
    results["train_rss_delta_mb"] = mem_after - mem_before
    results["train_final_loss"] = train_losses[-1]
    results["train_losses"] = train_losses

    # ── Inference ─────────────────────────────────────────────────────────
    temperature = 0.5
    model.training = False
    if use_bitlinear:
        for layer in model.layers:
            if hasattr(layer.attn, "wq"):
                for mod_name in ["wq", "wk", "wv", "wo"]:
                    m = getattr(layer.attn, mod_name)
                    if isinstance(m, tz.nn.BitLinear):
                        m.quantize_weights()
            if hasattr(layer.mlp, "fc1"):
                for mod_name in ["fc1", "fc2"]:
                    m = getattr(layer.mlp, mod_name)
                    if isinstance(m, tz.nn.BitLinear):
                        m.quantize_weights()

    mem_before = proc.memory_info().rss / (1024 * 1024)
    tracemalloc.start()
    inf_latencies = []
    inf_tokens_generated = []

    for _ in range(num_inference_samples):
        key_caches = [[] for _ in range(model.n_layer)]
        val_caches = [[] for _ in range(model.n_layer)]
        token_id = BOS
        sample = []
        t0 = time.perf_counter()
        for pos_id in range(model.block_size):
            logits = model.forward_single(token_id, pos_id, key_caches, val_caches)
            logits_data = logits.data.float()
            import torch
            scaled = logits_data / temperature
            probs = torch.softmax(scaled, dim=0)
            token_id = int(torch.multinomial(probs, 1).item())
            if token_id == BOS:
                break
            sample.append(uchars[token_id])
        t1 = time.perf_counter()
        inf_latencies.append(t1 - t0)
        inf_tokens_generated.append(len(sample))

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mem_after = proc.memory_info().rss / (1024 * 1024)

    total_tokens = sum(inf_tokens_generated)
    total_time = sum(inf_latencies)

    results["inference_total_time_s"] = total_time
    results["inference_avg_latency_ms"] = (total_time / num_inference_samples) * 1000
    results["inference_throughput_tokens_s"] = total_tokens / total_time if total_time > 0 else 0
    results["inference_peak_memory_mb"] = peak / (1024 * 1024)
    results["inference_rss_delta_mb"] = mem_after - mem_before
    results["inference_total_tokens"] = total_tokens
    results["inference_tokens_per_sample"] = inf_tokens_generated
    results["inference_latencies_ms"] = [l * 1000 for l in inf_latencies]

    return results


if __name__ == "__main__":
    import torch
    print(f"Ternary-Zero version: {tz.__version__}")
    print(f"PyTorch backend: {torch.__version__}")
    print(f"CUDA available: {tz.is_cuda_available()}")
    print()

    # Standard (FP32) variant
    r = run_benchmark(use_bitlinear=False)
    print(f"\n{'='*60}")
    print(f"Ternary-Zero microGPT (FP32 Linear)")
    print(f"{'='*60}")
    print(f"Parameters:            {r['num_params']}")
    print(f"Weight memory:         {r['weight_bytes']} bytes")
    print(f"Train avg step:        {r['train_avg_step_ms']:.1f}ms")
    print(f"Train throughput:      {r['train_throughput_steps_s']:.2f} steps/s")
    print(f"Train peak memory:     {r['train_peak_memory_mb']:.1f}MB")
    print(f"Train final loss:      {r['train_final_loss']:.4f}")
    print(f"Inference avg latency: {r['inference_avg_latency_ms']:.1f}ms")
    print(f"Inference throughput:  {r['inference_throughput_tokens_s']:.1f} tokens/s")

    # BitLinear (ternary quantized) variant
    r2 = run_benchmark(use_bitlinear=True)
    print(f"\n{'='*60}")
    print(f"Ternary-Zero microGPT (BitLinear 2-bit Ternary)")
    print(f"{'='*60}")
    print(f"Parameters:            {r2['num_params']}")
    print(f"Weight memory:         {r2['weight_bytes']} bytes ({r['weight_bytes']}x -> {r['weight_bytes']/max(r2['weight_bytes'],1):.0f}x compression)")
    print(f"Train avg step:        {r2['train_avg_step_ms']:.1f}ms")
    print(f"Train throughput:      {r2['train_throughput_steps_s']:.2f} steps/s")
    print(f"Train peak memory:     {r2['train_peak_memory_mb']:.1f}MB")
    print(f"Train final loss:      {r2['train_final_loss']:.4f}")
    print(f"Inference avg latency: {r2['inference_avg_latency_ms']:.1f}ms")
    print(f"Inference throughput:  {r2['inference_throughput_tokens_s']:.1f} tokens/s")
