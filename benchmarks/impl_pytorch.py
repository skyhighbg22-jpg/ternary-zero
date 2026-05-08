"""
PyTorch implementation of Karpathy's microGPT.
Uses torch tensors with autograd for automatic differentiation.
KV cache entries are detached to prevent quadratic graph growth.
"""

import os
import random
import time
import tracemalloc
import psutil

random.seed(42)

import torch
import torch.nn.functional as F

torch.manual_seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


class TorchMicroGPT(torch.nn.Module):

    def __init__(self, vocab_size, n_embd=16, block_size=16, n_layer=1, n_head=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.wte = torch.nn.Embedding(vocab_size, n_embd)
        self.wpe = torch.nn.Embedding(block_size, n_embd)
        self.lm_head = torch.nn.Linear(n_embd, vocab_size, bias=False)

        self.layers = torch.nn.ModuleList()
        for _ in range(n_layer):
            self.layers.append(torch.nn.ModuleDict({
                "attn_wq": torch.nn.Linear(n_embd, n_embd, bias=False),
                "attn_wk": torch.nn.Linear(n_embd, n_embd, bias=False),
                "attn_wv": torch.nn.Linear(n_embd, n_embd, bias=False),
                "attn_wo": torch.nn.Linear(n_embd, n_embd, bias=False),
                "mlp_fc1": torch.nn.Linear(n_embd, 4 * n_embd, bias=False),
                "mlp_fc2": torch.nn.Linear(4 * n_embd, n_embd, bias=False),
            }))

        std = 0.08
        for p in self.parameters():
            torch.nn.init.normal_(p, mean=0.0, std=std)

    def forward_single(self, token_id, pos_id, key_cache, val_cache):
        """Forward one token. Keys/values from cache are detached (no grad)."""
        tid = torch.tensor(token_id, dtype=torch.long, device=DEVICE)
        pos_t = torch.tensor(pos_id, dtype=torch.long, device=DEVICE)

        x = self.wte(tid) + self.wpe(pos_t)
        x = F.rms_norm(x, (self.n_embd,))

        for li, layer in enumerate(self.layers):
            x_res = x
            x = F.rms_norm(x, (self.n_embd,))

            q = layer["attn_wq"](x)
            k = layer["attn_wk"](x)
            v = layer["attn_wv"](x)

            key_cache[li].append(k.detach())
            val_cache[li].append(v.detach())

            x_attn = torch.zeros(self.n_embd, device=DEVICE)
            for h in range(self.n_head):
                hs = h * self.head_dim
                he = hs + self.head_dim
                q_h = q[hs:he]
                k_h = torch.stack([kc[hs:he] for kc in key_cache[li]])
                v_h = torch.stack([vc[hs:he] for vc in val_cache[li]])

                attn_logits = k_h @ q_h / (self.head_dim ** 0.5)
                attn_w = F.softmax(attn_logits, dim=0)
                head_out = attn_w @ v_h
                x_attn[hs:he] = head_out

            x = layer["attn_wo"](x_attn)
            x = x + x_res

            x_res = x
            x = F.rms_norm(x, (self.n_embd,))
            x = layer["mlp_fc1"](x)
            x = F.relu(x)
            x = layer["mlp_fc2"](x)
            x = x + x_res

        logits = self.lm_head(x)
        return logits

    def forward_train(self, token_ids):
        """Forward sequence, computing per-position loss with detached KV cache."""
        n = len(token_ids) - 1
        key_cache = [[] for _ in range(self.n_layer)]
        val_cache = [[] for _ in range(self.n_layer)]

        total_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
        for pos in range(n):
            logits = self.forward_single(token_ids[pos], pos, key_cache, val_cache)
            target = torch.tensor(token_ids[pos + 1], dtype=torch.long, device=DEVICE)
            loss_t = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
            total_loss = total_loss + loss_t

        return total_loss / n


def run_benchmark(num_train_steps=20, num_inference_samples=5):
    docs = load_data()
    uchars, BOS, vocab_size = build_tokenizer(docs)

    model = TorchMicroGPT(vocab_size).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, betas=(0.85, 0.99), eps=1e-8
    )

    results = {
        "implementation": "pytorch",
        "device": DEVICE,
        "num_params": sum(p.numel() for p in model.parameters()),
        "vocab_size": vocab_size,
        "train_steps": num_train_steps,
        "inference_samples": num_inference_samples,
    }

    proc = psutil.Process(os.getpid())

    # ── Training ──
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
        train_losses.append(loss.item())

    if DEVICE == "cuda":
        torch.cuda.synchronize()
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

    # ── Inference ──
    temperature = 0.5
    model.eval()
    mem_before = proc.memory_info().rss / (1024 * 1024)
    tracemalloc.start()
    inf_latencies = []
    inf_tokens_generated = []

    with torch.no_grad():
        for _ in range(num_inference_samples):
            key_cache = [[] for _ in range(model.n_layer)]
            val_cache = [[] for _ in range(model.n_layer)]
            token_id = BOS
            sample = []
            t0 = time.perf_counter()
            for pos_id in range(model.block_size):
                logits = model.forward_single(token_id, pos_id, key_cache, val_cache)
                probs = F.softmax(logits / temperature, dim=0)
                token_id = torch.multinomial(probs, 1).item()
                if token_id == BOS:
                    break
                sample.append(uchars[token_id])
            if DEVICE == "cuda":
                torch.cuda.synchronize()
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
    r = run_benchmark()
    print(f"\n{'='*60}")
    print(f"PyTorch microGPT Benchmark Results (device={r['device']})")
    print(f"{'='*60}")
    print(f"Parameters:            {r['num_params']}")
    print(f"Train steps:           {r['train_steps']}")
    print(f"Train total time:      {r['train_total_time_s']:.3f}s")
    print(f"Train avg step:        {r['train_avg_step_ms']:.1f}ms")
    print(f"Train throughput:      {r['train_throughput_steps_s']:.2f} steps/s")
    print(f"Train peak memory:     {r['train_peak_memory_mb']:.1f}MB")
    print(f"Train final loss:      {r['train_final_loss']:.4f}")
    print(f"Inference avg latency: {r['inference_avg_latency_ms']:.1f}ms")
    print(f"Inference throughput:  {r['inference_throughput_tokens_s']:.1f} tokens/s")
    print(f"Inference peak memory: {r['inference_peak_memory_mb']:.1f}MB")
