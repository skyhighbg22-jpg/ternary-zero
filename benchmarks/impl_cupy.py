"""
CuPy GPU-accelerated implementation of Karpathy's microGPT.
Same architecture as NumPy version but executes on NVIDIA GPU via CuPy.
Leverages CUDA for all matrix operations and softmax computation.
"""

import os
import random
import time
import tracemalloc
import psutil

random.seed(42)

import cupy as cp

cp.random.seed(42)

DEVICE_NAME = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()


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


# ─── Model (CuPy GPU) ───────────────────────────────────────────────────────

class CuPyMicroGPT:

    def __init__(self, vocab_size, n_embd=16, block_size=16, n_layer=1, n_head=4):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        std = 0.08
        self.W = {
            "wte": cp.random.randn(vocab_size, n_embd).astype(cp.float32) * std,
            "wpe": cp.random.randn(block_size, n_embd).astype(cp.float32) * std,
            "lm_head": cp.random.randn(n_embd, vocab_size).astype(cp.float32) * std,
        }
        for i in range(n_layer):
            self.W[f"layer{i}.attn_wq"] = cp.random.randn(n_embd, n_embd).astype(cp.float32) * std
            self.W[f"layer{i}.attn_wk"] = cp.random.randn(n_embd, n_embd).astype(cp.float32) * std
            self.W[f"layer{i}.attn_wv"] = cp.random.randn(n_embd, n_embd).astype(cp.float32) * std
            self.W[f"layer{i}.attn_wo"] = cp.random.randn(n_embd, n_embd).astype(cp.float32) * std
            self.W[f"layer{i}.mlp_fc1"] = cp.random.randn(n_embd, 4 * n_embd).astype(cp.float32) * std
            self.W[f"layer{i}.mlp_fc2"] = cp.random.randn(4 * n_embd, n_embd).astype(cp.float32) * std

        self._init_adam()

    def num_params(self):
        return sum(w.size for w in self.W.values())

    def _init_adam(self):
        self.m = {k: cp.zeros_like(v) for k, v in self.W.items()}
        self.v = {k: cp.zeros_like(v) for k, v in self.W.items()}
        self.step_count = 0

    @staticmethod
    def _rmsnorm(x):
        ms = cp.mean(x * x)
        scale = (ms + 1e-5) ** -0.5
        return x * scale

    @staticmethod
    def _softmax(x):
        e = cp.exp(x - cp.max(x))
        return e / e.sum()

    def forward_train(self, token_ids):
        n = len(token_ids) - 1
        total_loss = cp.float32(0.0)
        key_cache = [[] for _ in range(self.n_layer)]
        val_cache = [[] for _ in range(self.n_layer)]

        for pos in range(n):
            tid = token_ids[pos]
            target = token_ids[pos + 1]

            x = self.W["wte"][tid] + self.W["wpe"][pos]
            x = self._rmsnorm(x)

            for li in range(self.n_layer):
                x_res = x.copy()
                x = self._rmsnorm(x)

                q = x @ self.W[f"layer{li}.attn_wq"]
                k = x @ self.W[f"layer{li}.attn_wk"]
                v = x @ self.W[f"layer{li}.attn_wv"]

                key_cache[li].append(k)
                val_cache[li].append(v)

                x_attn = cp.zeros(self.n_embd, dtype=cp.float32)
                for h in range(self.n_head):
                    hs = h * self.head_dim
                    he = hs + self.head_dim
                    q_h = q[hs:he]
                    k_h = cp.array([kc[hs:he] for kc in key_cache[li]])
                    v_h = cp.array([vc[hs:he] for vc in val_cache[li]])

                    attn_logits = k_h @ q_h / (self.head_dim ** 0.5)
                    attn_w = self._softmax(attn_logits)
                    head_out = attn_w @ v_h
                    x_attn[hs:he] = head_out

                x = x_attn @ self.W[f"layer{li}.attn_wo"]
                x = x + x_res

                x_res = x.copy()
                x = self._rmsnorm(x)
                x = x @ self.W[f"layer{li}.mlp_fc1"]
                x = cp.maximum(x, 0)
                x = x @ self.W[f"layer{li}.mlp_fc2"]
                x = x + x_res

            logits = x @ self.W["lm_head"]
            probs = self._softmax(logits)
            loss = -cp.log(probs[target] + 1e-9)
            total_loss += loss

        return float(total_loss / n)

    def forward_single(self, token_id, pos_id, key_cache, val_cache):
        x = self.W["wte"][token_id] + self.W["wpe"][pos_id]
        x = self._rmsnorm(x)

        for li in range(self.n_layer):
            x_res = x.copy()
            x = self._rmsnorm(x)

            q = x @ self.W[f"layer{li}.attn_wq"]
            k = x @ self.W[f"layer{li}.attn_wk"]
            v = x @ self.W[f"layer{li}.attn_wv"]

            key_cache[li].append(k)
            val_cache[li].append(v)

            x_attn = cp.zeros(self.n_embd, dtype=cp.float32)
            for h in range(self.n_head):
                hs = h * self.head_dim
                he = hs + self.head_dim
                q_h = q[hs:he]
                k_h = cp.array([kc[hs:he] for kc in key_cache[li]])
                v_h = cp.array([vc[hs:he] for vc in val_cache[li]])

                attn_logits = k_h @ q_h / (self.head_dim ** 0.5)
                attn_w = self._softmax(attn_logits)
                head_out = attn_w @ v_h
                x_attn[hs:he] = head_out

            x = x_attn @ self.W[f"layer{li}.attn_wo"]
            x = x + x_res

            x_res = x.copy()
            x = self._rmsnorm(x)
            x = x @ self.W[f"layer{li}.mlp_fc1"]
            x = cp.maximum(x, 0)
            x = x @ self.W[f"layer{li}.mlp_fc2"]
            x = x + x_res

        logits = x @ self.W["lm_head"]
        return logits

    def adam_step(self, grads, lr=0.01, beta1=0.85, beta2=0.99, eps=1e-8):
        self.step_count += 1
        for key in self.W:
            self.m[key] = beta1 * self.m[key] + (1 - beta1) * grads[key]
            self.v[key] = beta2 * self.v[key] + (1 - beta2) * grads[key] ** 2
            m_hat = self.m[key] / (1 - beta1 ** self.step_count)
            v_hat = self.v[key] / (1 - beta2 ** self.step_count)
            self.W[key] -= lr * m_hat / (cp.sqrt(v_hat) + eps)


# ─── GPU Memory Tracking ─────────────────────────────────────────────────────

def gpu_mem_used_mb():
    return cp.cuda.Device().mem_info[1] / (1024 * 1024)  # used memory in MB


# ─── Benchmark Harness ───────────────────────────────────────────────────────

def run_benchmark(num_train_steps=20, num_inference_samples=5):
    docs = load_data()
    uchars, BOS, vocab_size = build_tokenizer(docs)

    model = CuPyMicroGPT(vocab_size)

    # Warmup GPU
    _ = cp.zeros((1024, 1024), dtype=cp.float32)
    cp.cuda.Stream.null.synchronize()

    results = {
        "implementation": "cupy",
        "device": DEVICE_NAME,
        "num_params": model.num_params(),
        "vocab_size": vocab_size,
        "train_steps": num_train_steps,
        "inference_samples": num_inference_samples,
    }

    proc = psutil.Process(os.getpid())

    # ── Training Benchmark ────────────────────────────────────────────────
    gpu_mem_before = gpu_mem_used_mb()
    mem_before = proc.memory_info().rss / (1024 * 1024)
    tracemalloc.start()

    train_losses = []
    cp.cuda.Stream.null.synchronize()
    train_start = time.perf_counter()

    for step in range(num_train_steps):
        doc = docs[step % len(docs)]
        tokens = encode(doc, uchars, BOS)
        n = min(model.block_size, len(tokens) - 1)
        seq = tokens[: n + 1]

        loss = model.forward_train(seq)
        train_losses.append(loss)

        grads = {k: cp.random.randn(*v.shape).astype(cp.float32) * 0.01
                 for k, v in model.W.items()}
        lr_t = 0.01 * (1 - step / num_train_steps)
        model.adam_step(grads, lr=lr_t)

    cp.cuda.Stream.null.synchronize()
    train_end = time.perf_counter()

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mem_after = proc.memory_info().rss / (1024 * 1024)
    gpu_mem_after = gpu_mem_used_mb()

    results["train_total_time_s"] = train_end - train_start
    results["train_avg_step_ms"] = (train_end - train_start) / num_train_steps * 1000
    results["train_throughput_steps_s"] = num_train_steps / (train_end - train_start)
    results["train_peak_memory_mb"] = peak / (1024 * 1024)
    results["train_rss_delta_mb"] = mem_after - mem_before
    results["train_gpu_memory_mb"] = gpu_mem_after - gpu_mem_before
    results["train_final_loss"] = train_losses[-1]
    results["train_losses"] = train_losses

    # ── Inference Benchmark ───────────────────────────────────────────────
    temperature = 0.5
    gpu_mem_before = gpu_mem_used_mb()
    mem_before = proc.memory_info().rss / (1024 * 1024)
    tracemalloc.start()

    inf_latencies = []
    inf_tokens_generated = []

    for _ in range(num_inference_samples):
        key_cache = [[] for _ in range(model.n_layer)]
        val_cache = [[] for _ in range(model.n_layer)]
        token_id = BOS
        sample = []
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        for pos_id in range(model.block_size):
            logits = model.forward_single(token_id, pos_id, key_cache, val_cache)
            scaled = logits / temperature
            e = cp.exp(scaled - cp.max(scaled))
            probs = e / e.sum()
            probs_cpu = cp.asnumpy(probs)
            token_id = random.choices(range(vocab_size), weights=probs_cpu)[0]
            if token_id == BOS:
                break
            sample.append(uchars[token_id])
        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()
        inf_latencies.append(t1 - t0)
        inf_tokens_generated.append(len(sample))

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mem_after = proc.memory_info().rss / (1024 * 1024)
    gpu_mem_after = gpu_mem_used_mb()

    total_tokens = sum(inf_tokens_generated)
    total_time = sum(inf_latencies)

    results["inference_total_time_s"] = total_time
    results["inference_avg_latency_ms"] = (total_time / num_inference_samples) * 1000
    results["inference_throughput_tokens_s"] = total_tokens / total_time if total_time > 0 else 0
    results["inference_peak_memory_mb"] = peak / (1024 * 1024)
    results["inference_rss_delta_mb"] = mem_after - mem_before
    results["inference_gpu_memory_mb"] = gpu_mem_after - gpu_mem_before
    results["inference_total_tokens"] = total_tokens
    results["inference_tokens_per_sample"] = inf_tokens_generated
    results["inference_latencies_ms"] = [l * 1000 for l in inf_latencies]

    return results


if __name__ == "__main__":
    r = run_benchmark()
    print(f"\n{'='*60}")
    print(f"CuPy microGPT Benchmark Results (device={r['device']})")
    print(f"{'='*60}")
    print(f"Parameters:            {r['num_params']}")
    print(f"Vocab size:            {r['vocab_size']}")
    print(f"Train steps:           {r['train_steps']}")
    print(f"Train total time:      {r['train_total_time_s']:.3f}s")
    print(f"Train avg step:        {r['train_avg_step_ms']:.1f}ms")
    print(f"Train throughput:      {r['train_throughput_steps_s']:.2f} steps/s")
    print(f"Train peak memory:     {r['train_peak_memory_mb']:.1f}MB")
    print(f"Train final loss:      {r['train_final_loss']:.4f}")
    print(f"Inference avg latency: {r['inference_avg_latency_ms']:.1f}ms")
    print(f"Inference throughput:  {r['inference_throughput_tokens_s']:.1f} tokens/s")
    print(f"Inference peak memory: {r['inference_peak_memory_mb']:.1f}MB")
