"""
Pure Python implementation of Karpathy's microGPT.
Baseline: scalar-level computation with zero external dependencies.
Every operation is hand-written using Python builtins only.
"""

import os
import math
import random
import time
import tracemalloc
import psutil

random.seed(42)


# ─── Data ────────────────────────────────────────────────────────────────────

def load_data(path="input.txt"):
    if not os.path.exists(path):
        import urllib.request
        names_url = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
        urllib.request.urlretrieve(names_url, path)
    docs = [line.strip() for line in open(path) if line.strip()]
    random.shuffle(docs)
    return docs


# ─── Tokenizer ───────────────────────────────────────────────────────────────

def build_tokenizer(docs):
    uchars = sorted(set("".join(docs)))
    BOS = len(uchars)
    vocab_size = len(uchars) + 1
    return uchars, BOS, vocab_size


def encode(doc, uchars, BOS):
    return [BOS] + [uchars.index(ch) for ch in doc] + [BOS]


# ─── Autograd ────────────────────────────────────────────────────────────────

class Value:
    __slots__ = ("data", "grad", "_children", "_local_grads")

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        return Value(
            self.data**other, (self,), (other * self.data ** (other - 1),)
        )

    def log(self):
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def exp(self):
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, lg in zip(v._children, v._local_grads):
                child.grad += lg * v.grad


# ─── Model Architecture ──────────────────────────────────────────────────────

def init_params(vocab_size, n_embd, block_size, n_layer):
    matrix = lambda nout, nin, std=0.08: [
        [Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)
    ]
    state_dict = {
        "wte": matrix(vocab_size, n_embd),
        "wpe": matrix(block_size, n_embd),
        "lm_head": matrix(vocab_size, n_embd),
    }
    for i in range(n_layer):
        state_dict[f"layer{i}.attn_wq"] = matrix(n_embd, n_embd)
        state_dict[f"layer{i}.attn_wk"] = matrix(n_embd, n_embd)
        state_dict[f"layer{i}.attn_wv"] = matrix(n_embd, n_embd)
        state_dict[f"layer{i}.attn_wo"] = matrix(n_embd, n_embd)
        state_dict[f"layer{i}.mlp_fc1"] = matrix(4 * n_embd, n_embd)
        state_dict[f"layer{i}.mlp_fc2"] = matrix(n_embd, 4 * n_embd)
    params = [p for mat in state_dict.values() for row in mat for p in row]
    return state_dict, params


def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]


def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def gpt(token_id, pos_id, keys, values, state_dict, n_layer, n_head, head_dim):
    tok_emb = state_dict["wte"][token_id]
    pos_emb = state_dict["wpe"][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for li in range(n_layer):
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f"layer{li}.attn_wq"])
        k = linear(x, state_dict[f"layer{li}.attn_wk"])
        v = linear(x, state_dict[f"layer{li}.attn_wv"])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs : hs + head_dim]
            k_h = [ki[hs : hs + head_dim] for ki in keys[li]]
            v_h = [vi[hs : hs + head_dim] for vi in values[li]]
            attn_logits = [
                sum(q_h[j] * k_h[t][j] for j in range(head_dim))
                / head_dim**0.5
                for t in range(len(k_h))
            ]
            attn_weights = softmax(attn_logits)
            head_out = [
                sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(head_dim)
            ]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f"layer{li}.attn_wo"])
        x = [a + b for a, b in zip(x, x_residual)]

        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f"layer{li}.mlp_fc1"])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f"layer{li}.mlp_fc2"])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict["lm_head"])
    return logits


# ─── Benchmark Harness ───────────────────────────────────────────────────────

def run_benchmark(num_train_steps=20, num_inference_samples=5):
    """Run a controlled benchmark of the pure Python microGPT."""

    docs = load_data()
    uchars, BOS, vocab_size = build_tokenizer(docs)

    n_layer = 1
    n_embd = 16
    block_size = 16
    n_head = 4
    head_dim = n_embd // n_head

    state_dict, params = init_params(vocab_size, n_embd, block_size, n_layer)

    learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
    m = [0.0] * len(params)
    v = [0.0] * len(params)

    results = {
        "implementation": "pure_python",
        "num_params": len(params),
        "vocab_size": vocab_size,
        "train_steps": num_train_steps,
        "inference_samples": num_inference_samples,
    }

    # ── Training Benchmark ────────────────────────────────────────────────
    proc = psutil.Process(os.getpid())
    mem_before_train = proc.memory_info().rss / (1024 * 1024)
    tracemalloc.start()

    train_losses = []
    train_start = time.perf_counter()

    for step in range(num_train_steps):
        doc = docs[step % len(docs)]
        tokens = encode(doc, uchars, BOS)
        n = min(block_size, len(tokens) - 1)

        keys = [[] for _ in range(n_layer)]
        values = [[] for _ in range(n_layer)]
        losses = []
        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            logits = gpt(
                token_id, pos_id, keys, values, state_dict, n_layer, n_head, head_dim
            )
            probs = softmax(logits)
            loss_t = -probs[target_id].log()
            losses.append(loss_t)
        loss = (1 / n) * sum(losses)
        loss.backward()

        lr_t = learning_rate * (1 - step / num_train_steps)
        for i, p in enumerate(params):
            m[i] = beta1 * m[i] + (1 - beta1) * p.grad
            v[i] = beta2 * v[i] + (1 - beta2) * p.grad**2
            m_hat = m[i] / (1 - beta1 ** (step + 1))
            v_hat = v[i] / (1 - beta2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat**0.5 + eps_adam)
            p.grad = 0

        train_losses.append(loss.data)

    train_end = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mem_after_train = proc.memory_info().rss / (1024 * 1024)

    results["train_total_time_s"] = train_end - train_start
    results["train_avg_step_ms"] = (train_end - train_start) / num_train_steps * 1000
    results["train_throughput_steps_s"] = num_train_steps / (train_end - train_start)
    results["train_peak_memory_mb"] = peak / (1024 * 1024)
    results["train_rss_delta_mb"] = mem_after_train - mem_before_train
    results["train_final_loss"] = train_losses[-1]
    results["train_losses"] = train_losses

    # ── Inference Benchmark ───────────────────────────────────────────────
    temperature = 0.5
    mem_before_inf = proc.memory_info().rss / (1024 * 1024)
    tracemalloc.start()

    inf_latencies = []
    inf_tokens_generated = []

    for _ in range(num_inference_samples):
        keys = [[] for _ in range(n_layer)]
        values = [[] for _ in range(n_layer)]
        token_id = BOS
        sample = []
        t0 = time.perf_counter()
        for pos_id in range(block_size):
            logits = gpt(
                token_id, pos_id, keys, values, state_dict, n_layer, n_head, head_dim
            )
            probs = softmax([l / temperature for l in logits])
            token_id = random.choices(
                range(vocab_size), weights=[p.data for p in probs]
            )[0]
            if token_id == BOS:
                break
            sample.append(uchars[token_id])
        t1 = time.perf_counter()
        inf_latencies.append(t1 - t0)
        inf_tokens_generated.append(len(sample))

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mem_after_inf = proc.memory_info().rss / (1024 * 1024)

    total_inf_tokens = sum(inf_tokens_generated)
    total_inf_time = sum(inf_latencies)

    results["inference_total_time_s"] = total_inf_time
    results["inference_avg_latency_ms"] = (total_inf_time / num_inference_samples) * 1000
    results["inference_throughput_tokens_s"] = (
        total_inf_tokens / total_inf_time if total_inf_time > 0 else 0
    )
    results["inference_peak_memory_mb"] = peak / (1024 * 1024)
    results["inference_rss_delta_mb"] = mem_after_inf - mem_before_inf
    results["inference_total_tokens"] = total_inf_tokens
    results["inference_tokens_per_sample"] = inf_tokens_generated
    results["inference_latencies_ms"] = [l * 1000 for l in inf_latencies]

    return results


if __name__ == "__main__":
    r = run_benchmark()
    print(f"\n{'='*60}")
    print(f"Pure Python microGPT Benchmark Results")
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
