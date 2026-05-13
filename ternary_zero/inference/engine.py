from __future__ import annotations

import time
import numpy as np
from typing import Optional, List, Callable

from .config import ModelConfig, WeightMapping, LLAMA_WEIGHT_MAP, detect_config, PRESET_CONFIGS
from .quantize import QuantizedModel, load_model_weights
from .model import build_model
from .layers import Transformer
from .cache import KVCache
from .sampler import sample
from .tokenizer import Tokenizer


class InferenceEngine:
    def __init__(
        self,
        model: Transformer,
        tokenizer: Tokenizer,
        config: ModelConfig,
        max_seq_len: int = 2048,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.max_seq_len = max_seq_len
        self.kv_cache = KVCache(config, max_seq_len)
        self._bos_id = tokenizer.bos_token_id or config.bos_token_id
        self._eos_id = tokenizer.eos_token_id or config.eos_token_id

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: Optional[ModelConfig] = None,
        alpha: float = 0.5,
        max_seq_len: int = 2048,
        force_cpu: bool = False,
        embed_fp16: bool = True,
        lm_head_quantize: bool = True,
        verbose: bool = True,
    ) -> "InferenceEngine":
        if config is None:
            if model_path.lower() in PRESET_CONFIGS:
                config = PRESET_CONFIGS[model_path.lower()]
            else:
                config = detect_config(model_path)

        qm = load_model_weights(
            model_path, config, alpha=alpha,
            embed_fp16=embed_fp16, lm_head_quantize=lm_head_quantize,
            verbose=verbose,
        )

        model = build_model(qm, force_cpu=force_cpu, verbose=verbose)
        del qm

        try:
            tokenizer = Tokenizer.from_pretrained(model_path)
        except ImportError:
            tokenizer = Tokenizer.from_pretrained(config.name)

        return cls(model, tokenizer, config, max_seq_len=max_seq_len)

    def prefill(self, tokens: List[int], verbose: bool = False) -> np.ndarray:
        self.kv_cache.reset()
        logits = None
        for pos, token_id in enumerate(tokens):
            if pos >= self.max_seq_len:
                break
            logits = self.model.forward(token_id, self.kv_cache, pos)
            if verbose and pos % 100 == 0:
                print(f"  Prefill: {pos}/{len(tokens)} tokens", end="\r")
        if verbose:
            print(f"  Prefill: {len(tokens)}/{len(tokens)} tokens - done")
        return logits

    def decode_next(self, token_id: int, position: int) -> np.ndarray:
        return self.model.forward(token_id, self.kv_cache, position)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        stop_on_eos: bool = True,
        stream: bool = False,
        callback: Optional[Callable[[str], None]] = None,
        verbose: bool = True,
    ) -> str:
        add_bos = True
        tokens = self.tokenizer.encode(prompt, add_special_tokens=add_bos)
        prompt_len = len(tokens)

        if verbose:
            print(f"Prompt: {len(prompt)} chars, {prompt_len} tokens")
            print(f"Generating up to {max_tokens} tokens...")
            print("-" * 60)

        t_start = time.perf_counter()
        logits = self.prefill(tokens, verbose=verbose)
        t_prefill = time.perf_counter() - t_start

        generated_tokens = []
        generated_text = ""
        position = prompt_len

        if logits is not None:
            first_token = sample(logits, temperature=temperature, top_k=top_k, top_p=top_p)
            generated_tokens.append(first_token)

            first_piece = self.tokenizer.decode([first_token], skip_special_tokens=True)
            generated_text += first_piece

            if stream and first_piece:
                print(first_piece, end="", flush=True)
            if callback and first_piece:
                callback(first_piece)

            position += 1

        t_decode_start = time.perf_counter()

        for i in range(1, max_tokens):
            if position >= self.max_seq_len:
                if verbose:
                    print(f"\n[Max sequence length {self.max_seq_len} reached]")
                break

            prev_token = generated_tokens[-1]
            logits = self.decode_next(prev_token, position)

            token = sample(logits, temperature=temperature, top_k=top_k, top_p=top_p)
            generated_tokens.append(token)

            if stop_on_eos and token == self._eos_id:
                generated_tokens.pop()
                break

            piece = self.tokenizer.decode([token], skip_special_tokens=True)
            generated_text += piece

            if stream and piece:
                print(piece, end="", flush=True)
            if callback and piece:
                callback(piece)

            position += 1

        t_decode = time.perf_counter() - t_decode_start
        t_total = time.perf_counter() - t_start

        if stream:
            print()

        if verbose:
            num_generated = len(generated_tokens)
            tokens_per_sec = num_generated / t_decode if t_decode > 0 else 0
            print("-" * 60)
            print(f"Generated {num_generated} tokens in {t_total:.2f}s")
            print(f"  Prefill: {t_prefill:.3f}s ({prompt_len} tokens)")
            print(f"  Decode:  {t_decode:.3f}s ({tokens_per_sec:.1f} tok/s)")
            print(f"  Total:   {t_total:.3f}s")

        return generated_text

    def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        verbose: bool = True,
    ) -> str:
        prompt_parts = []
        if system_prompt:
            prompt_parts.append(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>")
        prompt_parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
        prompt = "".join(prompt_parts)

        return self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            verbose=verbose,
        )

    def benchmark(
        self,
        prompt: str = "Hello, I am a language model.",
        max_tokens: int = 128,
        verbose: bool = True,
    ) -> dict:
        tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        prompt_len = len(tokens)

        t_start = time.perf_counter()
        self.prefill(tokens)
        t_prefill = time.perf_counter() - t_start

        position = prompt_len
        logits = None
        t_decode_start = time.perf_counter()

        for i in range(max_tokens):
            if position >= self.max_seq_len:
                break
            if logits is None:
                token = sample(np.zeros(self.config.vocab_size))
            else:
                token = sample(logits, temperature=0.0)
            logits = self.model.forward(token, self.kv_cache, position)
            position += 1

        t_decode = time.perf_counter() - t_decode_start

        result = {
            "prompt_tokens": prompt_len,
            "generated_tokens": max_tokens,
            "prefill_time_s": t_prefill,
            "decode_time_s": t_decode,
            "prefill_tokens_per_sec": prompt_len / t_prefill if t_prefill > 0 else 0,
            "decode_tokens_per_sec": max_tokens / t_decode if t_decode > 0 else 0,
            "kv_cache_mb": self.kv_cache.memory_mb(),
        }

        if verbose:
            print(f"\nBenchmark Results:")
            print(f"  Prompt tokens:     {result['prompt_tokens']}")
            print(f"  Generated tokens:  {result['generated_tokens']}")
            print(f"  Prefill:           {result['prefill_time_s']:.3f}s ({result['prefill_tokens_per_sec']:.1f} tok/s)")
            print(f"  Decode:            {result['decode_time_s']:.3f}s ({result['decode_tokens_per_sec']:.1f} tok/s)")
            print(f"  KV cache:          {result['kv_cache_mb']:.1f} MB")

        return result
