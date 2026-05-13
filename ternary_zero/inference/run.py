#!/usr/bin/env python3
"""
Ternary-Zero Inference Engine — CLI Runner
============================================

Run Llama models with ternary-quantized weights on GPU or CPU.

Usage:
    python -m ternary_zero.inference.run <model_path> [options]

Examples:
    python -m ternary_zero.inference.run ./models/llama-3.2-1b "Hello, world"
    python -m ternary_zero.inference.run meta-llama/Llama-3.2-1B --chat
    python -m ternary_zero.inference.run ./models/llama-3-8b --benchmark
"""

import argparse
import sys
import os


def main():
    parser = argparse.ArgumentParser(
        description="Ternary-Zero LLM Inference Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Generate text:
    python -m ternary_zero.inference.run ./models/llama-3.2-1b "The meaning of life is"

  Interactive chat:
    python -m ternary_zero.inference.run ./models/llama-3.2-1b --chat

  Benchmark performance:
    python -m ternary_zero.inference.run ./models/llama-3.2-1b --benchmark

  Stream output:
    python -m ternary_zero.inference.run ./models/llama-3.2-1b "Write a poem" --stream
        """,
    )

    parser.add_argument(
        "model",
        type=str,
        help="Path to model directory (with config.json and weights) or preset name",
    )
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",
        default=None,
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-K sampling (default: 0 = disabled)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-P (nucleus) sampling (default: 1.0 = disabled)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Ternary quantization alpha threshold (default: 0.5)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Interactive chat mode",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream tokens as they are generated",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU-only inference (no CUDA kernels)",
    )
    parser.add_argument(
        "--no-quantize-lm-head",
        action="store_true",
        help="Keep LM head in FP32 instead of ternary",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output (default: True)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    if args.quiet:
        args.verbose = False

    from .engine import InferenceEngine
    from .config import ModelConfig, detect_config, PRESET_CONFIGS

    print("=" * 60)
    print("  Ternary-Zero Inference Engine")
    print("=" * 60)

    config = None
    model_path = args.model
    if model_path.lower() in PRESET_CONFIGS:
        config = PRESET_CONFIGS[model_path.lower()]
        print(f"  Using preset: {config.name}")

    try:
        engine = InferenceEngine.from_pretrained(
            model_path=model_path,
            config=config,
            alpha=args.alpha,
            max_seq_len=args.max_seq_len,
            force_cpu=args.force_cpu,
            lm_head_quantize=not args.no_quantize_lm_head,
            verbose=args.verbose,
        )
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print(f"\nMake sure the model directory exists and contains:")
        print(f"  - config.json")
        print(f"  - *.safetensors or *.bin weight files")
        print(f"  - tokenizer.json or tokenizer files")
        sys.exit(1)
    except ImportError as e:
        print(f"\nError: {e}")
        print(f"\nInstall required packages:")
        print(f"  pip install safetensors transformers")
        sys.exit(1)

    print()
    print(f"  Model: {engine.config.name}")
    print(f"  Quantization: Ternary (alpha={args.alpha})")
    print(f"  Max sequence length: {args.max_seq_len}")
    print(f"  KV-cache memory: {engine.kv_cache.memory_mb():.1f} MB")
    print()

    if args.benchmark:
        print("Running benchmark...")
        engine.benchmark(verbose=True)
        return

    if args.chat:
        print("Interactive chat mode (type 'quit' to exit)")
        print("-" * 60)
        system_prompt = "You are a helpful assistant."

        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            if not user_input:
                continue

            response = engine.chat(
                message=user_input,
                system_prompt=system_prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                verbose=args.verbose,
            )
            print(f"\nAssistant: {response}")
        return

    prompt = args.prompt
    if prompt is None:
        prompt = "Hello, I am a language model."
        print(f"No prompt provided, using default: '{prompt}'")
        print()

    output = engine.generate(
        prompt=prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        stream=args.stream,
        verbose=args.verbose,
    )

    if not args.stream:
        print()
        print("=" * 60)
        print("Generated text:")
        print("-" * 60)
        print(output)
        print("=" * 60)


if __name__ == "__main__":
    main()
