from __future__ import annotations

import numpy as np
from typing import Optional, List, Dict

from .config import ModelConfig
from .quantize import QuantizedModel, QuantizedLayer
from .layers import (
    TernaryLinear, Attention, FeedForward,
    TransformerBlock, Transformer,
)


def build_model(
    qm: QuantizedModel,
    force_cpu: bool = False,
    verbose: bool = True,
) -> Transformer:
    config = qm.config

    if verbose:
        print(f"\nBuilding {config.name} transformer...")
        print(f"  Layers: {config.num_layers}")
        print(f"  Hidden: {config.hidden_size}")
        print(f"  Heads: {config.num_attention_heads} (KV: {config.num_key_value_heads})")
        print(f"  FFN: {config.intermediate_size}")
        print(f"  Vocab: {config.vocab_size}")

    blocks: List[TransformerBlock] = []

    for layer_idx, layer_data in enumerate(qm.layers):
        q_ql: QuantizedLayer = layer_data["q_proj"]
        k_ql: QuantizedLayer = layer_data["k_proj"]
        v_ql: QuantizedLayer = layer_data["v_proj"]
        o_ql: QuantizedLayer = layer_data["o_proj"]
        gate_ql: QuantizedLayer = layer_data["gate_proj"]
        up_ql: QuantizedLayer = layer_data["up_proj"]
        down_ql: QuantizedLayer = layer_data["down_proj"]

        q_proj = TernaryLinear.from_quantized(q_ql, force_cpu=force_cpu)
        k_proj = TernaryLinear.from_quantized(k_ql, force_cpu=force_cpu)
        v_proj = TernaryLinear.from_quantized(v_ql, force_cpu=force_cpu)
        o_proj = TernaryLinear.from_quantized(o_ql, force_cpu=force_cpu)

        attention = Attention(config, q_proj, k_proj, v_proj, o_proj)

        gate = TernaryLinear.from_quantized(gate_ql, force_cpu=force_cpu)
        up = TernaryLinear.from_quantized(up_ql, force_cpu=force_cpu)
        down = TernaryLinear.from_quantized(down_ql, force_cpu=force_cpu)

        ffn = FeedForward(gate, up, down)

        input_norm_w = layer_data.get("input_norm")
        post_norm_w = layer_data.get("post_attn_norm")

        block = TransformerBlock(
            attention=attention,
            feed_forward=ffn,
            input_norm_weight=input_norm_w,
            post_attn_norm_weight=post_norm_w,
            config=config,
        )
        blocks.append(block)

        if verbose:
            packed_bytes = sum(
                ql.packed_weights.nbytes
                for ql in [q_ql, k_ql, v_ql, o_ql, gate_ql, up_ql, down_ql]
            )
            print(f"  Layer {layer_idx:>2}: {packed_bytes / (1024**2):.1f} MB ternary weights")

    lm_head: Optional[TernaryLinear] = None
    lm_head_weight: Optional[np.ndarray] = None

    if qm.lm_head is not None:
        lm_head = TernaryLinear.from_quantized(qm.lm_head, force_cpu=force_cpu)
        if verbose:
            print(f"  LM Head:   {qm.lm_head.packed_weights.nbytes / (1024**2):.1f} MB ternary")
    else:
        lm_head_weight = qm.get_lm_head_weight()
        if lm_head_weight is not None:
            if verbose:
                print(f"  LM Head:   {lm_head_weight.nbytes / (1024**2):.1f} MB FP32")

    model = Transformer(
        config=config,
        blocks=blocks,
        embed_tokens=qm.embed_tokens,
        norm_weight=qm.norm_weight,
        lm_head=lm_head,
        lm_head_weight=lm_head_weight,
    )

    if verbose:
        print(f"  Model built successfully.")

    return model
