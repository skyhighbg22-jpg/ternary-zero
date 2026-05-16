from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List


@dataclass(frozen=True)
class ModelConfig:
    name: str
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    tie_word_embeddings: bool = False
    bos_token_id: int = 128000
    eos_token_id: int = 128001
    model_type: str = "llama"

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_kv_heads(self) -> int:
        return self.num_key_value_heads

    @property
    def num_queries_per_kv(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def max_layers(self) -> int:
        return self.num_layers

    @property
    def total_params(self) -> int:
        h = self.hidden_size
        inter = self.intermediate_size
        n_heads = self.num_attention_heads
        n_kv = self.num_key_value_heads
        head_dim = self.head_dim
        embed = self.vocab_size * h
        qkv = (n_heads * head_dim + 2 * n_kv * head_dim) * h
        o_proj = n_heads * head_dim * h
        ffn = 3 * inter * h
        norms = 4 * h
        per_layer = qkv + o_proj + ffn + norms
        return embed + self.num_layers * per_layer + h + embed

    def weight_bytes_ternary(self) -> int:
        return self.total_params // 4

    def weight_bytes_fp16(self) -> int:
        return self.total_params * 2

    def weight_bytes_fp32(self) -> int:
        return self.total_params * 4


@dataclass
class WeightMapping:
    layer_prefix: str = "model.layers"
    embed_tokens: str = "model.embed_tokens.weight"
    final_norm: str = "model.norm.weight"
    lm_head: str = "lm_head.weight"
    q_proj: str = "self_attn.q_proj.weight"
    k_proj: str = "self_attn.k_proj.weight"
    v_proj: str = "self_attn.v_proj.weight"
    o_proj: str = "self_attn.o_proj.weight"
    gate_proj: str = "mlp.gate_proj.weight"
    up_proj: str = "mlp.up_proj.weight"
    down_proj: str = "mlp.down_proj.weight"
    input_norm: str = "input_layernorm.weight"
    post_attn_norm: str = "post_attention_layernorm.weight"


LLAMA_WEIGHT_MAP = WeightMapping()

LLAMA_32_3B = ModelConfig(
    name="Llama-3.2-3B",
    hidden_size=3072,
    intermediate_size=8192,
    num_layers=28,
    num_attention_heads=24,
    num_key_value_heads=8,
    vocab_size=128256,
    max_position_embeddings=131072,
    rms_norm_eps=1e-5,
    rope_theta=500000.0,
    bos_token_id=128000,
    eos_token_id=128001,
)

LLAMA_3_8B = ModelConfig(
    name="Llama-3-8B",
    hidden_size=4096,
    intermediate_size=14336,
    num_layers=32,
    num_attention_heads=32,
    num_key_value_heads=8,
    vocab_size=128256,
    max_position_embeddings=8192,
    rms_norm_eps=1e-5,
    rope_theta=500000.0,
    bos_token_id=128000,
    eos_token_id=128001,
)

LLAMA_2_7B = ModelConfig(
    name="Llama-2-7B",
    hidden_size=4096,
    intermediate_size=11008,
    num_layers=32,
    num_attention_heads=32,
    num_key_value_heads=32,
    vocab_size=32000,
    max_position_embeddings=4096,
    rms_norm_eps=1e-5,
    rope_theta=10000.0,
    bos_token_id=1,
    eos_token_id=2,
)

LLAMA_2_13B = ModelConfig(
    name="Llama-2-13B",
    hidden_size=5120,
    intermediate_size=13824,
    num_layers=40,
    num_attention_heads=40,
    num_key_value_heads=40,
    vocab_size=32000,
    max_position_embeddings=4096,
    rms_norm_eps=1e-5,
    rope_theta=10000.0,
    bos_token_id=1,
    eos_token_id=2,
)

LLAMA_370B = ModelConfig(
    name="Llama-3.1-70B",
    hidden_size=8192,
    intermediate_size=28672,
    num_layers=80,
    num_attention_heads=64,
    num_key_value_heads=8,
    vocab_size=128256,
    max_position_embeddings=131072,
    rms_norm_eps=1e-5,
    rope_theta=500000.0,
    bos_token_id=128000,
    eos_token_id=128001,
)

PRESET_CONFIGS: Dict[str, ModelConfig] = {
    "llama-3.2-3b": LLAMA_32_3B,
    "llama-3-8b": LLAMA_3_8B,
    "llama-2-7b": LLAMA_2_7B,
    "llama-2-13b": LLAMA_2_13B,
    "llama-3.1-70b": LLAMA_370B,
}


def detect_config(model_path: str) -> ModelConfig:
    import json
    from pathlib import Path

    p = Path(model_path)
    config_path = p / "config.json"
    manifest_path = p / "patch_manifest.json"

    if not config_path.exists() and manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        mc = manifest.get("model_config", {})
        source_path = manifest.get("source_model_path", "")
        name = manifest.get("model_name", str(p))
        if source_path and Path(source_path).exists():
            try:
                return detect_config(source_path)
            except Exception:
                pass
        if mc:
            return ModelConfig(
                name=name,
                hidden_size=mc["hidden_size"],
                intermediate_size=mc["intermediate_size"],
                num_layers=mc.get("num_hidden_layers", mc.get("num_layers", 28)),
                num_attention_heads=mc.get("num_attention_heads", 24),
                num_key_value_heads=mc.get("num_key_value_heads", 8),
                vocab_size=mc["vocab_size"],
            )

    if not config_path.exists():
        raise FileNotFoundError(f"No config.json found at {config_path}")

    with open(config_path) as f:
        cfg = json.load(f)

    model_type = cfg.get("model_type", "llama")
    if model_type != "llama":
        raise ValueError(
            f"Only Llama architecture is currently supported, got '{model_type}'. "
            f"Coming soon: Mistral, Phi, Qwen."
        )

    eos_id = cfg.get("eos_token_id", 128001)
    if isinstance(eos_id, list):
        eos_id = eos_id[0]

    return ModelConfig(
        name=cfg.get("_name_or_path", str(p)),
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        num_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg.get("num_key_value_heads", cfg["num_attention_heads"]),
        vocab_size=cfg["vocab_size"],
        max_position_embeddings=cfg.get("max_position_embeddings", 131072),
        rms_norm_eps=cfg.get("rms_norm_eps", 1e-5),
        rope_theta=cfg.get("rope_theta", 500000.0),
        tie_word_embeddings=cfg.get("tie_word_embeddings", False),
        bos_token_id=cfg.get("bos_token_id", 128000),
        eos_token_id=eos_id,
        model_type=model_type,
    )
