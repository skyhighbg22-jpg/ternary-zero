from __future__ import annotations

from typing import List, Optional


class Tokenizer:
    def __init__(self, tokenizer_obj):
        self._tok = tokenizer_obj
        self._bos_id: Optional[int] = None
        self._eos_id: Optional[int] = None

    @classmethod
    def from_pretrained(cls, model_path: str) -> "Tokenizer":
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            return cls(tok)
        except ImportError:
            pass

        try:
            from tokenizers import Tokenizer as HFTokenizer
            tok = HFTokenizer.from_file(f"{model_path}/tokenizer.json")
            wrapper = _TokenizerFastWrapper(tok)
            return cls(wrapper)
        except (ImportError, FileNotFoundError):
            pass

        raise ImportError(
            f"Could not load tokenizer from '{model_path}'. "
            f"Install 'transformers' or 'tokenizers': pip install transformers"
        )

    @property
    def bos_token_id(self) -> Optional[int]:
        if self._bos_id is not None:
            return self._bos_id
        if hasattr(self._tok, "bos_token_id"):
            self._bos_id = self._tok.bos_token_id
            return self._bos_id
        return None

    @property
    def eos_token_id(self) -> Optional[int]:
        if self._eos_id is not None:
            return self._eos_id
        if hasattr(self._tok, "eos_token_id"):
            eid = self._tok.eos_token_id
            if isinstance(eid, list):
                eid = eid[0]
            self._eos_id = eid
            return self._eos_id
        return None

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        if hasattr(self._tok, "encode"):
            result = self._tok.encode(text, add_special_tokens=add_special_tokens)
            if isinstance(result, dict):
                return result.get("input_ids", result.get("ids", []))
            return list(result)
        raise RuntimeError("Tokenizer does not support encode()")

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        if hasattr(self._tok, "decode"):
            return self._tok.decode(token_ids, skip_special_tokens=skip_special_tokens)
        raise RuntimeError("Tokenizer does not support decode()")


class _TokenizerFastWrapper:
    def __init__(self, tok):
        self._tok = tok
        self.bos_token_id = None
        self.eos_token_id = None

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        encoding = self._tok.encode(text, add_special_tokens=add_special_tokens)
        return encoding.ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(token_ids, skip_special_tokens=skip_special_tokens)
