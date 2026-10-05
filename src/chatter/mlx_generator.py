"""MLX backend (Apple Silicon) for the answer ``Generator`` protocol.

Uses mlx-lm with greedy decoding, like the transformers backend. The default
model is a 4-bit quantization of the same Qwen3-4B-Instruct-2507, so answers
can differ slightly in wording. mlx-lm is imported lazily so other platforms
never need it.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from chatter.answer import GeneratorConfig, chat_messages


class MLXGenerator:
    """Greedy chat generation with mlx-lm. ``model``/``tokenizer`` are injectable for tests."""

    def __init__(
        self,
        config: GeneratorConfig = GeneratorConfig(backend="mlx"),
        *,
        model: Any = None,
        tokenizer: Any = None,
    ) -> None:
        try:
            import mlx.core as mx
            from mlx_lm import load, stream_generate
            from mlx_lm.sample_utils import make_sampler
        except ImportError as exc:  # not Apple Silicon, or the mlx extra is missing
            raise ImportError(
                "the mlx backend needs mlx-lm, which is only available on Apple Silicon "
                "(pip install mlx-lm); use --backend transformers elsewhere"
            ) from exc
        self._config = config
        self._mx = mx
        self._stream_generate = stream_generate
        self._sampler = make_sampler(temp=0.0)  # greedy, like the transformers backend
        if model is None or tokenizer is None:
            model, tokenizer = load(config.resolved_model, trust_remote_code=config.trust_remote_code)
        self._model = model
        self._tokenizer = tokenizer
        self._peak_gb: float | None = None

    @property
    def name(self) -> str:
        return self._config.resolved_model

    @property
    def device(self) -> str:
        return "mlx"

    @property
    def backend(self) -> str:
        return "mlx"

    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        return [len(self._tokenizer.encode(text, add_special_tokens=False)) for text in texts]

    def prompt_tokens(self, system: str, user: str) -> int:
        return len(self._prompt_ids(system, user))

    def generate(self, system: str, user: str, *, max_new_tokens: int) -> Iterator[str]:
        self._mx.reset_peak_memory()
        self._peak_gb = None
        for response in self._stream_generate(
            self._model,
            self._tokenizer,
            self._prompt_ids(system, user),
            max_tokens=max_new_tokens,
            sampler=self._sampler,
        ):
            self._peak_gb = response.peak_memory
            if response.text:
                yield response.text

    def memory_stats(self) -> dict[str, Any]:
        if self._peak_gb is None:
            return {}
        return {
            "accelerator_mb": self._peak_gb * 1000,  # mlx-lm reports GB (1e9 bytes)
            "accelerator_metric": "mlx peak memory during the answer",
        }

    def _prompt_ids(self, system: str, user: str) -> list[int]:
        return list(
            self._tokenizer.apply_chat_template(
                chat_messages(system, user), add_generation_prompt=True, tokenize=True
            )
        )
