"""Local language model for fine-tuned LLMs.

This module provides a LocalLanguageModel class that wraps HuggingFace
transformers models for use with the simulation framework.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from concordia.language_model import language_model


class LocalLanguageModel(language_model.LanguageModel):
    """Local language model for fine-tuned LLMs using HuggingFace transformers.

    This model supports loading local model weights and running inference
    without external API calls, enabling the use of custom fine-tuned models.
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str | None = None,
        device: str = "auto",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> None:
        """Initialize the local language model.

        Args:
            model_path: Path to the model weights or HuggingFace model ID.
            tokenizer_path: Optional path to tokenizer. Defaults to model_path.
            device: Device to run on ('auto', 'cuda', 'cpu', or specific device).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional arguments passed to the model.
        """
        self._model_path = model_path
        self._tokenizer_path = tokenizer_path or model_path
        self._device = device
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._extra_kwargs = kwargs
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        """Lazy load the model and tokenizer on first use."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Local model support requires transformers and torch. "
                "Install with: pip install transformers torch"
            ) from e

        # Resolve device
        if self._device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self._device

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._tokenizer_path,
            trust_remote_code=True,
        )

        # Load model
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            device_map=device if device != "cpu" else None,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            trust_remote_code=True,
        )

        if device == "cpu":
            self._model = self._model.to(device)

        self._model.eval()

    def sample_text(
        self,
        prompt: str,
        *,
        max_tokens: int = 5000,
        terminators: tuple = (),
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
        timeout: float = 60,
        seed: int | None = None,
    ) -> str:
        """Generate text from the local model.

        Args:
            prompt: The prompt text.
            max_tokens: Maximum tokens to generate.
            terminators: Terminator strings (used for stop sequences).
            temperature: Sampling temperature. Uses instance default if 1.0.
            top_p: Top-p sampling parameter.
            top_k: Top-k sampling parameter.
            timeout: Timeout in seconds (not enforced for local models).
            seed: Random seed for reproducibility.

        Returns:
            Generated text string.
        """
        self._ensure_loaded()

        import torch

        # Use instance temperature if not overridden
        effective_temp = self._temperature if temperature == 1.0 else temperature
        effective_max_tokens = min(max_tokens, self._max_tokens)

        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)

        # Tokenize input
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=effective_max_tokens,
                temperature=effective_temp if effective_temp > 0 else 1.0,
                do_sample=effective_temp > 0,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode output (exclude input tokens)
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Apply terminators if provided
        for terminator in terminators:
            if terminator in response:
                response = response.split(terminator)[0]

        return response.strip()

    def sample_choice(
        self,
        prompt: str,
        responses: Sequence[str],
        *,
        seed: int | None = None,
    ) -> tuple[int, str, Mapping[str, Any]]:
        """Choose from a set of responses.

        Args:
            prompt: The prompt text.
            responses: Available response options.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (index, chosen response, metadata).
        """
        self._ensure_loaded()

        import torch

        if seed is not None:
            torch.manual_seed(seed)

        # Calculate log probabilities for each response
        log_probs = []
        for response in responses:
            full_text = prompt + response
            inputs = self._tokenizer(full_text, return_tensors="pt")
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs, labels=inputs["input_ids"])
                log_probs.append(-outputs.loss.item())

        # Select the response with highest log probability
        best_idx = max(range(len(responses)), key=lambda i: log_probs[i])

        return best_idx, responses[best_idx], {"log_probs": log_probs}
