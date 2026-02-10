"""
NRP-based adapters for benchmarking framework.

This module provides NRP (Envoy AI Gateway) adapters:
- NRPModelUtils: Implementation of ModelUtils
- NRPModelProvider: ModelProvider implementation using NRPModelUtils

See: https://nrp.ai/documentation/userdocs/ai/llm-managed/#available-models
"""

import base64
import logging
import os
from enum import Enum
from io import BytesIO
from typing import Optional, Union

from PIL import Image

# Check for optional dependency
try:
    from openai import OpenAI
    _NRP_AVAILABLE = True
except ImportError:
    OpenAI = None
    _NRP_AVAILABLE = False

from ...framework.interfaces import ModelProvider
from ...framework.model_utils import ModelUtils

class CaptionModelSelector(str, Enum):
    """Supported NRP chat/multimodal models (per NRP docs; exclude deprecated)."""

    QWEN3 = "qwen3"
    GPT_OSS = "gpt-oss"
    KIMI = "kimi"
    GLM_4_7 = "glm-4.7"
    MINIMAX_M2 = "minimax-m2"
    GLM_V = "glm-v"
    GEMMA3 = "gemma3"

    @classmethod
    def from_str(cls, s: str) -> "CaptionModelSelector":
        """Return the enum member whose value is s, or raise ValueError."""
        for m in cls:
            if m.value == s:
                return m
        raise ValueError(f"Unsupported NRP LLM Model: {s}")

def _model_str(model: Union[CaptionModelSelector, str]) -> str:
    """Normalize NRPModel or str to the API model string."""
    if isinstance(model, CaptionModelSelector):
        return model.value
    return CaptionModelSelector.from_str(model).value

def _check_nrp_available():
    """Check if OpenAI dependency is installed for NRP."""
    if not _NRP_AVAILABLE:
        raise ImportError(
            "NRP adapters require 'openai'. "
            "Install it with: pip install imsearch_eval[nrp]"
        )

class NRPModelUtils(ModelUtils):
    """
    NRP-based implementation of ModelUtils (caption generation only).

    Uses the NRP Envoy AI Gateway via an OpenAI-compatible client.
    Embedding is not supported; calculate_embedding raises NotImplementedError.
    """

    def __init__(self, client: "OpenAI"):
        """
        Initialize NRP model utils.

        Args:
            client: OpenAI client configured for NRP Envoy (base_url, api_key).
        """
        _check_nrp_available()
        if client is None:
            raise ValueError("client cannot be None")
        self.client = client

    def calculate_embedding(
        self,
        text: str,
        image: Optional[Image.Image] = None,
        model_name: str = "clip"
    ):
        """Embedding is not yet supported."""
        raise NotImplementedError("Embedding is not yet supported with NRP adapter.")

    def generate_caption(
        self,
        image: Image.Image,
        prompt: str,
        model_name: Union[CaptionModelSelector, str] = CaptionModelSelector.GEMMA3
    ) -> Optional[str]:
        """
        Generate a caption for an image using NRP Envoy AI Gateway.

        Args:
            image: PIL Image to caption
            prompt: Prompt to use for the model
            model_name: NRP model (CaptionModelSelector enum or string, e.g. "gemma3", "qwen3", "glm-v").

        Returns:
            Generated caption string or None on error
        """
        try:
            model_str = _model_str(model_name)
        except ValueError:
            raise ValueError(f"Unsupported NRP LLM Model: {model_name}")

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        try:
            # Do not pass max_tokens per NRP docs (https://nrp.ai/documentation/userdocs/ai/llm-managed/)
            response = self.client.chat.completions.create(
                model=model_str,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}"
                                },
                            },
                        ],
                    }
                ],
            )

            answer_str = response.choices[0].message.content
            logging.info(f"[NRP {model_str}] Final Generated Description: {answer_str}")
            return answer_str

        except Exception as e:
            logging.error(f"[NRP] Error during {model_str} inference via OpenAI client: {str(e)}")
            return None


class NRPModelProvider(ModelProvider):
    """NRP model provider (caption generation only) using NRPModelUtils."""

    def __init__(
        self,
        api_key: str = os.environ.get("NRP_API_KEY"),
        base_url: str = "https://ellm.nrp-nautilus.io/v1",
        **client_kwargs,
    ):
        """
        Initialize NRP model provider.

        Creates an OpenAI-compatible client for the NRP Envoy AI Gateway.

        Args:
            api_key: NRP API token (defaults to environment variable "NRP_API_KEY").
            base_url: Envoy gateway URL. Defaults to the NRP-managed LLM endpoint.
            **client_kwargs: Optional extra arguments passed to the OpenAI client.
        """
        _check_nrp_available()
        self.client = OpenAI(api_key=api_key, base_url=base_url, **client_kwargs)
        self.model_utils = NRPModelUtils(self.client)

    def get_embedding(
        self,
        text: str,
        image: Optional[Image.Image] = None,
        model_name: str = "default"
    ):
        """Embedding is not yet supported."""
        raise NotImplementedError("Embedding is not yet supported with NRP adapter.")

    def generate_caption(
        self,
        image: Image.Image,
        prompt: str,
        model_name: Union[CaptionModelSelector, str] = CaptionModelSelector.GEMMA3
    ) -> str:
        """
        Generate a caption for an image.

        Args:
            image: PIL Image to caption
            prompt: Prompt to use for the model
            model_name: NRP model (CaptionModelSelector enum or string, e.g. "gemma3", "qwen3", "glm-v")

        Returns:
            Generated caption string (empty string on error)
        """
        result = self.model_utils.generate_caption(image, prompt, model_name)
        return result if result else ""
