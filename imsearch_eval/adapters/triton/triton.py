"""
Triton-based adapters for benchmarking framework.

This module provides all Triton-related adapters:
- TritonModelUtils: Implementation of ModelUtils interface using Triton
- TritonModelProvider: ModelProvider implementation using TritonModelUtils
"""

import logging
from typing import Optional, Tuple

# Check for optional dependencies
try:
    import tritonclient.grpc as TritonClient
    _TRITON_AVAILABLE = True
except ImportError:
    TritonClient = None
    _TRITON_AVAILABLE = False

import numpy as np
from PIL import Image

from ...framework.interfaces import ModelProvider
from ...framework.model_utils import ModelUtils, clip_logits_per_image, fuse_embeddings


def _check_triton_available():
    """Check if Triton dependencies are installed."""
    if not _TRITON_AVAILABLE:
        raise ImportError(
            "Triton adapters require 'tritonclient[grpc]'. "
            "Install it with: pip install imsearch_eval[triton]"
        )


class TritonModelUtils(ModelUtils):
    """
    Triton-based implementation of ModelUtils.
    
    Provides embedding and caption generation using models served via Triton Inference Server.
    """
    
    def __init__(self, triton_client):
        """
        Initialize Triton model utils.
        
        Args:
            triton_client: Triton inference server client
        """
        _check_triton_available()
        if triton_client is None:
            raise ValueError("triton_client cannot be None")
        self.triton_client = triton_client
    
    def calculate_embedding(
        self,
        text: str,
        image: Optional[Image.Image] = None,
        model_name: str = "clip"
    ) -> Optional[np.ndarray]:
        """
        Calculate embedding for text and/or image using Triton.
        
        Args:
            text: Text to embed
            image: Optional PIL Image to embed
            model_name: Name of the model to use ("clip", "colbert", "align")
        
        Returns:
            Embedding vector (numpy array) or None on error
        """
        if model_name == "clip":
            return self.get_clip_embeddings(text, image)
        elif model_name == "colbert":
            return self.get_colbert_embedding(text)
        elif model_name == "align":
            return self.get_allign_embeddings(text, image)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    
    def generate_caption(
        self,
        image: Image.Image,
        prompt: str,
        model_name: str = "gemma3",
        enable_thinking: bool = True
    ) -> Optional[str]:
        """
        Generate a caption for an image using Triton.
        
        Args:
            image: PIL Image to caption
            prompt: Prompt to use for the model
            model_name: Name of the model to use ("gemma3", "qwen2_5")
            enable_thinking: Whether to enable thinking (default: True). Not all models support thinking.
        Returns:
            Generated caption string or None on error
        """
        if model_name == "gemma3":
            return self.gemma3_run_model(image, prompt, enable_thinking)
        elif model_name == "qwen2_5":
            return self.qwen2_5_run_model(image, prompt, enable_thinking)
        else:
            raise ValueError(f"Unknown caption model name: {model_name}")
    
    @staticmethod
    def _clip_infer_inputs(text: str, image: Optional[Image.Image] = None):
        """Build CLIP InferInputs with a leading batch dim of 1 (max_batch_size > 0)."""
        text_np = np.array([[text.encode("utf-8")]], dtype=object)
        if image is not None:
            image_np = np.expand_dims(np.asarray(image, dtype=np.float32), 0)
        else:
            image_np = np.zeros((1, 1, 1, 3), dtype=np.float32)
        inputs = [
            TritonClient.InferInput("text", list(text_np.shape), "BYTES"),
            TritonClient.InferInput("image", list(image_np.shape), "FP32"),
        ]
        inputs[0].set_data_from_numpy(text_np)
        inputs[1].set_data_from_numpy(image_np)
        return inputs

    def _infer_clip(
        self,
        text: str,
        image: Optional[Image.Image] = None,
        request_logit_scale: bool = False,
    ):
        """
        Run Triton CLIP and return (text_embedding, image_embedding[, logit_scale]).

        On failure returns (None, None) or (None, None, None) if request_logit_scale.
        """
        inputs = self._clip_infer_inputs(text, image)

        outputs = [
            TritonClient.InferRequestedOutput("text_embedding"),
            TritonClient.InferRequestedOutput("image_embedding"),
        ]
        if request_logit_scale:
            outputs.append(TritonClient.InferRequestedOutput("logit_scale"))

        try:
            results = self.triton_client.infer(
                model_name="clip", inputs=inputs, outputs=outputs
            )
            text_embedding = results.as_numpy("text_embedding")[0]
            image_embedding = results.as_numpy("image_embedding")[0]
            if request_logit_scale:
                logit_scale = float(results.as_numpy("logit_scale").reshape(-1)[0])
                return text_embedding, image_embedding, logit_scale
            return text_embedding, image_embedding
        except Exception as e:
            logging.error(f"Error during CLIP inference: {str(e)}")
            if request_logit_scale:
                return None, None, None
            return None, None

    def get_clip_embeddings(
        self,
        text: str,
        image: Optional[Image.Image] = None,
        alpha: float = 0.7
    ) -> Optional[np.ndarray]:
        """
        Embed text and/or image using CLIP encoder served via Triton Inference Server.
        
        Args:
            text: Text to embed
            image: Optional PIL Image to embed
            alpha: Weight for fusing image and text embeddings (default: 0.7)
        
        Returns:
            Fused embedding vector (numpy array) or None on error
        """
        text_embedding, image_embedding = self._infer_clip(text, image)
        if text_embedding is None:
            return None

        if image is not None:
            return fuse_embeddings(image_embedding, text_embedding, alpha=alpha)
        return text_embedding

    def get_clip_embedding_pair(
        self,
        text: str,
        image: Image.Image,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Return (caption_embedding, image_embedding) from Triton CLIP without fusion.

        Used when indexing separate caption_vector and image_vector fields.
        """
        if image is None:
            raise ValueError("image is required for get_clip_embedding_pair")
        text_embedding, image_embedding = self._infer_clip(text, image)
        if text_embedding is None or image_embedding is None:
            return None, None
        return text_embedding, image_embedding

    def get_clip_query_embedding(
        self, text: str
    ) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Encode query text once via Triton CLIP.

        Returns (text_embedding, logit_scale) where logit_scale is
        exp(model.logit_scale), matching HF CLIPModel logits_per_image.
        On failure returns (None, None).
        """
        text_embedding, _, logit_scale = self._infer_clip(
            text, image=None, request_logit_scale=True
        )
        if text_embedding is None or logit_scale is None:
            return None, None
        return text_embedding, logit_scale

    def clip_image_text_score(
        self,
        query: str,
        image: Image.Image,
        text_embedding=None,
        logit_scale=None,
    ) -> float:
        """
        CLIP similarity between a text query and an image via Triton.

        Matches Hugging Face CLIPModel logits_per_image for a single pair:
        L2-normalize image/text embeddings, then multiply cosine by exp(logit_scale).

        Pass precomputed ``text_embedding`` and ``logit_scale`` to skip the text
        tower (empty text is sent so Triton only runs get_image_features).
        """
        if text_embedding is not None and logit_scale is not None:
            _, image_embedding = self._infer_clip("", image)
            if image_embedding is None:
                return 0.0
        else:
            text_embedding, image_embedding, logit_scale = self._infer_clip(
                query, image, request_logit_scale=True
            )
            if text_embedding is None or image_embedding is None or logit_scale is None:
                return 0.0

        scores = clip_logits_per_image(text_embedding, [image_embedding], logit_scale)
        return float(scores[0])
    
    def get_colbert_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Embed text using ColBERT encoder served via Triton Inference Server.
        
        Args:
            text: Text to embed
        
        Returns:
            Token-level embeddings of shape [num_tokens, 128] or None on error
        """
        # Prepare input
        text_bytes = text.encode("utf-8")
        input_tensor = np.array([text_bytes], dtype="object")  # batch size = 1
        
        # Prepare inputs & outputs for Triton
        inputs = [
            TritonClient.InferInput("text", input_tensor.shape, "BYTES")
        ]
        outputs = [
            TritonClient.InferRequestedOutput("embedding"),
            TritonClient.InferRequestedOutput("token_lengths")
        ]
        
        # Add tensors
        inputs[0].set_data_from_numpy(input_tensor)
        
        # Run inference
        try:
            results = self.triton_client.infer(model_name="colbert", inputs=inputs, outputs=outputs)
            
            # Retrieve and reshape output
            emb_flat = results.as_numpy("embedding")  # shape: (1, max_len * 128)
            token_lengths = results.as_numpy("token_lengths")  # shape: (1,)
            num_tokens = token_lengths[0]
            
            # Reshape and unpad
            emb_3d = emb_flat.reshape(1, -1, 128)
            token_embeddings = emb_3d[0, :num_tokens, :]  # shape: [num_tokens, 128]
        except Exception as e:
            logging.error(f"Error during ColBERT inference: {str(e)}")
            return None
        
        return token_embeddings
    
    def get_allign_embeddings(
        self,
        text: str,
        image: Optional[Image.Image] = None,
        alpha: float = 0.7
    ) -> Optional[np.ndarray]:
        """
        Embed text and/or image using ALIGN encoder served via Triton Inference Server.
        
        Args:
            text: Text to embed
            image: Optional PIL Image to embed
            alpha: Weight for fusing image and text embeddings (default: 0.7)
        
        Returns:
            Fused embedding vector (numpy array) or None on error
        """
        # Prepare inputs
        text_bytes = text.encode("utf-8")
        text_np = np.array([text_bytes], dtype="object")
        
        # Prepare image input
        if image is not None:
            image_np = np.array(image).astype(np.float32)
        else:
            image_np = np.zeros((1, 1, 3), dtype=np.float32)
        
        # Create Triton input objects
        inputs = [
            TritonClient.InferInput("text", [1], "BYTES"),
            TritonClient.InferInput("image", list(image_np.shape), "FP32")
        ]
        
        inputs[0].set_data_from_numpy(text_np)
        inputs[1].set_data_from_numpy(image_np)
        
        outputs = [
            TritonClient.InferRequestedOutput("text_embedding"),
            TritonClient.InferRequestedOutput("image_embedding")
        ]
        
        # Run inference
        try:
            results = self.triton_client.infer(model_name="align", inputs=inputs, outputs=outputs)
            text_embedding = results.as_numpy("text_embedding")[0]
            image_embedding = results.as_numpy("image_embedding")[0]
        except Exception as e:
            logging.error(f"Error during ALIGN inference: {str(e)}")
            return None
        
        # Fuse embeddings
        if image is not None:
            embedding = fuse_embeddings(image_embedding, text_embedding, alpha=alpha)
        else:
            embedding = text_embedding
        
        return embedding
    
    def gemma3_run_model(self, image: Image.Image, prompt: str, enable_thinking: bool = True) -> Optional[str]:
        """
        Generate a caption for an image using Gemma3 model served via Triton.
        
        Args:
            image: PIL Image to caption
            prompt: Prompt to use for the model
            enable_thinking: Whether to enable thinking (default: True). Not all models support thinking.
        Returns:
            Generated caption string or None on error
        """
        # Leading batch dim of 1 required when Triton max_batch_size > 0.
        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)
        prompt_np = np.array([[prompt.encode("utf-8")]], dtype=object)

        inputs = [
            TritonClient.InferInput("image", list(image_np.shape), "UINT8"),
            TritonClient.InferInput("prompt", list(prompt_np.shape), "BYTES"),
        ]
        outputs = [
            TritonClient.InferRequestedOutput("answer")
        ]

        inputs[0].set_data_from_numpy(image_np)
        inputs[1].set_data_from_numpy(prompt_np)

        try:
            response = self.triton_client.infer(model_name="gemma3", inputs=inputs, outputs=outputs)

            answer = response.as_numpy("answer").reshape(-1)[0]
            answer_str = (
                answer.decode("utf-8") if isinstance(answer, (bytes, np.bytes_)) else str(answer)
            )

            logging.info(f'[GEMMA3] Final Generated Description: {answer_str}')
            return answer_str
        except Exception as e:
            logging.error(f"[GEMMA3] Error during Gemma3 inference: {str(e)}")
            return None
    
    def qwen2_5_run_model(self, image: Image.Image, prompt: str, enable_thinking: bool = True) -> Optional[str]:
        """
        Generate a caption for an image using Qwen2.5-VL model served via Triton.
        
        Args:
            image: PIL Image to caption
            prompt: Prompt to use for the model
            enable_thinking: Whether to enable thinking (default: True). Not all models support thinking.
        Returns:
            Generated caption string or None on error
        """
        # Leading batch dim of 1 required when Triton max_batch_size > 0.
        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)
        prompt_np = np.array([[prompt.encode("utf-8")]], dtype=object)

        inputs = [
            TritonClient.InferInput("image", list(image_np.shape), "UINT8"),
            TritonClient.InferInput("prompt", list(prompt_np.shape), "BYTES"),
        ]
        outputs = [
            TritonClient.InferRequestedOutput("answer")
        ]

        inputs[0].set_data_from_numpy(image_np)
        inputs[1].set_data_from_numpy(prompt_np)

        try:
            response = self.triton_client.infer(model_name="qwen2_5_vl", inputs=inputs, outputs=outputs)

            answer = response.as_numpy("answer").reshape(-1)[0]
            answer_str = (
                answer.decode("utf-8") if isinstance(answer, (bytes, np.bytes_)) else str(answer)
            )

            logging.info(f'[QWEN2_5_VL] Final Generated Description: {answer_str}')
            return answer_str
        except Exception as e:
            logging.error(f"[QWEN2_5_VL] Error during Qwen2.5-VL inference: {str(e)}")
            return None

class TritonModelProvider(ModelProvider):
    """Triton model provider using TritonModelUtils."""
    
    def __init__(self, triton_client):
        """
        Initialize Triton model provider.
        
        Args:
            triton_client: Triton inference server client
        """
        _check_triton_available()
        self.triton_client = triton_client
        self.model_utils = TritonModelUtils(triton_client)
    
    def get_embedding(
        self, 
        text: str, 
        image: Optional[Image.Image] = None,
        model_name: str = "clip"
    ):
        """
        Get embedding for text and/or image.
        
        Args:
            text: Text to embed
            image: Optional PIL Image to embed
            model_name: Name of the model to use ("clip", "colbert", "align")
            
        Returns:
            Embedding vector (numpy array)
        """
        return self.model_utils.calculate_embedding(text, image, model_name)
    
    def generate_caption(self, image: Image.Image, prompt: str , model_name: str = "gemma3", enable_thinking: bool = True) -> str:
        """
        Generate a caption for an image.
        
        Args:
            image: PIL Image to caption
            prompt: Prompt to use for the model
            model_name: Name of the model to use ("gemma3", "qwen2_5")
            enable_thinking: Whether to enable thinking (default: True). Not all models support thinking.
        Returns:
            Generated caption string
        """
        result = self.model_utils.generate_caption(image, prompt, model_name, enable_thinking)
        return result if result else ""

