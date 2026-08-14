"""
Abstract model utilities interface.

This module defines the abstract interface for model utilities that calculate
embeddings and generate captions.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from PIL import Image


def fuse_embeddings(img_emb: np.ndarray, txt_emb: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Fuse two L2-normalized embeddings using weighted sum and re-normalize.
    
    Args:
        img_emb: Image embedding vector (shape: (D,))
        txt_emb: Text embedding vector (shape: (D,))
        alpha: Weight for image embedding (0.0 = text only, 1.0 = image only)
    
    Returns:
        Fused embedding vector (shape: (D,))
    """
    if img_emb.shape != txt_emb.shape:
        raise ValueError("img_emb and txt_emb must have the same dimension")
    
    # Weighted sum
    combined = alpha * img_emb + (1.0 - alpha) * txt_emb
    
    # Re-normalize
    norm = np.linalg.norm(combined)
    if norm == 0.0:
        return txt_emb.copy()
    return (combined / norm).astype(np.float32)


def clip_logits_per_image(text_embedding, image_vectors, logit_scale) -> np.ndarray:
    """
    Vectorized CLIP logits_per_image: cosine(image, text) * logit_scale.

    ``image_vectors`` is (N, D); ``text_embedding`` is (D,).
    Rows with missing or zero vectors score 0.0.
    """
    text = np.asarray(text_embedding, dtype=np.float32).reshape(-1)
    images = np.asarray(image_vectors, dtype=np.float32)
    if images.ndim == 1:
        images = images.reshape(1, -1)
    n = images.shape[0]
    scores = np.zeros(n, dtype=np.float32)
    text_norm = np.linalg.norm(text)
    if text_norm == 0.0:
        return scores
    text = text / text_norm
    img_norms = np.linalg.norm(images, axis=1)
    valid = np.isfinite(img_norms) & (img_norms > 0)
    if not np.any(valid):
        return scores
    images_n = images[valid] / img_norms[valid, None]
    scores[valid] = (images_n @ text) * float(logit_scale)
    return scores


class ModelUtils(ABC):
    """
    Abstract interface for model utilities.
    
    This class defines the interface for calculating embeddings and generating captions.
    Concrete implementations should provide specific model backends (e.g., Triton, OpenAI, etc.).
    """
    
    @abstractmethod
    def calculate_embedding(
        self,
        text: str,
        image: Optional[Image.Image] = None,
        model_name: str = "clip"
    ) -> Optional[np.ndarray]:
        """
        Calculate embedding for text and/or image.
        
        Args:
            text: Text to embed
            image: Optional PIL Image to embed
            model_name: Name of the model to use (e.g., "clip", "colbert", "align")
        
        Returns:
            Embedding vector (numpy array) or None on error
        """
        pass
    
    @abstractmethod
    def generate_caption(
        self,
        image: Image.Image,
        prompt: str,
        model_name: str = "gemma3"
    ) -> Optional[str]:
        """
        Generate a caption for an image.
        
        Args:
            image: PIL Image to caption
            prompt: Prompt to use for the model
            model_name: Name of the model to use (e.g., "gemma3", "qwen2_5")
        Returns:
            Generated caption string or None on error
        """
        pass

