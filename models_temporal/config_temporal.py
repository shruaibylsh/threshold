"""
Configuration for Frame Order Prediction (Temporal Learning)
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class TemporalConfig:
    """
    Configuration for frame order prediction self-supervised learning.

    The model learns temporal structure by predicting the original position
    of each frame in a shuffled sequence.
    """

    # Data
    num_frames: int = 7  # Number of frames per sequence
    num_channels: int = 1  # Grayscale
    image_height: int = 32
    image_width: int = 64

    # TimeSformer Encoder (matching OLD implementation)
    patch_size: int = 8  # 8x8 patches (from OLD)
    embed_dim: int = 384
    num_hidden_layers: int = 6  # depth
    num_attention_heads: int = 6
    mlp_ratio: float = 4.0
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.0

    # Training hyperparameters (matching OLD SimCLR)
    learning_rate: float = 3e-4
    weight_decay: float = 5e-4
    batch_size: int = 128
    max_epochs: int = 200
    warmup_epochs: int = 10

    # Frame order prediction specifics
    shuffle_strategy: str = "full"  # "full" = shuffle all 7 frames
    position_classes: int = 7  # Each frame predicts its position (0-6)


def get_temporal_config() -> TemporalConfig:
    """Get default temporal config."""
    return TemporalConfig()
