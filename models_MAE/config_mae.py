"""
Configuration for TimeSformer MAE (Masked Autoencoder) pretraining.
"""
from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class MAEConfig:
    # Data dimensions
    num_frames: int = 7
    image_height: int = 32
    image_width: int = 64
    num_channels: int = 1  # Grayscale
    patch_size: int = 8     # 8x8 spatial patches

    # Encoder architecture (TimeSformer)
    encoder_embed_dim: int = 384
    encoder_depth: int = 6
    encoder_num_heads: int = 6
    encoder_mlp_ratio: float = 4.0

    # Decoder architecture (lightweight)
    decoder_embed_dim: int = 192
    decoder_depth: int = 4
    decoder_num_heads: int = 3
    decoder_mlp_ratio: float = 4.0

    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.0

    # Masking strategy - ASYMMETRIC for threshold learning
    # Frames 0-1 (before): mask heavily (90%)
    # Frames 2-4 (at threshold): mask moderately (50%) - provide context
    # Frames 5-6 (after): mask heavily (90%)
    frame_mask_ratios: Tuple[float, ...] = (0.90, 0.90, 0.50, 0.50, 0.50, 0.90, 0.90)

    # Loss
    norm_pix_loss: bool = True  # Normalize pixel targets

    # Training hyperparameters
    learning_rate: float = 1.5e-4
    weight_decay: float = 0.05
    batch_size: int = 64
    warmup_epochs: int = 10
    max_epochs: int = 200

    # Data
    num_typologies: int = 8  # t1 through t8

    def __post_init__(self):
        """Calculate derived properties."""
        self.num_patches_h = self.image_height // self.patch_size
        self.num_patches_w = self.image_width // self.patch_size
        self.num_patches_per_frame = self.num_patches_h * self.num_patches_w
        self.total_patches = self.num_frames * self.num_patches_per_frame

        # Calculate effective mask ratio
        total_mask_ratio = sum(self.frame_mask_ratios) / len(self.frame_mask_ratios)

        print(f"MAE Configuration:")
        print(f"  Image: {self.image_height}×{self.image_width}, {self.num_channels} channel(s)")
        print(f"  Patch size: {self.patch_size}×{self.patch_size}")
        print(f"  Patches per frame: {self.num_patches_per_frame} ({self.num_patches_h}×{self.num_patches_w})")
        print(f"  Total frames: {self.num_frames}")
        print(f"  Total patches: {self.total_patches}")
        print(f"  Masking strategy: ASYMMETRIC")
        print(f"    Frames 0-1 (before): {self.frame_mask_ratios[0]:.0%} masked")
        print(f"    Frames 2-4 (at threshold): {self.frame_mask_ratios[2]:.0%} masked")
        print(f"    Frames 5-6 (after): {self.frame_mask_ratios[5]:.0%} masked")
        print(f"  Average mask ratio: {total_mask_ratio:.1%}")
        print(f"  Encoder: {self.encoder_depth} layers, dim={self.encoder_embed_dim}")
        print(f"  Decoder: {self.decoder_depth} layers, dim={self.decoder_embed_dim}")

def get_mae_config():
    """Get default MAE configuration."""
    return MAEConfig()

def get_small_mae_config():
    """Get smaller MAE config for faster experimentation."""
    return MAEConfig(
        encoder_embed_dim=256,
        encoder_depth=4,
        encoder_num_heads=4,
        decoder_embed_dim=128,
        decoder_depth=2,
        decoder_num_heads=4,
    )
