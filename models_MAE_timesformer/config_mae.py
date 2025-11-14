"""
Configuration for TimeSformer MAE (Masked Autoencoder) pretraining.

This uses the Video MAE approach: TimeSformer encoder + uniform tube masking.
"""
from dataclasses import dataclass, field

@dataclass
class MAEConfig:
    # Data dimensions
    num_frames: int = 7
    image_height: int = 32
    image_width: int = 64
    num_channels: int = 1  # Grayscale
    patch_size: int = 8     # 8x8 spatial patches

    # Encoder architecture (TimeSformer with divided space-time attention)
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

    # Masking strategy - UNIFORM TUBE MASKING (Video MAE standard)
    # All frames use the same mask ratio (enables TimeSformer compatibility)
    mask_ratio: float = 0.9  # 75% masked (high ratio like Video MAE)

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

        # Calculate visible patches after masking
        patches_kept_per_frame = int(self.num_patches_per_frame * (1 - self.mask_ratio))
        total_visible = self.num_frames * patches_kept_per_frame

        print(f"MAE Configuration:")
        print(f"  Image: {self.image_height}×{self.image_width}, {self.num_channels} channel(s)")
        print(f"  Patch size: {self.patch_size}×{self.patch_size}")
        print(f"  Patches per frame: {self.num_patches_per_frame} ({self.num_patches_h}×{self.num_patches_w})")
        print(f"  Total frames: {self.num_frames}")
        print(f"  Total patches: {self.total_patches}")
        print(f"  Masking strategy: UNIFORM TUBE MASKING (Video MAE)")
        print(f"    Mask ratio: {self.mask_ratio:.0%} (all frames)")
        print(f"    Visible patches per frame: {patches_kept_per_frame}")
        print(f"    Total visible patches: {total_visible} / {self.total_patches}")
        print(f"  Encoder: TimeSformer (divided space-time)")
        print(f"    {self.encoder_depth} layers, dim={self.encoder_embed_dim}")
        print(f"  Decoder: Standard Transformer")
        print(f"    {self.decoder_depth} layers, dim={self.decoder_embed_dim}")

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
        mask_ratio=0.75,
    )