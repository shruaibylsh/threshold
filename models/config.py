"""
Model configuration for TimeSformer with VideoMAE pretraining.
"""
from dataclasses import dataclass

@dataclass
class TimeSformerMAEConfig:
    # Video dimensions
    num_frames: int = 7
    image_height: int = 32
    image_width: int = 64
    num_channels: int = 3
    # Patch settings
    patch_size: int = 4  # 4x4 spatial patches

    # Encoder architecture (TimeSformer)
    hidden_size: int = 384  # Embedding dimension
    num_hidden_layers: int = 8  # Transformer depth
    num_attention_heads: int = 6
    intermediate_size: int = 1536  # FFN dimension (4x hidden_size)
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1

    # Decoder architecture (lightweight)
    decoder_hidden_size: int = 192
    decoder_num_hidden_layers: int = 4
    decoder_num_attention_heads: int = 3
    decoder_intermediate_size: int = 768

    # MAE training
    mask_ratio: float = 0.9  # NOTE: With asymmetric masking, this is not uniformly applied
    norm_pix_loss: bool = True

    # Asymmetric masking ratios (per frame)
    # Frames 0-1 (before) and 5-6 (after): mask heavily to force learning
    # Frames 2-4 (context): mask moderately to provide reconstruction context
    frame_mask_ratios: tuple = (0.90, 0.90, 0.50, 0.50, 0.50, 0.90, 0.90)

    # Classification
    num_labels: int = 8  # 8 threshold typologies
    def __post_init__(self):
        """Calculate derived properties."""
        # Number of patches
        self.num_patches_height = self.image_height // self.patch_size
        self.num_patches_width = self.image_width // self.patch_size
        self.num_patches_per_frame = self.num_patches_height * self.num_patches_width
        # Total tokens (patches across all frames)
        self.total_patches = self.num_frames * self.num_patches_per_frame
        print(f"Configuration:")
        print(f"  Image size: {self.image_height}x{self.image_width}")
        print(f"  Patch size: {self.patch_size}x{self.patch_size}")
        print(f"  Patches per frame: {self.num_patches_per_frame} ({self.num_patches_height}x{self.num_patches_width})")
        print(f"  Total frames: {self.num_frames}")
        print(f"  Total patches: {self.total_patches}")
        print(f"  Visible patches (~10%): ~{int(self.total_patches * (1 - self.mask_ratio))}")
        print(f"  Masked patches (~90%): ~{int(self.total_patches * self.mask_ratio)}")

def get_mae_config():
    return TimeSformerMAEConfig()

def get_small_config():
    return TimeSformerMAEConfig(
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=4,
        intermediate_size=1024,
        decoder_hidden_size=128,
        decoder_num_hidden_layers=2,
        decoder_num_attention_heads=4,
        decoder_intermediate_size=512,
    )