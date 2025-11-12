"""
TimeSformer with Temporal Order Prediction for self-supervised learning.

Based on "Shuffle and Learn: Unsupervised Learning using Temporal Order Verification"
(Misra & van der Heijden, ECCV 2016)

Task: Given two frames from a sequence, predict if they are in correct temporal order.
"""
import torch
import torch.nn as nn
from timesformer_mae import TimeSformerEncoder, PatchEmbed


class TimeSformerTemporalOrder(nn.Module):
    """
    TimeSformer with Temporal Order Prediction.

    Takes 2 frames as input and predicts if they are in correct order.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Use the same encoder architecture as MAE
        # But adapt for 2-frame input instead of 7
        self.encoder = TimeSformerEncoder(config)

        # Classifier head for binary classification
        # Input: pooled sequence embedding
        # Output: binary prediction (0=wrong order, 1=correct order)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, 1)  # Binary classification
        )

    def forward(self, frame_pair):
        """
        Forward pass for temporal order prediction.

        Args:
            frame_pair: (B, 2, C, H, W) - two frames stacked

        Returns:
            logits: (B, 1) - raw logits for binary classification
        """
        B = frame_pair.shape[0]

        # Encode the 2-frame sequence
        # The encoder will apply divided space-time attention
        x = self.encoder.patch_embed(frame_pair)  # (B, 2*P, D) where P=128
        x = x + self.encoder.pos_embed[:, :x.shape[1], :]  # Add positional embeddings
        x = self.encoder.pos_drop(x)

        # Apply transformer blocks
        num_frames = 2  # We only have 2 frames
        num_patches_per_frame = self.encoder.patch_embed.num_patches

        for block in self.encoder.blocks:
            x = block(x, num_frames, num_patches_per_frame)

        x = self.encoder.norm(x)

        # Global average pooling across all patches
        # (B, 2*P, D) -> (B, D)
        pooled = x.mean(dim=1)

        # Binary classification
        logits = self.classifier(pooled)  # (B, 1)

        return logits


def create_temporal_order_model(config):
    """Factory function to create temporal order model."""
    return TimeSformerTemporalOrder(config)
