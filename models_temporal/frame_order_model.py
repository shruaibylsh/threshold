"""
Frame Order Prediction Model for Temporal Learning

The model learns temporal structure by predicting the original position
of each frame in a shuffled video sequence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from timesformer_encoder import TimeSformerEncoder


class PositionPredictionHead(nn.Module):
    """
    Predicts the original position (0-6) for each frame in the sequence.

    For each frame, we get a 384-dim embedding and predict which position
    it originally came from (7-way classification).
    """

    def __init__(self, in_dim: int = 384, hidden_dim: int = 256, num_positions: int = 7):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, num_positions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) frame embeddings

        Returns:
            logits: (B, T, num_positions) position predictions for each frame
        """
        # x: (B, T, D)
        h = self.fc1(x)  # (B, T, hidden_dim)
        h = self.relu(h)
        h = self.dropout(h)
        logits = self.fc2(h)  # (B, T, num_positions)
        return logits


class FrameOrderModel(nn.Module):
    """
    Frame Order Prediction model using TimeSformer encoder.

    Training procedure:
    1. Take a sequence of 7 frames
    2. Shuffle them randomly (store the permutation)
    3. Pass shuffled sequence through encoder
    4. Predict the original position of each frame
    5. Compute cross-entropy loss for each frame's position prediction
    """

    def __init__(
        self,
        in_ch: int = 1,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        drop: float = 0.1,
        attn_drop: float = 0.0,
        patch: Tuple[int, int] = (8, 8),
        T: int = 7,
        H: int = 32,
        W: int = 64,
        num_positions: int = 7,
    ):
        super().__init__()

        # TimeSformer encoder (from OLD implementation)
        self.encoder = TimeSformerEncoder(
            in_ch=in_ch,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=drop,
            attn_drop=attn_drop,
            patch=patch,
            T=T,
            H=H,
            W=W,
        )

        self.T = T
        self.S = self.encoder.S  # Number of spatial patches per frame

        # Position prediction head
        self.position_head = PositionPredictionHead(
            in_dim=embed_dim, hidden_dim=256, num_positions=num_positions
        )

    def forward(
        self, x: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, C, H, W) shuffled frames
            targets: (B, T) ground truth original positions for each frame

        Returns:
            loss: scalar loss
            logits: (B, T, num_positions) position predictions
        """
        B, T, C, H, W = x.shape

        # Encode shuffled sequence - get tokens before pooling
        # We need to manually run encoder forward to get tokens
        x_flat = x.reshape(B * T, C, H, W)  # [B*T, C, H, W]
        tokens, (Hp, Wp) = self.encoder.patch_embed(x_flat)  # [B*T, S, D]
        S = tokens.shape[1]
        tokens = tokens.reshape(B, T, S, -1)  # [B, T, S, D]

        # Add positional embeddings
        tokens = tokens + self.encoder.temb[:, :T, :].unsqueeze(2) + self.encoder.semb[:, :S, :].unsqueeze(1)
        x_tokens = tokens.reshape(B, T * S, -1)  # [B, T*S, D]

        # Run through transformer blocks
        for blk in self.encoder.blocks:
            x_tokens = blk(x_tokens, T=T, S=S)

        x_tokens = self.encoder.norm(x_tokens)  # [B, T*S, D]

        # Reshape to separate temporal and spatial dimensions
        # (B, T*S, D) -> (B, T, S, D)
        h = x_tokens.view(B, T, S, -1)

        # Average pool over spatial patches for each frame
        # (B, T, S, D) -> (B, T, D)
        h_frames = h.mean(dim=2)

        # Predict position for each frame
        # (B, T, D) -> (B, T, num_positions)
        logits = self.position_head(h_frames)

        # Compute cross-entropy loss
        # logits: (B, T, num_positions), targets: (B, T)
        # Flatten for cross-entropy
        logits_flat = logits.view(B * self.T, -1)  # (B*T, num_positions)
        targets_flat = targets.view(B * self.T)  # (B*T,)

        loss = F.cross_entropy(logits_flat, targets_flat)

        return loss, logits

    def get_frame_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract frame-level embeddings (for evaluation/visualization).

        Args:
            x: (B, T, C, H, W) frames in ORIGINAL order

        Returns:
            embeddings: (B, T, D) frame embeddings
        """
        B, T, C, H, W = x.shape

        # Encode - get tokens before pooling
        x_flat = x.reshape(B * T, C, H, W)  # [B*T, C, H, W]
        tokens, (Hp, Wp) = self.encoder.patch_embed(x_flat)  # [B*T, S, D]
        S = tokens.shape[1]
        tokens = tokens.reshape(B, T, S, -1)  # [B, T, S, D]

        # Add positional embeddings
        tokens = tokens + self.encoder.temb[:, :T, :].unsqueeze(2) + self.encoder.semb[:, :S, :].unsqueeze(1)
        x_tokens = tokens.reshape(B, T * S, -1)  # [B, T*S, D]

        # Run through transformer blocks
        for blk in self.encoder.blocks:
            x_tokens = blk(x_tokens, T=T, S=S)

        x_tokens = self.encoder.norm(x_tokens)  # [B, T*S, D]

        # Reshape and pool
        h = x_tokens.view(B, T, S, -1)  # (B, T, S, D)
        h_frames = h.mean(dim=2)  # (B, T, D)

        return h_frames


def create_frame_order_model(config) -> FrameOrderModel:
    """Create frame order model from config."""
    model = FrameOrderModel(
        in_ch=config.num_channels,
        embed_dim=config.embed_dim,
        depth=config.num_hidden_layers,
        num_heads=config.num_attention_heads,
        mlp_ratio=config.mlp_ratio,
        drop=config.hidden_dropout_prob,
        attn_drop=config.attention_probs_dropout_prob,
        patch=(config.patch_size, config.patch_size),
        T=config.num_frames,
        H=config.image_height,
        W=config.image_width,
        num_positions=config.position_classes,
    )
    return model