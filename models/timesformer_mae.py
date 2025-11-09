"""
TimeSformer with VideoMAE Pretraining - HuggingFace Implementation
Uses HuggingFace's TimeSformer model for robust, tested encoder.

Key Fix: Properly handles masking while preserving spatiotemporal structure.
"""

import torch
import torch.nn as nn
from transformers import TimesformerModel, TimesformerConfig


class MAEDecoder(nn.Module):
    """Lightweight decoder for masked autoencoding."""
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Project from encoder to decoder dimension
        self.decoder_embed = nn.Linear(config.hidden_size, config.decoder_hidden_size, bias=True)

        # Positional embeddings for decoder
        num_patches = config.total_patches
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, config.decoder_hidden_size))

        # Transformer decoder blocks
        self.decoder_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.decoder_hidden_size,
                nhead=config.decoder_num_attention_heads,
                dim_feedforward=config.decoder_intermediate_size,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
            )
            for _ in range(config.decoder_num_hidden_layers)
        ])

        self.decoder_norm = nn.LayerNorm(config.decoder_hidden_size)

        # Prediction head: decoder_dim -> patch pixels
        patch_dim = config.patch_size ** 2 * config.num_channels
        self.decoder_pred = nn.Linear(config.decoder_hidden_size, patch_dim, bias=True)

        # Initialize weights
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        Args:
            x: (B, N, encoder_dim) - encoded patches
        Returns:
            pred: (B, N, patch_dim) - reconstructed patches
        """
        # Project to decoder dimension
        x = self.decoder_embed(x)

        # Add positional embeddings
        x = x + self.decoder_pos_embed

        # Apply decoder blocks
        for block in self.decoder_blocks:
            x = block(x)

        x = self.decoder_norm(x)

        # Predict pixel values
        x = self.decoder_pred(x)
        return x


class TimeSformerMAE(nn.Module):
    """
    TimeSformer with VideoMAE pretraining using HuggingFace implementation.

    This version uses HuggingFace's TimeSformer for the encoder, which is:
    - Well-tested and robust
    - Handles divided space-time attention correctly
    - Maintained by the community
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Create HuggingFace TimeSformer config
        # Note: HuggingFace expects square images, but we handle rectangular via proper config
        hf_config = TimesformerConfig(
            image_size=max(config.image_height, config.image_width),  # Use larger dimension
            patch_size=config.patch_size,
            num_channels=config.num_channels,
            num_frames=config.num_frames,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            attention_type="divided_space_time",  # Critical for our use case
        )

        # Initialize encoder (without classification head)
        self.encoder = TimesformerModel(hf_config, add_pooling_layer=False)

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Decoder
        self.decoder = MAEDecoder(config)

        self.mask_ratio = config.mask_ratio

        # Calculate patch counts
        self.num_patches_h = config.num_patches_height
        self.num_patches_w = config.num_patches_width
        self.num_patches_per_frame = self.num_patches_h * self.num_patches_w
        self.total_patches = self.num_patches_per_frame * config.num_frames

    def random_masking(self, x):
        """
        Randomly mask patches using mask tokens.
        CRITICAL: Returns ALL N patches (preserves spatiotemporal structure).

        Args:
            x: (B, N, D) patch embeddings
        Returns:
            x_masked: (B, N, D) - SAME SHAPE, masked patches replaced with mask token
            mask: (B, N) - binary mask (1 = masked, 0 = visible)
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - self.mask_ratio))

        # Random shuffle
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Binary mask: 0 = keep, 1 = mask
        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # Replace masked patches with mask token
        # CRITICAL: We keep all N patches in the sequence
        mask_tokens = self.mask_token.expand(B, N, -1)
        x_masked = x * (1 - mask.unsqueeze(-1)) + mask_tokens * mask.unsqueeze(-1)

        return x_masked, mask

    def patchify(self, imgs):
        """
        Convert images to patches.
        Args:
            imgs: (B, T, C, H, W)
        Returns:
            patches: (B, N, patch_dim) where N = T * H/P * W/P
        """
        B, T, C, H, W = imgs.shape
        p = self.config.patch_size
        h, w = H // p, W // p

        x = imgs.reshape(B, T, C, h, p, w, p)
        x = x.permute(0, 1, 3, 5, 4, 6, 2).reshape(B, T * h * w, p * p * C)
        return x

    def unpatchify(self, x):
        """
        Convert patches back to images.
        Args:
            x: (B, N, patch_dim)
        Returns:
            imgs: (B, T, C, H, W)
        """
        B = x.shape[0]
        p = self.config.patch_size
        T = self.config.num_frames
        C = self.config.num_channels
        h, w = self.num_patches_h, self.num_patches_w

        x = x.reshape(B, T, h, w, p, p, C)
        x = x.permute(0, 1, 6, 2, 4, 3, 5).reshape(B, T, C, h * p, w * p)
        return x

    def forward_loss(self, imgs, pred, mask):
        """
        Compute MSE loss on masked patches only.
        Args:
            imgs: (B, T, C, H, W) original images
            pred: (B, N, patch_dim) predicted patches
            mask: (B, N) binary mask (1 = masked, 0 = visible)
        Returns:
            loss: scalar
        """
        target = self.patchify(imgs)

        if self.config.norm_pix_loss:
            # Normalize each patch
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5

        # MSE loss
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # Mean over patch pixels

        # Compute loss only on masked patches
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, pixel_values):
        """
        Forward pass with masked autoencoding.

        Args:
            pixel_values: (B, T, C, H, W) - input video frames
        Returns:
            loss: reconstruction loss
            pred: predicted patches
            mask: binary mask (1 = masked, 0 = visible)
        """
        B, T, C, H, W = pixel_values.shape

        # HuggingFace TimeSformer expects (B, C, T, H, W)
        pixel_values_hf = pixel_values.permute(0, 2, 1, 3, 4).contiguous()

        # Get patch embeddings before encoder
        # We manually extract embeddings to apply masking before attention
        embeddings = self.encoder.embeddings.patch_embeddings(pixel_values_hf)

        # Flatten spatial dimensions: (B, D, T, H', W') -> (B, T*H'*W', D)
        B_emb, D_emb, T_emb, H_emb, W_emb = embeddings.shape
        embeddings = embeddings.flatten(2).transpose(1, 2)  # (B, N, D)

        # Apply random masking - KEEPS ALL N PATCHES
        embeddings_masked, mask = self.random_masking(embeddings)  # (B, N, D), (B, N)

        # Add positional embeddings (HuggingFace handles this)
        if self.encoder.embeddings.position_embeddings is not None:
            embeddings_masked = embeddings_masked + self.encoder.embeddings.position_embeddings

        # Pass through encoder transformer blocks
        encoder_output = self.encoder.encoder(embeddings_masked)
        latent = encoder_output.last_hidden_state  # (B, N, D)

        # Apply layer norm
        latent = self.encoder.layernorm(latent)

        # Decode to reconstruct all patches
        pred = self.decoder(latent)  # (B, N, patch_dim)

        # Compute reconstruction loss only on masked patches
        loss = self.forward_loss(pixel_values, pred, mask)

        return loss, pred, mask