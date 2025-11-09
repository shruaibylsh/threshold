"""
TimeSformer with VideoMAE Pretraining.
Implements:
- TimeSformer encoder with divided space-time attention
- VideoMAE masking and reconstruction framework
- Lightweight decoder for pretraining
FIXED: Uses mask tokens instead of removing patches to preserve spatiotemporal structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math


class PatchEmbed(nn.Module):
    """
    Video to Patch Embedding.
    Divides each frame into non-overlapping patches.
    """
    def __init__(self, image_height=32, image_width=64, patch_size=4, in_channels=3, embed_dim=384):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.patch_size = patch_size
        self.num_patches_h = image_height // patch_size
        self.num_patches_w = image_width // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        # Convolutional projection
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            patches: (B, T * num_patches, embed_dim)
        """
        B, T, C, H, W = x.shape
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.proj(x)
        x = rearrange(x, '(b t) e h w -> b t (h w) e', b=B, t=T)
        x = rearrange(x, 'b t p e -> b (t p) e')
        return x


class DividedSpaceTimeAttention(nn.Module):
    """
    Divided Space-Time Attention (core of TimeSformer).
    Alternates between:
    1. Temporal attention: Each spatial position attends across time
    2. Spatial attention: Within each frame, patches attend to each other
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., attention_type='temporal'):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.attention_type = attention_type
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, num_frames, num_patches_per_frame):
        """
        Args:
            x: (B, T*P, D) where T*P must equal num_frames * num_patches_per_frame
            num_frames: Number of frames
            num_patches_per_frame: Patches per frame
        Returns:
            x: (B, T*P, D)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.attention_type == 'temporal':
            # Temporal attention: attend across time for each spatial position
            q = rearrange(q, 'b h (t p) d -> b h p t d', t=num_frames, p=num_patches_per_frame)
            k = rearrange(k, 'b h (t p) d -> b h p t d', t=num_frames, p=num_patches_per_frame)
            v = rearrange(v, 'b h (t p) d -> b h p t d', t=num_frames, p=num_patches_per_frame)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v)
            x = rearrange(x, 'b h p t d -> b (t p) (h d)')
        else:  # spatial
            # Spatial attention: attend within each frame
            q = rearrange(q, 'b h (t p) d -> b h t p d', t=num_frames, p=num_patches_per_frame)
            k = rearrange(k, 'b h (t p) d -> b h t p d', t=num_frames, p=num_patches_per_frame)
            v = rearrange(v, 'b h (t p) d -> b h t p d', t=num_frames, p=num_patches_per_frame)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v)
            x = rearrange(x, 'b h t p d -> b (t p) (h d)')
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with divided space-time attention."""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., attention_type='temporal'):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = DividedSpaceTimeAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, attention_type=attention_type
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x, num_frames, num_patches_per_frame):
        x = x + self.attn(self.norm1(x), num_frames, num_patches_per_frame)
        x = x + self.mlp(self.norm2(x))
        return x


class TimeSformerEncoder(nn.Module):
    """TimeSformer Encoder with divided space-time attention."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Patch embedding
        self.patch_embed = PatchEmbed(
            image_height=config.image_height,
            image_width=config.image_width,
            patch_size=config.patch_size,
            in_channels=config.num_channels,
            embed_dim=config.hidden_size,
        )
        # Positional embeddings
        num_patches = self.patch_embed.num_patches * config.num_frames
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, config.hidden_size))
        self.pos_drop = nn.Dropout(p=config.hidden_dropout_prob)
        # Transformer blocks (alternating temporal and spatial attention)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=config.hidden_size,
                num_heads=config.num_attention_heads,
                mlp_ratio=config.intermediate_size / config.hidden_size,
                qkv_bias=True,
                drop=config.hidden_dropout_prob,
                attn_drop=config.attention_probs_dropout_prob,
                attention_type='temporal' if i % 2 == 0 else 'spatial',
            )
            for i in range(config.num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size)
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
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
        Forward pass through encoder.
        Args:
            x: (B, N, D) patch embeddings (N must equal total_patches)
        Returns:
            x: (B, N, D) encoded features
        """
        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)
        # Pass through transformer blocks
        num_frames = self.config.num_frames
        num_patches_per_frame = self.patch_embed.num_patches
        for block in self.blocks:
            x = block(x, num_frames, num_patches_per_frame)
        x = self.norm(x)
        return x


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
            x: (B, N, encoder_dim) - encoded patches (all N patches, including masked ones)
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
    """Complete TimeSformer with VideoMAE pretraining."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = TimeSformerEncoder(config)
        self.decoder = MAEDecoder(config)
        # Learnable mask token (replaces masked patches during encoding)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.mask_ratio = config.mask_ratio

    def random_masking(self, x):
        """
        Random masking of patches using mask tokens.
        CRITICAL: Keeps all N patches to preserve spatiotemporal structure.
        Args:
            x: (B, N, D) patch embeddings
        Returns:
            x_masked: (B, N, D) - patches with mask tokens (SAME SHAPE!)
            mask: (B, N) - binary mask (1 = masked, 0 = visible)
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - self.mask_ratio))
        # Generate random noise for each patch
        noise = torch.rand(B, N, device=x.device)
        # Sort noise to get random shuffling
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        # Create binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        # Replace masked patches with mask token
        # CRITICAL: We keep all N patches, just replace content
        mask_tokens = self.mask_token.repeat(B, N, 1)
        x_masked = x * (1 - mask.unsqueeze(-1)) + mask_tokens * mask.unsqueeze(-1)
        return x_masked, mask

    def patchify(self, imgs):
        """
        Convert images to patches.
        Args:
            imgs: (B, T, C, H, W)
        Returns:
            patches: (B, T*H*W/P^2, P^2*C)
        """
        B, T, C, H, W = imgs.shape
        p = self.config.patch_size
        h = H // p
        w = W // p
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
        p = self.config.patch_size
        T = self.config.num_frames
        C = self.config.num_channels
        h = self.config.num_patches_height
        w = self.config.num_patches_width
        x = x.reshape(x.shape[0], T, h, w, p, p, C)
        x = x.permute(0, 1, 6, 2, 4, 3, 5).reshape(x.shape[0], T, C, h * p, w * p)
        return x

    def forward_loss(self, imgs, pred, mask):
        """
        Compute reconstruction loss only on masked patches.
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
            target = (target - mean) / (var + 1.e-6) ** 0.5
        # MSE loss
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # Mean over patch pixels
        # Compute loss only on masked patches
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, imgs):
        """
        Forward pass with masked autoencoding.
        Args:
            imgs: (B, T, C, H, W) - input video frames
        Returns:
            loss: reconstruction loss
            pred: predicted patches
            mask: binary mask (1 = masked, 0 = visible)
        """
        # 1. Convert images to patch embeddings
        x = self.encoder.patch_embed(imgs)  # (B, N, D) where N = total_patches
        # 2. Apply random masking (replaces masked patches with mask token)
        # CRITICAL: x_masked has same shape as x (all N patches preserved)
        x_masked, mask = self.random_masking(x)  # (B, N, D), (B, N)
        # 3. Encode with full spatiotemporal structure preserved
        # Now encoder receives all N=896 patches, so divided attention works
        latent = self.encoder(x_masked)  # (B, N, D)
        # 4. Decode to reconstruct all patches
        pred = self.decoder(latent)  # (B, N, patch_dim)
        # 5. Compute reconstruction loss only on masked patches
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask
