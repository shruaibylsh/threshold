"""
TimeSformer MAE (Masked Autoencoder) for threshold sequence pretraining.

Architecture:
- Encoder: TimeSformer with divided space-time attention
- Decoder: Lightweight standard transformer
- Masking: Uniform tube masking (Video MAE standard)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models_MAE_timesformer.masking import random_masking_uniform_tubes, patchify, unpatchify


class PatchEmbed(nn.Module):
    """2D conv patch embedding."""
    def __init__(self, in_ch=1, embed_dim=384, patch_h=8, patch_w=8):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=(patch_h, patch_w),
                              stride=(patch_h, patch_w))

    def forward(self, x):  # x: [B, C, H, W]
        x = self.proj(x)   # [B, D, H/Ph, W/Pw]
        B, D, Hn, Wn = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()  # [B, Hn*Wn, D]
        return x, (Hn, Wn)


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=6, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1,2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DividedSpaceTimeBlock(nn.Module):
    """
    TimeSformer divided space-time attention block.
    Used in ENCODER for processing visible patches after tube masking.
    """
    def __init__(self, dim, num_heads=6, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        # Temporal attention
        self.norm_t1 = nn.LayerNorm(dim)
        self.attn_t  = Attention(dim, num_heads, attn_drop, drop)
        self.norm_t2 = nn.LayerNorm(dim)
        self.mlp_t   = MLP(dim, mlp_ratio, drop)

        # Spatial attention
        self.norm_s1 = nn.LayerNorm(dim)
        self.attn_s  = Attention(dim, num_heads, attn_drop, drop)
        self.norm_s2 = nn.LayerNorm(dim)
        self.mlp_s   = MLP(dim, mlp_ratio, drop)

    def forward(self, x, T, S):
        """
        x: [B, T*S, D]  (T frames, S patches per frame)
        T: number of frames
        S: number of VISIBLE patches per frame (same for all frames after uniform masking)
        """
        B, N, D = x.shape
        assert N == T * S, f"Expected {T*S} tokens, got {N}"

        # ===== Temporal Attention =====
        # Reshape: [B, T*S, D] -> [B, S, T, D] -> [B*S, T, D]
        xt = x.reshape(B, T, S, D).transpose(1, 2).contiguous()   # [B, S, T, D]
        xt = xt.reshape(B * S, T, D)  # [B*S, T, D] - each spatial position attends across time

        # Attention + residual + MLP
        y = self.norm_t1(xt)
        y = self.attn_t(y)
        xt = xt + y
        xt = xt + self.mlp_t(self.norm_t2(xt))

        # Reshape back: [B*S, T, D] -> [B, S, T, D] -> [B, T, S, D] -> [B, T*S, D]
        xt = xt.reshape(B, S, T, D).transpose(1, 2).reshape(B, T * S, D)

        # ===== Spatial Attention =====
        # Reshape: [B, T*S, D] -> [B, T, S, D] -> [B*T, S, D]
        xs = xt.reshape(B, T, S, D).reshape(B * T, S, D)  # [B*T, S, D] - each frame attends spatially

        # Attention + residual + MLP
        y = self.norm_s1(xs)
        y = self.attn_s(y)
        xs = xs + y
        xs = xs + self.mlp_s(self.norm_s2(xs))

        # Reshape back: [B*T, S, D] -> [B, T, S, D] -> [B, T*S, D]
        x = xs.reshape(B, T, S, D).reshape(B, T * S, D)

        return x


class TransformerBlock(nn.Module):
    """
    Standard transformer block.
    Used in DECODER (operates on full sequence at once).
    """
    def __init__(self, dim, num_heads=6, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TimeSformerMAE(nn.Module):
    """
    MAE with TimeSformer encoder (divided space-time) and lightweight decoder.
    Uses uniform tube masking - Video MAE standard approach.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Image params
        self.T = config.num_frames
        self.H, self.W = config.image_height, config.image_width
        self.C = config.num_channels
        self.patch_size = config.patch_size
        self.S = (self.H // self.patch_size) * (self.W // self.patch_size)

        # Patch embedding
        self.patch_embed = PatchEmbed(
            in_ch=self.C,
            embed_dim=config.encoder_embed_dim,
            patch_h=self.patch_size,
            patch_w=self.patch_size
        )

        # Positional embeddings (encoder) - TimeSformer style: temporal + spatial
        self.temb = nn.Parameter(torch.zeros(1, self.T, config.encoder_embed_dim))
        self.semb = nn.Parameter(torch.zeros(1, self.S, config.encoder_embed_dim))
        nn.init.trunc_normal_(self.temb, std=0.02)
        nn.init.trunc_normal_(self.semb, std=0.02)

        # Encoder - TimeSformer with divided space-time attention!
        self.encoder_blocks = nn.ModuleList([
            DividedSpaceTimeBlock(
                config.encoder_embed_dim,
                config.encoder_num_heads,
                config.encoder_mlp_ratio,
                config.dropout,
                config.attention_dropout
            )
            for _ in range(config.encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(config.encoder_embed_dim)

        # Decoder embedding
        self.decoder_embed = nn.Linear(config.encoder_embed_dim, config.decoder_embed_dim)

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Decoder positional embedding (learned, not factorized)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.T * self.S, config.decoder_embed_dim)
        )
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

        # Decoder blocks (standard transformer)
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(
                config.decoder_embed_dim,
                config.decoder_num_heads,
                config.decoder_mlp_ratio,
                config.dropout,
                config.attention_dropout
            )
            for _ in range(config.decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(config.decoder_embed_dim)

        # Prediction head: patches -> pixels
        self.decoder_pred = nn.Linear(
            config.decoder_embed_dim,
            self.patch_size * self.patch_size * self.C
        )

    def forward_encoder(self, x, mask_ratio=None):
        """
        Encode with uniform tube masking.

        Args:
            x: [B, T, C, H, W]
            mask_ratio: Mask ratio (default: use config)

        Returns:
            x_encoded: [B, T*S_visible, D] encoded visible patches
            mask: [B, T*S] binary mask
            ids_restore: [B, T*S] for restoring order
        """
        if mask_ratio is None:
            mask_ratio = self.config.mask_ratio

        B, T, C, H, W = x.shape

        # Patchify each frame
        x = x.reshape(B * T, C, H, W)
        tokens, _ = self.patch_embed(x)  # [B*T, S, D]
        tokens = tokens.reshape(B, T, self.S, -1)  # [B, T, S, D]

        # Add positional embeddings (factorized: temporal + spatial)
        tokens = tokens + self.temb[:, :T, :].unsqueeze(2) + self.semb[:, :self.S, :].unsqueeze(1)
        x = tokens.reshape(B, T * self.S, -1)  # [B, T*S, D]

        # Uniform tube masking
        x_masked, mask, ids_restore = random_masking_uniform_tubes(
            x, mask_ratio, self.S
        )
        # x_masked: [B, T*S_visible, D] where S_visible is same for all frames!

        # TimeSformer encoder (divided space-time attention)
        # After uniform masking: each frame has S_visible patches (regular structure!)
        S_visible = x_masked.shape[1] // T
        for blk in self.encoder_blocks:
            x_masked = blk(x_masked, T=T, S=S_visible)

        x_masked = self.encoder_norm(x_masked)

        return x_masked, mask, ids_restore

    def forward_decoder(self, x_encoded, ids_restore):
        """
        Decoder: add mask tokens, run through transformer, predict pixels.

        Args:
            x_encoded: [B, T*S_visible, encoder_dim]
            ids_restore: [B, T*S]

        Returns:
            pred: [B, T*S, patch_pixels]
        """
        # Project to decoder dim
        x = self.decoder_embed(x_encoded)  # [B, T*S_visible, decoder_dim]

        # Append mask tokens
        B, N_keep, D = x.shape
        N = self.T * self.S
        mask_tokens = self.mask_token.repeat(B, N - N_keep, 1)  # [B, N_masked, D]
        x_full = torch.cat([x, mask_tokens], dim=1)  # [B, N, D]

        # Unshuffle to original order
        x_full = torch.gather(
            x_full, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, D)
        )  # [B, N, D]

        # Add positional embedding
        x = x_full + self.decoder_pos_embed

        # Decoder transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)

        # Predict pixels
        pred = self.decoder_pred(x)  # [B, T*S, patch_pixels]

        return pred

    def forward_loss(self, imgs, pred, mask):
        """
        Compute reconstruction loss (MSE) on masked patches only.

        Args:
            imgs: [B, T, C, H, W] original images
            pred: [B, T*S, patch_pixels] predictions
            mask: [B, T*S] binary mask (1 = masked, 0 = visible)

        Returns:
            loss: scalar
        """
        # Patchify targets
        target = patchify(imgs, self.patch_size)  # [B, T*S, patch_pixels]

        if self.config.norm_pix_loss:
            # Normalize per-patch
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5

        # MSE loss on masked patches only
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [B, T*S]

        # Average over masked patches
        loss = (loss * mask).sum() / mask.sum()

        return loss

    def forward(self, imgs, mask_ratio=None):
        """
        Forward pass: encode with masking, decode, compute loss.

        Args:
            imgs: [B, T, C, H, W]
            mask_ratio: Mask ratio (default: use config)

        Returns:
            loss: scalar
            pred: [B, T*S, patch_pixels]
            mask: [B, T*S]
        """
        # Encode
        x_encoded, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)

        # Decode
        pred = self.forward_decoder(x_encoded, ids_restore)

        # Loss
        loss = self.forward_loss(imgs, pred, mask)

        return loss, pred, mask