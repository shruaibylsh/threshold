"""
TimeSformer with VideoMAE Pretraining.
"""

import torch
import torch.nn as nn
from einops import rearrange
import math


class PatchEmbed(nn.Module):
    """Video to Patch Embedding."""
    def __init__(self, image_height=32, image_width=64, patch_size=4, in_channels=3, embed_dim=384):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.patch_size = patch_size
        self.num_patches_h = image_height // patch_size
        self.num_patches_w = image_width // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
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
    """Divided Space-Time Attention."""
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
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if self.attention_type == 'temporal':
            q = rearrange(q, 'b h (t p) d -> b h p t d', t=num_frames, p=num_patches_per_frame)
            k = rearrange(k, 'b h (t p) d -> b h p t d', t=num_frames, p=num_patches_per_frame)
            v = rearrange(v, 'b h (t p) d -> b h p t d', t=num_frames, p=num_patches_per_frame)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v)
            x = rearrange(x, 'b h p t d -> b (t p) (h d)')
        else:  # spatial
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
        
        self.patch_embed = PatchEmbed(
            image_height=config.image_height,
            image_width=config.image_width,
            patch_size=config.patch_size,
            in_channels=config.num_channels,
            embed_dim=config.hidden_size,
        )
        
        num_patches = self.patch_embed.num_patches * config.num_frames
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, config.hidden_size))
        self.pos_drop = nn.Dropout(p=config.hidden_dropout_prob)
        
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

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, T, C, H, W) or (B, N, D) for masked input
            mask: Optional mask for visible tokens
        Returns:
            x: (B, N, D) encoded representations
        """
        if x.dim() == 5:  # Full video input
            # Patch embedding
            x = self.patch_embed(x)  # (B, T*P, D)
        
        # Add positional embedding
        if mask is not None:
            x = x + self.pos_embed[:, mask, :]
        else:
            x = x + self.pos_embed
        
        x = self.pos_drop(x)
        
        # Apply transformer blocks
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
        
        self.decoder_embed = nn.Linear(config.hidden_size, config.decoder_hidden_size, bias=True)
        num_patches = config.total_patches
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, config.decoder_hidden_size))
        
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
        patch_dim = config.patch_size ** 2 * config.num_channels
        self.decoder_pred = nn.Linear(config.decoder_hidden_size, patch_dim, bias=True)
        
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
        x = self.decoder_embed(x)
        x = x + self.decoder_pos_embed
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x


class TimeSformerMAE(nn.Module):
    """Complete TimeSformer with VideoMAE pretraining."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = TimeSformerEncoder(config)
        self.decoder = MAEDecoder(config)
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        self.mask_ratio = config.mask_ratio

    def random_masking(self, x):
        B, N, D = x.shape
        len_keep = int(N * (1 - self.mask_ratio))
        
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        # THE FIX: Replace masked patches with mask token, keep all N patches
        mask_tokens = self.mask_token.expand(B, N, -1)
        x_masked = x * (1 - mask.unsqueeze(-1)) + mask_tokens * mask.unsqueeze(-1)
        
        return x_masked, mask

    def patchify(self, imgs):
        B, T, C, H, W = imgs.shape
        p = self.config.patch_size
        h = H // p
        w = W // p
        x = imgs.reshape(B, T, C, h, p, w, p)
        x = x.permute(0, 1, 3, 5, 4, 6, 2).reshape(B, T * h * w, p * p * C)
        return x

    def unpatchify(self, x):
        p = self.config.patch_size
        T = self.config.num_frames
        C = self.config.num_channels
        h = self.config.num_patches_height
        w = self.config.num_patches_width
        x = x.reshape(x.shape[0], T, h, w, p, p, C)
        x = x.permute(0, 1, 6, 2, 4, 3, 5).reshape(x.shape[0], T, C, h * p, w * p)
        return x

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        if self.config.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** 0.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, imgs):
        """
        Args:
            imgs: (B, T, C, H, W)
        Returns:
            loss, pred, mask
        """
        # 1. Patch embedding
        x = self.encoder.patch_embed(imgs)  # (B, 896, D)
        
        # 2. Apply masking
        x_masked, mask = self.random_masking(x)  # (B, 896, D)
        
        # 3. Encode
        latent = self.encoder(x_masked)  # (B, 896, D)
        
        # 4. Decode
        pred = self.decoder(latent)  # (B, 896, patch_dim)
        
        # 5. Loss
        loss = self.forward_loss(imgs, pred, mask)
        
        return loss, pred, mask