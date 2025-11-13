import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- helpers ----
class PatchEmbed(nn.Module):
    """
    2D conv patchify: (C,H,W) -> (N_patches, D). Uses kernel=stride=patch_size.
    """
    def __init__(self, in_ch=1, embed_dim=384, patch_h=8, patch_w=8):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=(patch_h, patch_w),
                              stride=(patch_h, patch_w))
    def forward(self, x):  # x: [B, C, H, W]
        x = self.proj(x)   # [B, D, H/Ph, W/Pw]
        B, D, Hn, Wn = x.shape
        # flatten -> tokens per frame
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
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, heads, N, C_head]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B,heads,N,N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1,2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DividedSpaceTimeBlock(nn.Module):
    """
    TimeSformer 'divided space-time' attention:
      1) Temporal attention per spatial token across T
      2) Spatial attention per frame across S (patch tokens)
    """
    def __init__(self, dim, num_heads=6, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm_t1 = nn.LayerNorm(dim)
        self.attn_t  = Attention(dim, num_heads, attn_drop, drop)
        self.norm_t2 = nn.LayerNorm(dim)
        self.mlp_t   = MLP(dim, mlp_ratio, drop)

        self.norm_s1 = nn.LayerNorm(dim)
        self.attn_s  = Attention(dim, num_heads, attn_drop, drop)
        self.norm_s2 = nn.LayerNorm(dim)
        self.mlp_s   = MLP(dim, mlp_ratio, drop)

    def forward(self, x, T, S):
        """
        x: [B, T*S, D] without class token
        T: frames, S: spatial tokens per frame
        """
        B, N, D = x.shape
        assert N == T * S

        # --- Temporal attention: fix spatial index, attend over T ---
        xt = x.reshape(B, T, S, D).transpose(1, 2).contiguous()   # [B, S, T, D]
        xt = xt.reshape(B * S, T, D)                              # [B*S, T, D]
        y = self.norm_t1(xt)
        y = self.attn_t(y)
        xt = xt + y
        xt = xt + self.mlp_t(self.norm_t2(xt))
        xt = xt.reshape(B, S, T, D).transpose(1, 2).reshape(B, T * S, D)

        # --- Spatial attention: per-frame over S ---
        xs = xt.reshape(B, T, S, D).reshape(B * T, S, D)
        y = self.norm_s1(xs)
        y = self.attn_s(y)
        xs = xs + y
        xs = xs + self.mlp_s(self.norm_s2(xs))
        x  = xs.reshape(B, T, S, D).reshape(B, T * S, D)
        return x

class TimeSformerEncoder(nn.Module):
    """
    Minimal TimeSformer (no class token; mean pooling).
    For grayscale clips: [B,T,1,H,W] with H=32, W=64, patch=(8,8) => S=4*8=32 tokens/frame.
    """
    def __init__(self, in_ch=1, embed_dim=384, depth=6, num_heads=6,
                 mlp_ratio=4.0, drop=0.1, attn_drop=0.0, patch=(8,8), T=7, H=32, W=64):
        super().__init__()
        self.T = T
        self.H, self.W = H, W
        self.patch = patch
        self.patch_embed = PatchEmbed(in_ch, embed_dim, patch_h=patch[0], patch_w=patch[1])
        # spatial tokens per frame after patching
        self.S = (H // patch[0]) * (W // patch[1])

        # positional embeddings: temporal + spatial (additive)
        self.temb = nn.Parameter(torch.zeros(1, T, embed_dim))       # [1,T,D]
        self.semb = nn.Parameter(torch.zeros(1, self.S, embed_dim))  # [1,S,D]
        nn.init.trunc_normal_(self.temb, std=0.02)
        nn.init.trunc_normal_(self.semb, std=0.02)

        self.blocks = nn.ModuleList([
            DividedSpaceTimeBlock(embed_dim, num_heads, mlp_ratio, drop, attn_drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):  # x: [B,T,1,H,W]
        B, T, C, H, W = x.shape
        assert T == self.T and H == self.H and W == self.W, "Set encoder sizes to your data."
        # patchify each frame -> tokens per frame
        x = x.reshape(B * T, C, H, W)                 # [B*T,1,H,W]
        tokens, (Hp, Wp) = self.patch_embed(x)        # [B*T, S, D]
        S = tokens.shape[1]
        assert S == self.S
        tokens = tokens.reshape(B, T, S, -1)          # [B,T,S,D]

        # add temporal + spatial positional embeddings
        tokens = tokens + self.temb[:, :T, :].unsqueeze(2) + self.semb[:, :S, :].unsqueeze(1)
        x = tokens.reshape(B, T * S, -1)              # SAFE reshape instead of view

        # divided space-time blocks
        for blk in self.blocks:
            x = blk(x, T=T, S=S)

        x = self.norm(x)
        # mean pool over all tokens -> [B,D]
        feat = x.mean(dim=1)
        return feat  # h
