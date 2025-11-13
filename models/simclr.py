"""
SimCLR (Simple Framework for Contrastive Learning of Visual Representations)

Based on Chen et al., ICML 2020 and your previous working implementation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """2-layer MLP with ReLU; output L2-normalized z."""
    def __init__(self, in_dim, hidden=512, out_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, h):
        """
        Args:
            h: (B, in_dim) encoder embeddings
        Returns:
            z: (B, out_dim) L2-normalized projections
        """
        z = F.relu(self.fc1(h), inplace=True)
        z = self.fc2(z)
        z = F.normalize(z, dim=-1)  # L2 normalize
        return z


class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    @torch.amp.autocast("cuda", enabled=True)  # force float32 math here (safe for both CPU/GPU)
    def forward(self, z1, z2):
        """
        z1, z2: [B, d] projected & L2-normalized embeddings (may arrive in fp16)
        Returns scalar loss in float32.
        """
        # Ensure float32 to avoid fp16 overflow/underflow in logits & masking
        z1 = F.normalize(z1.float(), dim=1)
        z2 = F.normalize(z2.float(), dim=1)

        z = torch.cat([z1, z2], dim=0)                # [2B, d]
        B = z1.shape[0]

        # cosine similarity matrix scaled by temperature -> logits shape [2B, 2B]
        logits = (z @ z.t()) / self.temperature       # float32

        # mask self-contrast on the diagonal with -inf (safe in float32)
        eye = torch.eye(2 * B, device=logits.device, dtype=torch.bool)
        logits = logits.masked_fill(eye, float('-inf'))

        # positives: for i in [0..B-1], positive is i+B; for i in [B..2B-1], positive is i-B
        pos_idx = torch.arange(B, device=logits.device)
        targets = torch.cat([pos_idx + B, pos_idx], dim=0)  # [2B]

        loss = F.cross_entropy(logits, targets)
        return loss


# ---- SimCLR wrapper module ----
class SimCLR(nn.Module):
    def __init__(self, encoder: nn.Module, proj_hidden=512, proj_out=128, temperature=0.1):
        super().__init__()
        self.encoder = encoder
        # infer encoder dim
        with torch.no_grad():
            dummy = torch.zeros(2, 7, 1, 32, 64)
            enc_dim = encoder(dummy).shape[-1]
        self.head = ProjectionHead(enc_dim, hidden=proj_hidden, out_dim=proj_out)
        self.criterion = NTXentLoss(temperature=temperature)

    def forward(self, v1, v2):
        h1 = self.encoder(v1)      # [B,D]
        h2 = self.encoder(v2)      # [B,D]
        z1 = self.head(h1)         # [B,d]
        z2 = self.head(h2)
        loss = self.criterion(z1, z2)
        return loss, (h1, h2), (z1, z2)


def create_simclr_model(config=None, proj_hidden=512, proj_out=128, temperature=0.1):
    """
    Factory function to create SimCLR model.

    Args:
        config: Model configuration (optional, uses defaults if None)
        proj_hidden: Hidden dimension of projection head
        proj_out: Output dimension of projection head
        temperature: Temperature for NT-Xent loss

    Returns:
        SimCLR model
    """
    from timesformer_encoder import TimeSformerEncoder

    # Create encoder with correct parameters for grayscale clips
    encoder = TimeSformerEncoder(
        in_ch=1,           # Grayscale
        embed_dim=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        drop=0.1,
        attn_drop=0.0,
        patch=(8, 8),      # 8x8 patches
        T=7,
        H=32,
        W=64
    )

    model = SimCLR(
        encoder=encoder,
        proj_hidden=proj_hidden,
        proj_out=proj_out,
        temperature=temperature
    )
    return model