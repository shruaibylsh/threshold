import os, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from load_data import create_dataloaders, CFG   # your step 1 file
from transform import make_simclr_transforms, SimCLRWrapperDataset, collate_two_views
from timesformer_min import TimeSformerEncoder

# ---- Projection head & NT-Xent loss ----
class ProjectionHead(nn.Module):
    """2-layer MLP with ReLU; output L2-normalized z."""
    def __init__(self, in_dim, hidden=512, out_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
    def forward(self, h):
        z = F.relu(self.fc1(h), inplace=True)
        z = self.fc2(z)
        z = F.normalize(z, dim=-1)
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

def main():
    # --- data: use your existing train split (labels ignored here) ---
    dl_train, _, _ = create_dataloaders(CFG, batch_size=128, num_workers=4)
    aug = make_simclr_transforms()
    # Wrap underlying dataset from the DataLoader
    base_ds = dl_train.dataset
    ssl_ds  = SimCLRWrapperDataset(base_ds, aug)
    ssl_loader = DataLoader(ssl_ds, batch_size=128, shuffle=True,
                            num_workers=4, pin_memory=True, collate_fn=collate_two_views)

    # --- model ---
    encoder = TimeSformerEncoder(
        in_ch=1, embed_dim=384, depth=6, num_heads=6,
        mlp_ratio=4.0, drop=0.1, attn_drop=0.0,
        patch=(8,8), T=7, H=32, W=64
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimCLR(encoder, proj_hidden=512, proj_out=128, temperature=0.1).to(device)

    # --- optim ---
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)

    # --- train ---
    epochs = 50
    model.train()
    for ep in range(1, epochs+1):
        running = 0.0
        for v1, v2 in ssl_loader:
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)
            loss, *_ = model(v1, v2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item()
        sched.step()
        avg = running / max(1, len(ssl_loader))
        print(f"[SimCLR] epoch {ep:03d}  loss {avg:.4f}")

    # --- save encoder weights (for Step 3) ---
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.encoder.state_dict(), "checkpoints/timesformer_ssl.pth")
    print("Saved:", "checkpoints/timesformer_ssl.pth")

if __name__ == "__main__":
    main()
