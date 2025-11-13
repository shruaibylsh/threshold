"""
Asymmetric masking strategy for Threshold sequence MAE pretraining.

The key idea: Mask "before" and "after" frames heavily, but keep "at threshold"
frames relatively visible to provide reconstruction context.
"""
import torch
import numpy as np
from typing import Tuple


def random_masking_asymmetric(
    x: torch.Tensor,
    frame_mask_ratios: Tuple[float, ...],
    num_patches_per_frame: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform asymmetric random masking with different ratios per frame.

    Args:
        x: [B, T*S, D] patch embeddings (T frames, S patches/frame)
        frame_mask_ratios: Tuple of T mask ratios, one per frame
        num_patches_per_frame: S = patches per frame

    Returns:
        x_masked: [B, N_keep, D] visible patches only
        mask: [B, T*S] binary mask (1 = masked, 0 = kept)
        ids_restore: [B, T*S] indices to restore original order
    """
    B, N, D = x.shape
    T = len(frame_mask_ratios)
    S = num_patches_per_frame
    assert N == T * S, f"Expected {T*S} patches, got {N}"

    # Reshape to [B, T, S, D] to apply per-frame masking
    x_frames = x.reshape(B, T, S, D)

    # Lists to collect kept/masked patches
    x_kept_frames = []
    mask_frames = []
    ids_restore_frames = []

    for t in range(T):
        mask_ratio_t = frame_mask_ratios[t]
        x_t = x_frames[:, t, :, :]  # [B, S, D]

        # Number of patches to keep for this frame
        len_keep = int(S * (1 - mask_ratio_t))

        # Random shuffle
        noise = torch.rand(B, S, device=x.device)  # [B, S]
        ids_shuffle = torch.argsort(noise, dim=1)  # [B, S]
        ids_restore_t = torch.argsort(ids_shuffle, dim=1)  # [B, S]

        # Keep first len_keep patches
        ids_keep = ids_shuffle[:, :len_keep]  # [B, len_keep]
        x_kept_t = torch.gather(x_t, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # Binary mask: 1 = masked, 0 = kept
        mask_t = torch.ones(B, S, device=x.device)
        mask_t[:, :len_keep] = 0
        mask_t = torch.gather(mask_t, dim=1, index=ids_restore_t)  # [B, S]

        x_kept_frames.append(x_kept_t)
        mask_frames.append(mask_t)
        ids_restore_frames.append(ids_restore_t)

    # Concatenate across frames
    x_masked = torch.cat(x_kept_frames, dim=1)  # [B, sum(len_keep), D]
    mask = torch.cat(mask_frames, dim=1)  # [B, T*S]
    ids_restore = torch.cat(ids_restore_frames, dim=1)  # [B, T*S]

    return x_masked, mask, ids_restore


def patchify(imgs: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Convert images to patches.

    Args:
        imgs: [B, T, C, H, W]
        patch_size: P

    Returns:
        patches: [B, T*S, P*P*C] where S = (H/P) * (W/P)
    """
    B, T, C, H, W = imgs.shape
    assert H % patch_size == 0 and W % patch_size == 0

    Ph = H // patch_size
    Pw = W // patch_size
    S = Ph * Pw

    # Reshape: [B, T, C, Ph, P, Pw, P]
    x = imgs.reshape(B, T, C, Ph, patch_size, Pw, patch_size)

    # Permute: [B, T, Ph, Pw, P, P, C]
    x = x.permute(0, 1, 3, 5, 4, 6, 2).contiguous()

    # Flatten: [B, T, Ph*Pw, P*P*C] = [B, T, S, P*P*C]
    patches = x.reshape(B, T, S, patch_size * patch_size * C)

    # Flatten temporal: [B, T*S, P*P*C]
    patches = patches.reshape(B, T * S, -1)

    return patches


def unpatchify(patches: torch.Tensor, patch_size: int, T: int, C: int, H: int, W: int) -> torch.Tensor:
    """
    Convert patches back to images.

    Args:
        patches: [B, T*S, P*P*C]
        patch_size: P
        T: num frames
        C: channels
        H, W: image dimensions

    Returns:
        imgs: [B, T, C, H, W]
    """
    B = patches.shape[0]
    Ph = H // patch_size
    Pw = W // patch_size
    S = Ph * Pw

    # Reshape: [B, T, S, P*P*C]
    patches = patches.reshape(B, T, S, patch_size * patch_size * C)

    # Reshape: [B, T, Ph, Pw, P, P, C]
    patches = patches.reshape(B, T, Ph, Pw, patch_size, patch_size, C)

    # Permute: [B, T, C, Ph, P, Pw, P]
    x = patches.permute(0, 1, 6, 2, 4, 3, 5).contiguous()

    # Reshape: [B, T, C, H, W]
    imgs = x.reshape(B, T, C, H, W)

    return imgs
