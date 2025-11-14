"""
Uniform TUBE masking strategy for Video MAE with TimeSformer encoder.

Video MAE uses TUBE masking: same spatial positions are masked across all frames.
This creates temporal consistency and allows learning temporal dynamics.

UNIFORM masking: Same mask ratio for all frames (required for TimeSformer's
divided space-time attention, which needs regular T×S structure).
"""
import torch
import numpy as np
from typing import Tuple


def random_masking_uniform_tubes(
    x: torch.Tensor,
    mask_ratio: float,
    num_patches_per_frame: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform uniform TUBE masking - Video MAE standard approach.

    TUBE MASKING: Same spatial positions are masked/kept across all frames.
    UNIFORM: Same mask ratio for all frames (enables TimeSformer compatibility).

    Args:
        x: [B, T*S, D] patch embeddings (T frames, S patches/frame)
        mask_ratio: Single mask ratio for all frames (e.g., 0.75)
        num_patches_per_frame: S = patches per frame

    Returns:
        x_masked: [B, T*len_keep, D] visible patches (regular structure!)
        mask: [B, T*S] binary mask (1 = masked, 0 = kept)
        ids_restore: [B, T*S] indices to restore original order
    """
    B, N, D = x.shape
    T = N // num_patches_per_frame
    S = num_patches_per_frame
    assert N == T * S, f"Expected {T*S} patches, got {N}"

    # Reshape to [B, T, S, D]
    x_frames = x.reshape(B, T, S, D)

    # ============================================================
    # KEY: Sample random order ONCE for spatial positions
    # This creates TUBES - same spatial positions across all frames
    # ============================================================
    noise = torch.rand(B, S, device=x.device)  # [B, S] - SAME for all frames!
    ids_shuffle = torch.argsort(noise, dim=1)  # [B, S] - spatial position ordering
    ids_restore_spatial = torch.argsort(ids_shuffle, dim=1)  # [B, S]

    # Number of patches to keep (SAME for all frames - uniform!)
    len_keep = int(S * (1 - mask_ratio))

    # Keep first len_keep patches (same positions for all frames)
    ids_keep = ids_shuffle[:, :len_keep]  # [B, len_keep]

    # Gather kept patches for all frames
    x_kept_frames = []
    for t in range(T):
        x_t = x_frames[:, t, :, :]  # [B, S, D]
        x_kept_t = torch.gather(x_t, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        x_kept_frames.append(x_kept_t)

    # Concatenate: [B, T, len_keep, D] -> [B, T*len_keep, D]
    x_masked = torch.cat(x_kept_frames, dim=1)  # [B, T*len_keep, D]

    # Binary mask: 1 = masked, 0 = kept (same for all frames)
    mask_template = torch.ones(B, S, device=x.device)
    mask_template[:, :len_keep] = 0
    mask_template = torch.gather(mask_template, dim=1, index=ids_restore_spatial)  # [B, S]

    # Repeat for all frames
    mask = mask_template.unsqueeze(1).repeat(1, T, 1).reshape(B, T * S)  # [B, T*S]

    # ids_restore: repeat for all frames
    ids_restore = ids_restore_spatial.unsqueeze(1).repeat(1, T, 1).reshape(B, T * S)  # [B, T*S]

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