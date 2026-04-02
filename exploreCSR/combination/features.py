"""
DINO feature extraction for object-level representations.

Extracts per-frame feature vectors from DINO by pooling patch tokens
within the object mask. These features capture the object's appearance
and are used alongside CAPTRA pose vectors in the combination module.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def extract_object_features(
    segmenter,
    image_tensor: torch.Tensor,
    mask: np.ndarray,
) -> dict:
    """
    Extract DINO feature vectors for the masked object region.

    Computes three feature representations:
    - cls_token    : (D,) the CLS token — global image summary
    - patch_mean   : (D,) mean of ALL patch tokens (global)
    - object_mean  : (D,) mean of patch tokens WITHIN the object mask
                     (this is the most meaningful for tracking object changes)

    Parameters
    ----------
    segmenter    : DINOSegmenter instance
    image_tensor : (1, 3, H, W) input tensor
    mask         : (H_out, W_out) binary mask from segmentation

    Returns
    -------
    dict with keys: cls_token, patch_mean, object_mean, patch_grid_shape
    """
    image_tensor = image_tensor.to(segmenter.device)

    # Pad to patch-aligned size (same logic as DINOSegmenter.get_attention_map)
    _, _, H_orig, W_orig = image_tensor.shape
    p = segmenter.patch_size
    H_pad = math.ceil(H_orig / p) * p
    W_pad = math.ceil(W_orig / p) * p

    image_padded = F.interpolate(
        image_tensor, size=(H_pad, W_pad), mode="bilinear", align_corners=False
    )

    # Forward pass for features
    features = segmenter.extract_features(image_padded)

    # CLS token
    cls_token = features[0, 0, :].cpu().numpy()  # (D,)

    # Patch tokens → spatial grid
    patch_tokens = features[:, 1:, :]
    N = patch_tokens.shape[1]
    side = int(math.sqrt(N))
    patch_tokens = patch_tokens[:, : side * side, :]
    patch_grid = patch_tokens.reshape(1, side, side, -1)  # (1, H_p, W_p, D)

    B, H_p, W_p, D = patch_grid.shape
    patch_np = patch_grid[0].cpu().numpy()  # (H_p, W_p, D)

    # Global patch mean
    patch_mean = patch_np.reshape(-1, D).mean(axis=0)  # (D,)

    # Downsample mask to patch grid resolution
    mask_resized = F.interpolate(
        torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
        size=(H_p, W_p),
        mode="bilinear",
        align_corners=False,
    ).squeeze().numpy()

    mask_binary = mask_resized > 0.5  # (H_p, W_p)
    num_masked = mask_binary.sum()

    if num_masked > 0:
        # Mean of patch features inside the object mask
        object_features = patch_np[mask_binary]  # (N_masked, D)
        object_mean = object_features.mean(axis=0)  # (D,)
    else:
        # Fallback to global mean if mask is empty at patch resolution
        object_mean = patch_mean.copy()

    # L2-normalize all features for cosine-friendly comparisons
    for feat in [cls_token, patch_mean, object_mean]:
        norm = np.linalg.norm(feat)
        if norm > 1e-8:
            feat /= norm

    return {
        "cls_token": cls_token,
        "patch_mean": patch_mean,
        "object_mean": object_mean,
        "patch_grid_shape": (H_p, W_p, D),
        "num_masked_patches": int(num_masked),
    }
