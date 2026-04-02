"""
Centralized configuration and default values for the BrownCSR pipeline.

Override these by passing arguments directly to constructors or pipeline
functions — these are fallback defaults only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class CameraConfig:
    """Pinhole camera intrinsics and depth settings."""

    fx: float = 500.0
    fy: float = 500.0
    cx: Optional[float] = None  # None → image_width / 2
    cy: Optional[float] = None  # None → image_height / 2
    depth_scale: float = 1.0

    def intrinsics_matrix(self, image_width: int, image_height: int) -> np.ndarray:
        """Build a 3×3 intrinsics matrix K, filling cx/cy from image size if needed."""
        cx = self.cx if self.cx is not None else image_width / 2.0
        cy = self.cy if self.cy is not None else image_height / 2.0
        return np.array(
            [[self.fx, 0.0, cx], [0.0, self.fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass
class DepthConfig:
    """DepthAnything model configuration."""

    encoder: str = "vitb"
    features: int = 128
    out_channels: Tuple[int, ...] = (96, 192, 384, 768)
    input_size: Tuple[int, int] = (518, 518)
    outlier_percentile: Tuple[float, float] = (2.0, 98.0)
    gaussian_blur_ksize: Tuple[int, int] = (5, 5)


@dataclass
class SegmentationConfig:
    """DINO segmentation configuration."""

    model_name: str = "vit_small_patch16_224.dino"
    keep_fraction: float = 0.25
    top_k_heads: Optional[int] = None
    use_edge_refine: bool = True
    use_bilateral: bool = True
    use_image_edges: bool = True
    # For generate_object_mask (unsupervised)
    unsupervised_keep_fraction: float = 0.38
    use_depth_fusion: bool = True


@dataclass
class CAPTRAConfig:
    """CAPTRA pose estimation configuration."""

    min_points: int = 50
    pca_eps: float = 1e-6
    max_points_for_state: int = 10_000


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration combining all sub-configs."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    captra: CAPTRAConfig = field(default_factory=CAPTRAConfig)
