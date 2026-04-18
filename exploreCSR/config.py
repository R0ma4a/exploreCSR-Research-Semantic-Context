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

    fx: float = 591.0       # default: NOCS real camera
    fy: float = 590.0
    cx: Optional[float] = 322.0
    cy: Optional[float] = 244.0

    def intrinsics_matrix(self, image_width: int = 0, image_height: int = 0) -> np.ndarray:
        """Build a 3×3 intrinsics matrix K."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
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
    unsupervised_keep_fraction: float = 0.38
    use_depth_fusion: bool = True


@dataclass
class CAPTRAConfig:
    """Neural CAPTRA checkpoint paths and inference settings."""

    rot_dir: str = ""        # e.g. RGed-research/captra/runs/6_mug_rot
    coord_dir: str = ""      # e.g. RGed-research/captra/runs/6_mug_coord
    category: int = 6        # NOCS category index (1=bottle … 6=mug)
    num_points: int = 4096   # point cloud size fed to PointNet2


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration combining all sub-configs."""

    depth: DepthConfig = field(default_factory=DepthConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    captra: CAPTRAConfig = field(default_factory=CAPTRAConfig)
