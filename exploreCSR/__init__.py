"""
BrownCSR — Object-Centered Pose Estimation Pipeline
====================================================

RGB Images → Segmentation (DINO) + Depth (DepthAnything) → Pose (CAPTRA)
         → Combination Module (MLP) → Visual Feedback + Coordinate Features

Modules
-------
- browncsr.depth          : Monocular depth estimation (DepthAnything V2)
- browncsr.segmentation   : Object segmentation (DINO attention + prompt guidance)
- browncsr.pose           : Pose estimation (CAPTRA) and surface delta poses
- browncsr.visualization  : Plotting and overlay utilities
- browncsr.pipeline       : End-to-end pipeline runner
"""

__version__ = "0.1.0"
