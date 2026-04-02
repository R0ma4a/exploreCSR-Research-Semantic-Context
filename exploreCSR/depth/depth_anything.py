"""
DepthAnything V2 — monocular depth estimation wrapper.

Handles model loading, image preprocessing, depth prediction, and
post-processing (outlier removal, normalization, smoothing).
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import torch

import depth_anything_v2.dpt as dpt


class DepthAnything:
    """Wrapper around DepthAnything V2 for monocular depth estimation."""

    def __init__(
        self,
        checkpoint_path: str,
        encoder: str = "vitb",
        features: int = 128,
        out_channels: Tuple[int, ...] = (96, 192, 384, 768),
        input_size: Tuple[int, int] = (518, 518),
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size

        self.model = dpt.DepthAnythingV2(
            encoder=encoder,
            features=features,
            out_channels=list(out_channels),
            use_bn=False,
            use_clstoken=False,
        )

        state_dict = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    @staticmethod
    def load_rgb(image_path: str) -> np.ndarray:
        """Load an image as RGB uint8 (H, W, 3)."""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess(
        self, image_path: str
    ) -> Tuple[torch.Tensor, np.ndarray, int, int]:
        """
        Read an image, resize for the model, and return the input tensor
        along with the original RGB image and its dimensions.

        Returns
        -------
        image_tensor : (1, 3, H_in, W_in) float32 tensor on self.device
        rgb_image    : (H, W, 3) uint8 original-resolution RGB
        original_w   : int
        original_h   : int
        """
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        rgb_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        original_h, original_w = rgb_image.shape[:2]

        resized = cv2.resize(
            cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
            (self.input_size[1], self.input_size[0]),
        )
        normalized = resized.astype(np.float32) / 255.0
        chw = np.transpose(normalized, (2, 0, 1))
        tensor = torch.from_numpy(chw).unsqueeze(0).to(self.device)

        return tensor, rgb_image, original_w, original_h

    def frame_to_tensor(
        self, frame_bgr: np.ndarray
    ) -> Tuple[torch.Tensor, int, int]:
        """
        Convert a BGR frame (e.g. from cv2.VideoCapture) into a model-ready
        tensor. Same preprocessing as ``preprocess`` but from an in-memory frame.

        Returns
        -------
        tensor     : (1, 3, H_in, W_in) float32 tensor on self.device
        original_w : int
        original_h : int
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        original_h, original_w = rgb.shape[:2]

        resized = cv2.resize(rgb, (self.input_size[1], self.input_size[0]))
        normalized = resized.astype(np.float32) / 255.0
        chw = np.transpose(normalized, (2, 0, 1))
        tensor = torch.from_numpy(chw).unsqueeze(0).to(self.device)

        return tensor, original_w, original_h

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Run depth prediction. Returns raw (H_model, W_model) depth map."""
        with torch.no_grad():
            depth = self.model(image_tensor)
        return depth.squeeze(0).cpu().numpy()

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    @staticmethod
    def postprocess(
        depth: np.ndarray,
        original_w: int,
        original_h: int,
        outlier_percentile: Tuple[float, float] = (2.0, 98.0),
        blur_ksize: Tuple[int, int] = (5, 5),
    ) -> np.ndarray:
        """
        Clip outliers, normalize to [0, 1], resize to original resolution,
        and apply light Gaussian smoothing.
        """
        low, high = np.percentile(depth, outlier_percentile)
        depth = np.clip(depth, low, high)

        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth_norm = cv2.resize(
            depth_norm,
            (original_w, original_h),
            interpolation=cv2.INTER_CUBIC,
        )
        depth_norm = cv2.GaussianBlur(depth_norm, blur_ksize, 0)
        return depth_norm

    # ------------------------------------------------------------------
    # Convenience: full depth pipeline for a single image
    # ------------------------------------------------------------------

    def estimate_depth(self, image_path: str) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """
        End-to-end depth estimation from an image path.

        Returns
        -------
        depth_norm  : (H, W) float64, normalized depth map at original resolution
        rgb_image   : (H, W, 3) uint8 original RGB
        original_w  : int
        original_h  : int
        """
        tensor, rgb_image, w, h = self.preprocess(image_path)
        raw_depth = self.predict(tensor)
        depth_norm = self.postprocess(raw_depth, w, h)
        return depth_norm, rgb_image, w, h

    # ------------------------------------------------------------------
    # Visualization helpers (kept for backward compat)
    # ------------------------------------------------------------------

    @staticmethod
    def save_depth_image(
        depth_norm: np.ndarray, output_path: str, name: str
    ) -> None:
        """Save a colorized depth map as a PNG."""
        depth_8bit = (depth_norm * 255).astype("uint8")
        depth_color = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_PLASMA)
        cv2.imwrite(f"{output_path}{name}_depth.png", depth_color)

    @staticmethod
    def create_side_by_side(
        rgb_image: np.ndarray, depth_norm: np.ndarray
    ) -> np.ndarray:
        """Create a horizontal concatenation of RGB and colorized depth."""
        depth_vis = (depth_norm.squeeze() * 255).astype("uint8")
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        return np.hstack((rgb_image, depth_color))
