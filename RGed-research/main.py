import cv2

import depth_anything
from dino_segmenter import DINOSegmenter
import numpy as np
 
# ------------------------------------------------
# Paths
# ------------------------------------------------
# Use absolute paths so this script works regardless of the
# current working directory when it is invoked.
 
 
image_path = r"D:\Research Projects\exploreCSR-Research-Semantic-Context\RGed-research\imgs\obj00004\img00001.jpeg"
rae_weights_path = r"D:\Research Projects\exploreCSR-Research-Semantic-Context\RGed-research\checkpoints\depth_anything_v2_vitb.pth"
roman_weights_path = r"C:\Users\roman\Downloads\depth_anything_v2_vitb.pth"
prompt = "salt and pepper shaker"  # Example prompt for object segmentation
 
# -- Initialize Models --
converter = depth_anything.DepthAnything(rae_weights_path)
segmenter = DINOSegmenter(model_name="vit_small_patch16_224.dino")
 
# -- Preprocess Image --
image_tensor, rgb_image, original_width, original_height = converter.image_to_tensor(image_path)
 
# -- Generate Depth Map --
depth = converter.predict_depth(image_tensor)
depth_norm = converter.process_depth(depth, original_width, original_height)
 