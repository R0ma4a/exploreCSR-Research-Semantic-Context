import depth_anything
import dino
import numpy as np
 
# ------------------------------------------------
# Paths
# ------------------------------------------------
# Use absolute paths so this script works regardless of the
# current working directory when it is invoked.
 
 
image_path = r"D:\Research Projects\exploreCSR-Research-Semantic-Context\RGed-research\imgs\obj00001\img00001.jpeg"
rae_weights_path = r"D:\Research Projects\exploreCSR-Research-Semantic-Context\RGed-research\checkpoints\depth_anything_v2_vitb.pth"
roman_weights_path = r"C:\Users\roman\Downloads\depth_anything_v2_vitb.pth"
prompt = "bag"  # Example prompt for object segmentation
 
# -- Initialize Models --
converter = depth_anything.DepthAnything(rae_weights_path)
segmenter = dino.dino()
 
# -- Preprocess Image --
image_tensor, rgb_image, original_width, original_height = converter.image_to_tensor(image_path)
 
# -- Generate Depth Map --
depth = converter.predict_depth(image_tensor)
depth_norm = converter.process_depth(depth, original_width, original_height)
 
# -- Generate Object Mask Unsupervised --
#mask = segmenter.generate_object_mask(
#    image_tensor,
#    depth_norm,
#    (original_height, original_width)
#)
 
# -- Generate Object Mask with Prompt --
mask = segmenter.segment_from_prompt(
    image_tensor,
    prompt,
    output_size=(original_height, original_width)
)
 
# -- Debug check --
print("Mask unique values:", np.unique(mask))
 
# -- Visualizations --
 
# Overlay mask on original image
segmenter.visualize_mask_overlay(image_path, mask)
 
# Visualize masked image
segmenter.visualize_masked_image(image_path, mask)