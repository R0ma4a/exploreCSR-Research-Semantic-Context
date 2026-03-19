import depth_anything
import dino
import numpy as np

# ------------------------------------------------
# Paths
# ------------------------------------------------
# Use absolute paths so this script works regardless of the
# current working directory when it is invoked.
image_path = r"C:\Users\roman\Git Projects\BrownCSR\SimpleRecon\Work_dir\Input_IMGs\Related_IMGs_4\PXL_20260303_211335449.jpg"
weights = r"C:\Users\roman\Downloads\depth_anything_v2_vitb.pth"

# -- Initialize Models --
converter = depth_anything.DepthAnything(weights)
segmenter = dino.dino()

# -- Preprocess Image --
image_tensor, rgb_image, original_width, original_height = converter.image_to_tensor(image_path)

# -- Generate Depth Map --
depth = converter.predict_depth(image_tensor)
depth_norm = converter.process_depth(depth, original_width, original_height)

# -- Generate Object Mask --
mask = segmenter.generate_object_mask(
    image_tensor,
    depth_norm,
    (original_height, original_width)
)

# -- Debug check --
print("Mask unique values:", np.unique(mask))

# -- Visualizations --

# Overlay mask on original image
segmenter.visualize_mask_overlay(image_path, mask)

# Visualize masked image
segmenter.visualize_masked_image(image_path, mask)





