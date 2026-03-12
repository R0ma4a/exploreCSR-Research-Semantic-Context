import depth_anything
import dino

image_path = r"D:\Research Projects\exploreCSR-Research-Semantic-Context\RGed-research\imgs\obj00004\img00003.jpeg"
weights = r"D:\Research Projects\exploreCSR-Research-Semantic-Context\RGed-research\checkpoints\depth_anything_v2_vitb.pth"

# -- Initialize DepthAnything and DINO Models --
converter = depth_anything.DepthAnything(weights)
segmenter = dino.dino()

# -- Image Preprocessing --
image_tensor, rgb_image, original_width, original_height = converter.image_to_tensor(image_path)

# -- RGB Depth Prediction --
depth = converter.predict_depth(image_tensor)
depth_norm = converter.process_depth(depth, original_width, original_height)

# -- OPTIONAL: Visualize and Save --
# depthAnything.visualize_depth(depth_norm)
# depthAnything.save_depth(depth_norm, r"D:\Research Projects\exploreCSR-Research-Semantic-Context\RGed-research\imgs\\obj00004\\", "img00003")


# -- DINOv3 Feature Extraction --
features = segmenter.extract_features(image_tensor)
patch_grid = segmenter.process_patch_tokens(features)
patch_features_np, H, W = segmenter.prepare_features_for_clustering(patch_grid)





