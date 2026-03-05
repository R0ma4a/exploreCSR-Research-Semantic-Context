import depth_anything

image_path = r"D:\Research Projects\exploreCSR-Research-Semantic-Context\RGed-research\imgs\obj00004\img00003.jpeg"
weights = r"D:\Research Projects\exploreCSR-Research-Semantic-Context\RGed-research\checkpoints\depth_anything_v2_vitb.pth"

depthAnything = depth_anything.DepthAnything(weights)

image_tensor, rgb_image, original_width, original_height = depthAnything.image_to_tensor(image_path)

depth = depthAnything.predict_depth(image_tensor)

depth_norm = depthAnything.process_depth(depth, original_width, original_height)

depthAnything.visualize_depth(depth_norm)

depthAnything.save_depth(depth_norm, r"D:\Research Projects\exploreCSR-Research-Semantic-Context\RGed-research\imgs\\obj00004\\", "img00003")


