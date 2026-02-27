import depth_anything

image_path = r"D:\Research Projects\exploreCSR-Research-Semantic-Context\RGed-research\imgs\test_img.jpeg"

rgbd = depth_anything.DepthAnything()
image_tensor, rgb_image, original_width, original_height = rgbd.image_to_tensor(image_path)
depth = rgbd.predict_depth(image_tensor)
rgbd_image, depth_norm = rgbd.process_rgbd(depth, rgb_image, original_width, original_height)
rgbd.visualize_depth(depth_norm)
rgbd.save_rgbd(rgbd_image, "D:\\Research Projects\\exploreCSR-Research-Semantic-Context\\RGed-research\\imgs\\", "test_img_rgbd")
print(rgbd_image.shape, rgbd_image.dtype, rgbd_image.min(), rgbd_image.max())