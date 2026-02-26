import os
import cv2
import torch

def upload_image(img_path):
    """Load an image safely with OpenCV and convert to RGB."""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Failed to read image: {img_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def normalize_image_to_tensor(img):
    """Normalize image to [0,1] and convert to PyTorch tensor [1,3,H,W]."""
    img = img / 255.0
    img_tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float()
    return img_tensor

def test_image_encoding(img_path):
    """Quick test to ensure image loads correctly."""
    img = upload_image(img_path)
    print("Image loaded successfully:", img.shape)
    tensor = normalize_image_to_tensor(img)
    print("Tensor shape:", tensor.shape)

