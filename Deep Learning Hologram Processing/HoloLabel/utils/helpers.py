# utils/helpers.py
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QRectF

from PIL import Image
from utils.reconstruction import Reconstruction
import numpy as np
import torch
import random

def prepare_dataset(data, config):
    # Implement dataset preparation (e.g., PyTorch Dataset)
    pass

def load_unlabeled_images(images_dir):
    # Load images from the specified directory
    pass

def save_samples_for_labeling(samples, output_dir):
    # Save images and (optional) predictions to the output directory
    pass

def load_image_array(img_path):
    img = Image.open(img_path).convert('L')
    image_array = np.array(img, dtype=np.uint8)
    return image_array

def save_image_array(file, image):
    image_pil = Image.fromarray(image)
    image_pil.save(file)

def generate_rgb_stack_image(img_path):
        # Implement the method to generate RGB stack images
        image_array = load_image_array(img_path)

        # Create the Reconstruction object
        recon = Reconstruction(
            resolution=0.087,
            wavelength=405 / 1.33 / 1000,
            z_start=0,
            z_step=6,
            num_planes=3,
            im_x=image_array.shape[1],
            im_y=image_array.shape[0],
            shift_mean=105,
            shift_value=105
        )

        # Reconstruct the image
        image_tensor = torch.tensor(image_array, dtype=torch.float32)
        stack = recon.rec_3D_intensity(image_tensor)

        stack = stack.cpu().numpy()
        stack = (stack * 255).astype(np.uint8)

        return stack

def generate_focal_plane(image_path, resolution, wavelength, current_depth):
     # Convert the current image to a NumPy array
    img = Image.open(image_path).convert('L')
    image_array = np.array(img, dtype=np.uint8)

    # Create the Reconstruction object for the current depth
    recon = Reconstruction(
        resolution=resolution,
        wavelength=wavelength,
        z_start=current_depth,
        z_step=1,  # Not relevant since num_planes=1
        num_planes=1,
        im_x=image_array.shape[1],
        im_y=image_array.shape[0],
        shift_mean=105,    # Adjust based on your needs
        shift_value=105   # Adjust based on your needs
    )

    # Reconstruct the image at the current depth
    image_tensor = torch.tensor(image_array, dtype=torch.float32)
    image_slice = recon.rec_3D_intensity(image_tensor)[:,:,0]

    image_slice = image_slice.cpu().numpy()
    reconstructed_plane = (image_slice * 255).astype(np.uint8)

    return reconstructed_plane

def array_to_pixmap(image, RGB):
     # Convert NumPy array to bytes
    image_bytes = image.tobytes()
    height, width = image.shape[0:2]
    bytes_per_line = width

    if RGB:
        bytes_per_line *= 3
        qimage = QImage(image_bytes, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
    else:
        qimage = QImage(image_bytes, width, height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)

    return pixmap

def norm_to_pixel_rect(image_width, image_height, x_center_norm, y_center_norm, width_norm, height_norm):
    # Convert normalized coordinates to pixel coordinates
    x_center = x_center_norm * image_width
    y_center = y_center_norm * image_height
    width = width_norm * image_width
    height = height_norm * image_height
    x = x_center - width / 2
    y = y_center - height / 2
    rect = QRectF(x, y, width, height)

    return rect

def pixel_to_norm_rect(rect, width, height):
    # Convert bbox to normalized coordinates
    x_center_norm = (rect.x() + rect.width() / 2) / width
    y_center_norm = (rect.y() + rect.height() / 2) / height
    width_norm = rect.width() / width
    height_norm = rect.height() / height

    return x_center_norm, y_center_norm, width_norm, height_norm

def select_n_random_images(image_files, n):
    # Randomly select images
    if n > len(image_files):
        n = len(image_files)
    selected_images = random.sample(image_files, n)

    return selected_images
