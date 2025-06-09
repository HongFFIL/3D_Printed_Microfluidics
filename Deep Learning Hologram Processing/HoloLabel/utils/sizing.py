import numpy as np
import cv2
from utils.reconstruction import Reconstruction
from skimage.measure import regionprops, label
import torch
from segment_anything import sam_model_registry, SamPredictor

def setup_predictor():
    sam_checkpoint = "./sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return SamPredictor(sam)

def crop_particle(image_array, rect):
    x = int(rect.x())
    y = int(rect.y())
    width = int(rect.width())
    height = int(rect.height())
    cropped_image = image_array[y:y+height, x:x+width]
    return cropped_image

def reconstruct_at_focal_plane(cropped_image, best_z, reconstruction_params):
    rec = Reconstruction(
        resolution=reconstruction_params['resolution'],
        wavelength=reconstruction_params['wavelength'],
        z_start=best_z,
        z_step=1,
        num_planes=1,
        im_x=cropped_image.shape[1],
        im_y=cropped_image.shape[0],
        shift_mean=105,
        shift_value=105
    )

    image_tensor = torch.tensor(cropped_image, dtype=torch.float32)
    image_slice = rec.rec_3D_intensity(image_tensor)[:,:,0]

    image_slice = image_slice.cpu().numpy()
    reconstructed_plane = (image_slice * 255).astype(np.uint8)
    return reconstructed_plane

def segment_particle(reconstructed_image, predictor):
    predictor.set_image(cv2.cvtColor(reconstructed_image, cv2.COLOR_GRAY2BGR))

    # Use the center point as the input point
    height, width = reconstructed_image.shape
    center_x, center_y = width // 2, height // 2
    input_point = np.array([[center_x, center_y]])
    input_label = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    mask = masks[0].astype(np.uint8)
    return mask

def calculate_size_metrics(mask, pixel_size, focal_plane):
    # Compute properties of the segmented region
    labeled_mask = label(mask)
    regions = regionprops(labeled_mask)
    if not regions:
        return None  # No regions found

    region = regions[0]  # Assuming the largest region is the particle

    area = region.area * (pixel_size ** 2)
    perimeter = region.perimeter * pixel_size
    equivalent_diameter = region.equivalent_diameter * pixel_size
    major_axis_length = region.major_axis_length * pixel_size
    minor_axis_length = region.minor_axis_length * pixel_size
    eccentricity = region.eccentricity

    metrics = {
        'Area (µm²)': area,
        'Perimeter (µm)': perimeter,
        'Equivalent Diameter (µm)': equivalent_diameter,
        'Major Axis Length (µm)': major_axis_length,
        'Minor Axis Length (µm)': minor_axis_length,
        'Eccentricity': eccentricity,
        'Focal Plane (µm)': focal_plane
    }
    return metrics

