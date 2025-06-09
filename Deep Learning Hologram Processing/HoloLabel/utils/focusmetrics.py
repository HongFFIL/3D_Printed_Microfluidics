# utils/focusmetrics.py

import torch
import numpy as np
import cv2
from scipy import signal
from skimage.measure import regionprops, label
from utils.reconstruction import Reconstruction
import pywt
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import find_peaks, savgol_filter
import time


def adjust_peak(intensity_proj, target_peak, subtraction_factor):
    """
    Adjust the intensity of the image to match the target peak.

    Parameters:
        intensity_proj (numpy.ndarray): The input image.
        target_peak (float): The desired peak intensity.
        subtraction_factor (float): The value to subtract from the current peak.

    Returns:
        numpy.ndarray: The adjusted image.
    """
    # Compute histogram
    hist, _ = np.histogram(intensity_proj, bins=256, range=(0, 255))
    # Find the current peak intensity
    current_peak = np.argmax(hist)
    intensity_proj = intensity_proj.astype(float)
    if current_peak > 0:
        bottom_divide = current_peak - subtraction_factor
        # Avoid division by zero
        bottom_divide = bottom_divide if bottom_divide != 0 else 1
        intensity_proj = (intensity_proj - subtraction_factor) * target_peak / bottom_divide
        intensity_proj = np.clip(intensity_proj, 0, 255)
    adjusted_image = intensity_proj.astype(np.uint8)
    return adjusted_image


def obtain_infocus_image(source_img, resolution, wavelength, z, img_size):
    """
    Reconstruct the image at a specific focal plane.

    Parameters:
        source_img (numpy.ndarray): The input hologram image.
        resolution (float): The pixel resolution in micrometers.
        wavelength (float): The wavelength of the light in micrometers.
        z (float): The depth (focal plane) to reconstruct.
        img_size (tuple): The size of the image (height, width).

    Returns:
        numpy.ndarray: The reconstructed image at the specified focal plane.
    """
    rec = Reconstruction(
        resolution=resolution,
        wavelength=wavelength,
        z_start=z,
        z_step=1,
        num_planes=1,
        im_x=img_size[1],
        im_y=img_size[0],
        shift_mean=105,
        shift_value=105,
    )

    image_tensor = torch.tensor(source_img, dtype=torch.float32)
    image_slice = rec.rec_3D_intensity(image_tensor)[:,:,0]

    image_slice = image_slice.cpu().numpy()
    reconstructed_plane = (image_slice * 255).astype(np.uint8)

    return reconstructed_plane


import matplotlib.pyplot as plt

def find_best_focus_plane(
    source_img,
    img_size,
    num_z_planes,
    min_depth,
    z_step,
    resolution,
    wavelength,
    graphs=False,
):
    """
    Find the best focus plane for the given image using multiple focus metrics.

    Parameters:
        source_img (numpy.ndarray): The cropped hologram image.
        img_size (tuple): The size of the image (height, width).
        num_z_planes (int): Number of depth planes to evaluate.
        min_depth (float): Minimum depth.
        z_step (float): Depth step size.
        phi (float): Constant related to the reconstruction.
        resolution (float): The pixel resolution in micrometers.
        wavelength (float): The wavelength of the light in micrometers.
        num_candidates (int): Number of candidate depths to consider.
        known_focal_plane (float): Optional known focal plane for comparison.

    Returns:
        dict: A dictionary containing the best in-focus images and depths for each focus metric.
    """
    # Get candidate depths
    z_plane_list = [min_depth + i * z_step for i in range(num_z_planes)]

    # Initialize variables to store focus measures
    # Initialize variables to store focus measures
    focus_measures = {}
    best_focus_measures = {}
    best_z_values = {}

    # Dictionary of focus metric functions
    focus_metrics = {
        'laplacian_variance': lambda img: cv2.Laplacian(img, cv2.CV_64F).var(),
        'tenengrad': tenengrad_variance,
        'brenner': brenner_focus_measure,
        'variance': variance_of_intensity,
        'sml': sum_modified_laplacian,
        'wavelet': wavelet_focus_measure,
    }

    # Initialize focus measures dictionary
    for metric in focus_metrics.keys():
        focus_measures[metric] = []
        best_focus_measures[metric] = -np.inf
        best_z_values[metric] = None

    # Iterate over depths
    for idx, z in enumerate(z_plane_list):
        # Reconstruct image at this depth
        infocus_image = obtain_infocus_image(
            source_img, resolution, wavelength, z, img_size
        )

        # Compute focus measures for all metrics
        for metric_name, metric_func in focus_metrics.items():
            fm = metric_func(infocus_image)
            focus_measures[metric_name].append(fm)

    # Scaling and Visualization
    if graphs:
        plt.figure(figsize=(10, 6))
        
    for metric_name in focus_metrics.keys():
        fm_values = np.array(focus_measures[metric_name])

        # Smooth the data using Savitzky-Golay filter
        fm_values_smooth = savgol_filter(fm_values, window_length=11, polyorder=3)

        # Find peaks
        peaks, properties = find_peaks(fm_values_smooth)  # Adjust parameters if needed

        # Get the values of the peaks
        peak_values = fm_values_smooth[peaks]

        # Find the peak with the greatest magnitude minimum (closest to zero but still minimum)
        try:
            min_peak_idx = np.argmin(np.abs(peak_values))  # Find the index of the greatest magnitude minimum peak
            best_peak_idx = peaks[min_peak_idx]
        except:
            continue

        # Get best z and focus measure
        best_z = z_plane_list[best_peak_idx]
        best_z_values[metric_name] = best_z

        # # Find peaks
        # peaks, properties = find_peaks(fm_values_smooth, prominence=0.1)  # Adjust prominence as needed

        # # Get the prominences of the peaks
        # prominences = properties['prominences']

        # # Find the peak with the maximum prominence
        # try:
        #     max_prominence_idx = np.argmax(prominences)
        #     best_peak_idx = peaks[max_prominence_idx]
        # except:
        #     continue

        # # Get best z and focus measure
        # best_z = z_plane_list[best_peak_idx]
        # best_z_values[metric_name] = best_z

        # Scale the focus measures for plotting
        scaler = MinMaxScaler()
        fm_values_scaled = scaler.fit_transform(fm_values.reshape(-1, 1)).flatten()
        focus_measures[metric_name] = fm_values_scaled


        # Get the scaled best focus measure value
        best_fm_scaled = fm_values_scaled[best_peak_idx]

        if graphs:
            # Plot the scaled data
            plt.plot(z_plane_list, fm_values_scaled, label=metric_name)

            # Plot the scatter point for the best focus measure
            plt.scatter(
                best_z,
                best_fm_scaled,
                marker='o',
                s=100,
                label=f"{metric_name} best z",
            )

    if graphs:
        plt.xlabel('Depth (z)')
        plt.ylabel('Scaled Focus Measure')
        plt.title('Focus Measure vs. Depth for Different Metrics')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Return results
    results = {
        'best_z_values': best_z_values,
        'focus_measures': focus_measures,
        'z_plane_list': z_plane_list
    }

    return results


def tenengrad_variance(image):
    # Compute gradients using Sobel operator
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    # Compute gradient magnitude
    mag = np.sqrt(gx**2 + gy**2)
    # Compute focus measure as variance of gradient magnitude
    fm = mag.var()
    return fm

def brenner_focus_measure(image, d=2):
    # Shift image by d pixels
    shifted = np.roll(image, -d, axis=0)
    # Compute difference
    diff = image[:-d, :] - shifted[:-d, :]
    # Compute focus measure
    fm = np.sum(diff**2)
    return fm

def variance_of_intensity(image):
    fm = image.var()
    return fm

def sum_modified_laplacian(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    sml = np.sum(np.abs(laplacian))
    return sml

def wavelet_focus_measure(image):
    coeffs = pywt.wavedec2(image, 'db2', level=1)
    # Sum of the absolute values of the detail coefficients
    fm = sum(np.sum(np.abs(detail_coeff)) for detail_coeff in coeffs[1:])
    return fm
