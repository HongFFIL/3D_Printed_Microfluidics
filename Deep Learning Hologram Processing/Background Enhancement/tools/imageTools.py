import math
import random
import cv2
import torch
import numpy as np
from tools.hologramFunctions.reconstruction import Reconstruction, adjustPeak
from tools.hologramFunctions.focusmetrics import FocusMetricTest
import os
import albumentations as A


class Enhancement:
    def __init__(self, imgsz=(576,768),stackSize=20, meanIntesity=105, keepInitial=False) -> None:
        self.BK_COUNT = 0
        self.BK_NUM = stackSize
        self.SCALE_FACTOR = 2
        self.MEAN_INTENSITY = meanIntesity
        self.KEEP_INITIAL = keepInitial

        self.imgsz = imgsz

        self.BK_STACK = torch.zeros(
            [self.BK_NUM, self.imgsz[1], self.imgsz[0]],
            dtype=torch.float,
            device=torch.device("cuda"),
        )

    def movingWindowEnhance(self, image, toCPU=False):
        if type(image) is np.ndarray:
            image = torch.tensor(image, device="cuda")

        if self.BK_COUNT < self.BK_NUM:
            self.BK_STACK[self.BK_COUNT, :, :] = image
            self.BK_COUNT += 1
            enh_gpu = torch.zeros((self.imgsz[1], self.imgsz[0]), device="cuda")

        else:
            self.BK_STACK = torch.roll(self.BK_STACK, -1, 0)
            self.BK_STACK[-1, :, :] = image

            enh_gpu = torch.abs(
                (self.BK_STACK[-1, :, :] - torch.mean(self.BK_STACK, axis=0))
                * self.SCALE_FACTOR
                + self.MEAN_INTENSITY
            )

        # enh_gpu = torch.stack((enh_gpu,) * 3, axis=-1)

        if toCPU:
            enh_gpu = enh_gpu.cpu().numpy()
            enh_gpu = enh_gpu.astype(np.uint8)

        return enh_gpu
    
    def movingWindowEnhanceMedian(self, image, toCPU=False):
        if type(image) is np.ndarray:
            image = torch.tensor(image, device="cuda")

        if self.BK_COUNT < self.BK_NUM:
            self.BK_STACK[self.BK_COUNT, :, :] = image
            self.BK_COUNT += 1
            enh_gpu = torch.zeros((self.imgsz[1], self.imgsz[0]), device="cuda")

        else:
            self.BK_STACK = torch.roll(self.BK_STACK, -1, 0)
            self.BK_STACK[-1, :, :] = image

            # Use median for background calculation
            median_background = torch.median(self.BK_STACK, dim=0).values

            enh_gpu = torch.abs(
                (self.BK_STACK[-1, :, :] - median_background)
                * self.SCALE_FACTOR
                + self.MEAN_INTENSITY
            )

        if toCPU:
            enh_gpu = enh_gpu.cpu().numpy()
            enh_gpu = enh_gpu.astype(np.uint8)

        return enh_gpu
    
    def movingWindowEnhanceAdaptiveThreshold(self, image, toCPU=False, threshold_percentile=95):
        if type(image) is np.ndarray:
            image = torch.tensor(image, device="cuda")

        if self.BK_COUNT < self.BK_NUM:
            self.BK_STACK[self.BK_COUNT, :, :] = image
            self.BK_COUNT += 1
            enh_gpu = torch.zeros((self.imgsz[1], self.imgsz[0]), device="cuda")

        else:
            self.BK_STACK = torch.roll(self.BK_STACK, -1, 0)
            self.BK_STACK[-1, :, :] = image

            # Compute the adaptive threshold
            current_image = self.BK_STACK[-1, :, :]
            threshold_value = torch.quantile(current_image, threshold_percentile / 100.0)

            # Create a mask for high-intensity areas
            mask = current_image > threshold_value

            # Apply the mask to the background stack
            masked_stack = self.BK_STACK.clone()
            masked_stack[:, mask] = float('nan')  # Exclude high-intensity areas

            # Calculate the background using the mean while ignoring NaN values
            background = torch.nanmean(masked_stack, dim=0)

            # Replace NaNs with zeros for computation safety
            background[torch.isnan(background)] = 0

            # Enhance the image
            enh_gpu = torch.abs(
                (current_image - background) * self.SCALE_FACTOR + self.MEAN_INTENSITY
            )

        if toCPU:
            enh_gpu = enh_gpu.cpu().numpy()
            enh_gpu = enh_gpu.astype(np.uint8)

        return enh_gpu



def adjust_bbox(bbox):
    center_x, center_y, width, height = bbox
    
    # Clamp the width and height
    width = max(0.0, min(width, 1.0))
    height = max(0.0, min(height, 1.0))
    
    # Adjust the center coordinates to stay within [0, 1]
    center_x = min(max(center_x, 0.0), 1.0)
    center_y = min(max(center_y, 0.0), 1.0)
    
    # Ensure bounding box does not exceed image boundaries
    # This makes sure that if, for example, the center is at 0.95 and the width is 0.2, 
    # the resulting bounding box does not exceed the right boundary.
    width = min(width, 2.0 * (1.0 - center_x), 2.0 * center_x)
    height = min(height, 2.0 * (1.0 - center_y), 2.0 * center_y)

    return [center_x, center_y, width, height]

def augment(input_folder, save_folder, num_augmentations=3, augmentPercent=1.0):
    augmentations = A.Compose([
        A.Rotate(limit=10, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=1.0),
        A.ISONoise(color_shift=(0.1, 0.2), intensity=(0.5, 1.0), p=0.5),
        A.GridDistortion(num_steps=10, distort_limit=0.1, p=0.3),
        A.Blur(blur_limit=2, p=0.5)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    # Create directories for saving images and labels
    save_images_folder = os.path.join(save_folder, "images")
    save_labels_folder = os.path.join(save_folder, "labels")
    os.makedirs(save_images_folder, exist_ok=True)
    os.makedirs(save_labels_folder, exist_ok=True)

    image_files = [f for f in os.listdir(f"{input_folder}/images/") if os.path.isfile(os.path.join(f"{input_folder}/images/", f)) and f.endswith(('.png', '.jpg', '.jpeg'))]

    for img_file in image_files:
        print(img_file)
        if random.random() > augmentPercent:
            continue

        image_path = os.path.join(f"{input_folder}/images/", img_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label_path = f"{input_folder}/labels/{os.path.splitext(img_file)[0]}.txt"
        
        if not os.path.exists(label_path):
            print(f"Warning: No label file found for {img_file}. Creating an empty label.")
            open(label_path, 'a').close()  # This will create an empty file if it doesn't exist

        with open(label_path, 'r') as file:
            labels = file.readlines()

        bboxes = []
        class_labels = []
        for label in labels:
            parts = label.strip().split()
            class_label = int(parts[0])
            bbox = [float(part) for part in parts[1:]]
            bbox = adjust_bbox(bbox)
            bboxes.append(bbox)
            class_labels.append(class_label)

        for j in range(num_augmentations):
            augmented = augmentations(image=image, bboxes=bboxes, class_labels=class_labels)
            augmented_image = augmented['image']
            augmented_bboxes = augmented['bboxes']

            augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)

            # Save augmented image to the images sub-folder
            output_image_path = os.path.join(save_images_folder, f"{j}_{img_file}")
            cv2.imwrite(output_image_path, augmented_image)

            # Save augmented labels to the labels sub-folder
            output_label_path = os.path.join(save_labels_folder, f"{j}_{os.path.splitext(img_file)[0]}.txt")
            with open(output_label_path, 'w') as file:
                for k, bbox in enumerate(augmented_bboxes):
                    line = f"{class_labels[k]} {' '.join(map(str, bbox))}\n"
                    file.write(line)


def segmentBatchImage(base_img, show=False):
    if show:
        img_with_borders = base_img.copy()

    img_size = base_img.shape

    height_segments = math.ceil(img_size[0] / 640)
    width_segments = math.ceil(img_size[1] / 640)

    cross_img_size = (img_size[0] - 640 * 2, img_size[1] - 640 * 2)
    cross_height_segments = math.ceil(cross_img_size[0] / 640)
    cross_width_segments = math.ceil(cross_img_size[1] / 640)

    # Calculate step size for the middle segments
    height_step = cross_img_size[0] // max(1, cross_height_segments)
    width_step = cross_img_size[1] // max(1, cross_width_segments)

    crops = []
    for i in range(height_segments):
        for j in range(width_segments):
            if i == 0:  # top row
                start_y = 0
            elif i == height_segments - 1:  # bottom row
                start_y = img_size[0] - 640
            else:  # middle rows
                start_y = height_step * i + 640 // 2 - height_step // 2

            if j == 0:  # left column
                start_x = 0
            elif j == width_segments - 1:  # right column
                start_x = img_size[1] - 640
            else:  # middle columns
                start_x = width_step * j + 640 // 2 - width_step // 2

            crops.append(base_img[start_y : start_y + 640, start_x : start_x + 640])

            # Draw a rectangle on the image to represent the border of the segment
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
            if show:
                cv2.rectangle(
                    img_with_borders,
                    (start_x, start_y),
                    (start_x + 640, start_y + 640),
                    color,
                    4,
                )

    if show:
        scale = 4
        img_with_borders = cv2.resize(
            img_with_borders, (img_size[1] // scale, img_size[0] // scale)
        )
        cv2.imshow("Image with Borders", img_with_borders)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    img_batch = loadImagesGPU(crops)

    return img_batch


def loadImagesGPU(images, reshape=(640, 640)):
    if not type(images) == list:
        images = [images]

    tensor_imgs = []
    for img in images:
        img = cv2.resize(img, reshape)
        img = np.ascontiguousarray(img)
        img_tensor = (
            torch.from_numpy(img)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(torch.device("cuda"))
            .float()
        )
        tensor_imgs.append(img_tensor)

    img_batch = torch.cat(tensor_imgs, dim=0)

    return img_batch

def cropToBB(image, bb, size=100):
    x1, y1, x2, y2 = bb.astype(int)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    xmin = max(0, int(cx - size/2))
    xmax = min(768, int(cx + size/2))

    ymin = max(0, int(cy - size/2))
    ymax = min(576, int(cy + size/2))

    crop = image[ymin:ymax, xmin:xmax]
    return crop

def tripleStack(image, zStart = 6, z_step=6, auto_focus=False, wl=405, n=1.33, res=0.087):
    if auto_focus:
        num_planes = 1
        z_start, _, _ = FocusMetricTest(image, 1, 10, 405, 0.089, 0, 5)
        print(z_start)
    else:
        num_planes = 2
        z_start = zStart

    imgsz  = (image.shape[0], image.shape[1])
    # image = torch.tensor(image, device="cuda")
    recon = Reconstruction(
            resolution=0.087,
            wavelength=405 / 1.33 / 1000,
            z_start=z_start,
            z_step= z_step,
            num_planes=num_planes,
            im_x=imgsz[1],
            im_y=imgsz[0],
            shift_mean=70,
            shift_value=105, # 105
        )
    
    stack = recon.rec_3D_intensity(image)

    stack = stack * 255
    image_3d = image.unsqueeze(-1)
    # combined_stack = torch.cat((image_3d, stack), dim=-1)

    # Move to CPU and convert to a NumPy array
    # stack = combined_stack.cpu().numpy().astype(np.uint8)
    # stack = stack[:,:,2]

    return stack
