import os
import random
import cv2
import numpy as np

def load_bounding_boxes(file_path):
    bounding_boxes = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            label, center_x, center_y, width, height = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            bounding_boxes.append((label, center_x, center_y, width, height))
    return bounding_boxes

def create_circular_blending_mask(cell):
    h, w = cell.shape[:2]
    radius = min(h, w) // 2 - 12  # Slightly smaller to avoid edges
    mask = np.zeros((h, w), dtype=np.float32)
    center = (w // 2, h // 2)
    cv2.circle(mask, center, radius, 1, -1)  # Inner circle
    # mask = cv2.GaussianBlur(mask, (15, 15), 5)  # Smooth edges
    return mask

def extract_yeast_cells(image, bounding_boxes):
    h, w = image.shape[:2]
    yeast_cells = []
    existing_boxes = []

    def check_original_overlap(new_box):
        nx1, ny1, nx2, ny2 = new_box
        for ex1, ey1, ex2, ey2 in existing_boxes:
            if not (nx2 <= ex1 or nx1 >= ex2 or ny2 <= ey1 or ny1 >= ey2):
                return True
        return False

    for box in bounding_boxes:
        label, center_x, center_y, box_width, box_height = box
        x1 = max(0, int((center_x - box_width / 2) * w) - 25)
        x2 = min(w, int((center_x + box_width / 2) * w) + 25)
        y1 = max(0, int((center_y - box_height / 2) * h) - 25)
        y2 = min(h, int((center_y + box_height / 2) * h) + 25)

        if center_x < 100 / w or center_x > (w - 100) / w:
            continue
        if not (330 <= int(center_y * h) <= 880):
            continue

        updated_box = (x1, y1, x2, y2)
        if check_original_overlap(updated_box):
            continue

        original_box = (x1, y1, x2, y2)
        existing_boxes.append(original_box)
        yeast_cells.append((label, image[y1:y2, x1:x2], box_width, box_height, original_box, center_y))

    return yeast_cells

def load_yeast_cells_from_folders(class_folders):
    class_cells = {"0": [], "1": []}
    for image_folder, labels_folder in class_folders:
        for label_file in os.listdir(labels_folder):
            label_path = os.path.join(labels_folder, label_file)
            image_file = label_file.replace('.txt', '.jpg')
            image_path = os.path.join(image_folder, image_file)
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                bounding_boxes = load_bounding_boxes(label_path)
                cells = extract_yeast_cells(image, bounding_boxes)
                for cell in cells:
                    label = cell[0]
                    class_cells[label].append(cell)
    return class_cells

def blend_cells_onto_background(background, class_cells):
    blended_image = background.copy()
    h, w = blended_image.shape[:2]
    new_labels = []
    existing_boxes = []

    # Shuffle cells to avoid grouping augmented versions of the same cell
    random.shuffle(class_cells["0"])
    random.shuffle(class_cells["1"])

    num_cells_to_blend_0 = random.randint(1, 3)
    num_cells_to_blend_1 = random.randint(1, 3)

    # Balance classes
    max_cells_class_0 = len(class_cells["0"])
    max_cells_class_1 = len(class_cells["1"])
    max_cells_0 = min(max_cells_class_0, max_cells_class_1, num_cells_to_blend_0)
    max_cells_1 = min(max_cells_class_0, max_cells_class_1, num_cells_to_blend_1)

    selected_class_0 = [class_cells["0"].pop() for _ in range(min(max_cells_0, max_cells_class_0))]
    selected_class_1 = [class_cells["1"].pop() for _ in range(min(max_cells_1, max_cells_class_1))]
    selected_cells = selected_class_0 + selected_class_1

    for cell_data in selected_cells:
        label, cell, orig_width, orig_height, original_box, original_center_y = cell_data
        cell_h, cell_w = cell.shape[:2]

        for _ in range(100):
            x_offset = random.randint(20, w - cell_w - 20)
            y_offset = max(0, min(h - cell_h, int(original_center_y * h) - cell_h // 2))

            if not any((x_offset < eb[2] and x_offset + cell_w > eb[0] and y_offset < eb[3] and y_offset + cell_h > eb[1]) for eb in existing_boxes):
                existing_boxes.append((x_offset, y_offset, x_offset + cell_w, y_offset + cell_h))

                # Create a larger circular mask for blending with alpha edges
                radius = max(cell_h, cell_w) // 2 - 12
                center = (cell_w // 2, cell_h // 2)

                # mask = create_circular_blending_mask(cell)

                # # Apply Gaussian blur to the circular mask for smooth blending
                # mask = cv2.GaussianBlur(mask, (15, 15), 5)
                # mask = np.clip(mask, 0, 1)

                #############################################
                mask = create_circular_blending_mask(cell)

                # Define inner blending region
                alpha_radius = radius - 10
                if alpha_radius > 0:
                    mask_alpha = np.zeros_like(mask)
                    cv2.circle(mask_alpha, center, alpha_radius, 1, -1)

                    # Combine inner and outer masks
                    mask = cv2.addWeighted(mask, 0.5, mask_alpha, 0.5, 0)

                # Apply Gaussian blur for smooth edges
                mask = cv2.GaussianBlur(mask, (15, 15), 5)
                mask = np.clip(mask, 0, 1)
                #############################################

                mask_inv = 1 - mask

                # Extract the region of interest
                roi = blended_image[y_offset:y_offset + cell_h, x_offset:x_offset + cell_w]
                roi = roi.astype(np.float32) / 255
                cell = cell.astype(np.float32) / 255

                # Apply circular blending
                background_part = roi * mask_inv[..., None]
                cell_part = cell * mask[..., None]
                blended_roi = cv2.add(background_part, cell_part)
                blended_roi = np.clip(blended_roi * 255, 0, 255).astype(np.uint8)
                blended_image[y_offset:y_offset + cell_h, x_offset:x_offset + cell_w] = blended_roi

                # Calculate new bounding box coordinates and retain original width/height
                new_center_x = (x_offset + cell_w / 2) / w
                new_center_y = (y_offset + cell_h / 2) / h
                new_labels.append(f"{label} {new_center_x:.6f} {new_center_y:.6f} {orig_width:.6f} {orig_height:.6f}")
                break

    return blended_image, new_labels

def create_blended_images(background_path, class_folders, output_image_folder, output_label_folder, max_images):
    background = cv2.imread(background_path)
    class_cells = load_yeast_cells_from_folders(class_folders)
    image_count = 0

    while image_count < max_images:
        blended_image, blended_labels = blend_cells_onto_background(background, class_cells)
        if not blended_labels:
            print("No more valid cells to create images.")
            break

        output_image_path = os.path.join(output_image_folder, f"blended_med_{image_count + 1}.jpg")
        output_label_path = os.path.join(output_label_folder, f"blended_med_{image_count + 1}.txt")

        cv2.imwrite(output_image_path, blended_image)
        with open(output_label_path, 'w') as f:
            f.write('\n'.join(blended_labels))

        image_count += 1

    print(f"Created {image_count} blended images.")

# Paths
background_path = '/home/hong_data1/Documents/Delaney/background.jpg'
class_folders = [
    ('/home/hong_data1/Documents/Delaney/yeast_viability/live/focus_med/blend_images/', '/home/hong_data1/Documents/Delaney/yeast_viability/live/focus_med/blend_labels/'),
    ('/home/hong_data1/Documents/Delaney/yeast_viability/oven/focus_med_blend/images/', '/home/hong_data1/Documents/Delaney/yeast_viability/oven/focus_med_blend/labels/')
]  

# Replace with actual paths
output_image_folder = '/home/hong_data1/Documents/Delaney/yeast_viability/liveoven_blend/med/images/'  # Replace with actual path
output_label_folder = '/home/hong_data1/Documents/Delaney/yeast_viability/liveoven_blend/med/labels/'  # Replace with actual path
max_images = 400

# Create output directories
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_label_folder, exist_ok=True)

# Create blended images
create_blended_images(background_path, class_folders, output_image_folder, output_label_folder, max_images)
