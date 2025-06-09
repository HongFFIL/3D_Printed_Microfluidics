# utils/visualization.py

import cv2

def draw_bounding_boxes(image, boxes):
    for box in boxes:
        x_center, y_center, width, height = box['bbox']
        class_id = box['class_id']
        confidence = box['confidence']
        # Convert from YOLO format to pixel coordinates
        # Draw rectangles and labels on the image
    return image
