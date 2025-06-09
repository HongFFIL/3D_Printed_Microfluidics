# data_processing/data_loader.py

import os
import glob
import cv2

def load_images_labels(images_dir, labels_dir):
    image_paths = sorted(glob.glob(os.path.join(images_dir, '*.pgm')))
    label_paths = sorted(glob.glob(os.path.join(labels_dir, '*.txt')))

    data = []
    for img_path, lbl_path in zip(image_paths, label_paths):
        image = cv2.imread(img_path)
        labels = load_labels(lbl_path)
        data.append({'image': image, 'labels': labels})
    return data

def load_labels(label_path):
    labels = []
    with open(label_path, 'r') as file:
        for line in file:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            labels.append([class_id, x_center, y_center, width, height])
    return labels
