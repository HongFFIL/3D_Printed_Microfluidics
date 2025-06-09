# data_processing/data_augmentation.py

import albumentations as A

def augment_data(data):
    augmentations = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=45, p=0.5),
        # Add more augmentations as needed
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    augmented_data = []
    for item in data:
        transformed = augmentations(
            image=item['image'],
            bboxes=item['labels'],
            class_labels=[label[0] for label in item['labels']]
        )
        augmented_data.append({
            'image': transformed['image'],
            'labels': transformed['bboxes'],
            'class_labels': transformed['class_labels']
        })
    return augmented_data
