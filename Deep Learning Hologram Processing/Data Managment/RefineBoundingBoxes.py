import os
from collections import defaultdict

# Removes overlapping predictions and counts bounding boxes

def load_text_files(folder_path):
    """Loads text files from a folder."""
    text_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    data = {}
    for file in text_files:
        file_path = os.path.join(folder_path, file)
        with open(file_path, 'r') as f:
            data[file] = [list(map(float, line.strip().split())) for line in f]
    return data

def calculate_iou(box1, box2):
    """Calculates the Intersection over Union (IoU) of two bounding boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate the coordinates of the intersection rectangle
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    # Calculate area of intersection rectangle
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    intersection_area = inter_width * inter_height

    # Calculate area of both bounding boxes
    box1_area = w1 * h1
    box2_area = w2 * h2

    # Calculate union area
    union_area = box1_area + box2_area - intersection_area

    # Return IoU
    return intersection_area / union_area if union_area != 0 else 0

def filter_labels(data, iou_threshold=0.95, image_width=1080):
    """Filters out double labels based on IoU threshold, keeping the one with the highest confidence.
       Also filters out boxes near the image edges."""
    filtered_data = {}
    for file, entries in data.items():
        sorted_entries = sorted(entries, key=lambda x: -x[-1])  # Sort by confidence descending
        unique_boxes = []
        for label, x, y, w, h, confidence in sorted_entries:
            # Check if the center of the box is within 30 pixels of the image edge
            if x < 30 / image_width or x > (image_width - 30) / image_width:
                continue

            bbox = (x, y, w, h)
            is_duplicate = any(
                calculate_iou(bbox, (ux, uy, uw, uh)) >= iou_threshold
                for _, ux, uy, uw, uh, _ in unique_boxes
            )
            if not is_duplicate:
                unique_boxes.append((label, x, y, w, h, confidence))
        filtered_data[file] = unique_boxes
    return filtered_data

def save_filtered_labels(filtered_data, output_folder):
    """Saves the filtered labels to the specified output folder."""
    os.makedirs(output_folder, exist_ok=True)
    for file, entries in filtered_data.items():
        output_file = os.path.join(output_folder, file)
        with open(output_file, 'w') as f:
            for label, x, y, w, h, confidence in entries:
                f.write(f"{int(label)} {x} {y} {w} {h} {confidence}\n")
                # f.write(f"{int(label)} {x} {y} {w} {h}\n")
                

def count_labels(filtered_data):
    """Counts labels in the filtered data."""
    label_counts = defaultdict(int)
    for entries in filtered_data.values():
        for label, _, _, _, _, _ in entries:
            label_counts[label] += 1
    return label_counts

def main(folder_path):
    output_folder = "/home/hong_data1/Documents/HoloLabel/datasets/prediction4/high_conf/updated_labels/"
    data = load_text_files(folder_path)
    filtered_data = filter_labels(data)
    save_filtered_labels(filtered_data, output_folder)
    label_counts = count_labels(filtered_data)

    # Print results
    print("Label Counts:")
    for label, count in label_counts.items():
        print(f"Label {int(label)}: {count}")

if __name__ == "__main__":
    folder_path = "/home/hong_data1/Documents/HoloLabel/datasets/prediction4/high_conf/labels/"
    main(folder_path)
