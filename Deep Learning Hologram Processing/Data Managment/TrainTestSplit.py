import os
import shutil
import random

def split_dataset(image_folder, label_folder, output_folder, train_ratio=0.8, valid_ratio=0.2, test_ratio=0.0):
    """
    Split images and labels into train, valid, and test sets.

    Args:
        image_folder (str): Path to the folder containing images.
        label_folder (str): Path to the folder containing labels.
        output_folder (str): Path to the output folder for train, valid, and test sets.
        train_ratio (float): Proportion of data to include in the training set.
        valid_ratio (float): Proportion of data to include in the validation set.
        test_ratio (float): Proportion of data to include in the test set.
    """
    # assert train_ratio + valid_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"

    # Ensure output directories exist
    train_image_folder = os.path.join(output_folder, "train", "images")
    train_label_folder = os.path.join(output_folder, "train", "labels")
    valid_image_folder = os.path.join(output_folder, "valid", "images")
    valid_label_folder = os.path.join(output_folder, "valid", "labels")
    test_image_folder = os.path.join(output_folder, "test", "images")
    test_label_folder = os.path.join(output_folder, "test", "labels")

    os.makedirs(train_image_folder, exist_ok=True)
    os.makedirs(train_label_folder, exist_ok=True)
    os.makedirs(valid_image_folder, exist_ok=True)
    os.makedirs(valid_label_folder, exist_ok=True)
    os.makedirs(test_image_folder, exist_ok=True)
    os.makedirs(test_label_folder, exist_ok=True)

    # Get list of image files
    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
    random.shuffle(image_files)

    # Calculate split sizes
    total_files = len(image_files)
    train_count = int(total_files * train_ratio)
    valid_count = int(total_files * valid_ratio)

    # Split the data
    train_files = image_files[:train_count]
    valid_files = image_files[train_count:train_count + valid_count]
    test_files = image_files[train_count + valid_count:]

    def move_files(file_list, dest_image_folder, dest_label_folder):
        for image_file in file_list:
            file_name, _ = os.path.splitext(image_file)
            label_file = file_name + ".txt"

            # Move the image file
            src_image_path = os.path.join(image_folder, image_file)
            dest_image_path = os.path.join(dest_image_folder, image_file)
            shutil.copy(src_image_path, dest_image_path)

            # Move the label file
            src_label_path = os.path.join(label_folder, label_file)
            if os.path.exists(src_label_path):
                dest_label_path = os.path.join(dest_label_folder, label_file)
                shutil.copy(src_label_path, dest_label_path)

    # Move files to respective folders
    move_files(train_files, train_image_folder, train_label_folder)
    move_files(valid_files, valid_image_folder, valid_label_folder)
    move_files(test_files, test_image_folder, test_label_folder)

    print(f"Dataset split completed. Train: {len(train_files)}, Valid: {len(valid_files)}, Test: {len(test_files)}")

# Example usage
image_folder = '/home/hong_data1/Documents/Delaney/yeast_viability/liveoven_augment/small/images/'
label_folder = '/home/hong_data1/Documents/Delaney/yeast_viability/liveoven_augment/small/labels/'
output_folder = "/home/hong_data1/Documents/Delaney/yeast_viability/model_data_2/"

split_dataset(image_folder, label_folder, output_folder)
