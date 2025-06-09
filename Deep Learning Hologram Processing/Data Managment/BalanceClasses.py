import os
import random
import shutil

def load_class_files(label_folder, target_class):
    """Loads all label files and checks if they contain the target class."""
    class_files = []

    for label_file in os.listdir(label_folder):
        if label_file.endswith('.txt'):
            label_path = os.path.join(label_folder, label_file)
            with open(label_path, 'r') as f:
                lines = f.readlines()
                if all(line.startswith(f'{target_class} ') for line in lines):
                    class_files.append(os.path.splitext(label_file)[0])
    return class_files

def move_files(files_to_move, image_folder, label_folder, unused_image_folder, unused_label_folder):
    """Moves selected files to the unused folders."""
    os.makedirs(unused_image_folder, exist_ok=True)
    os.makedirs(unused_label_folder, exist_ok=True)

    for file_base in files_to_move:
        image_path = os.path.join(image_folder, f"{file_base}.jpg")
        label_path = os.path.join(label_folder, f"{file_base}.txt")

        if os.path.exists(image_path):
            shutil.move(image_path, os.path.join(unused_image_folder, f"{file_base}.jpg"))

        if os.path.exists(label_path):
            shutil.move(label_path, os.path.join(unused_label_folder, f"{file_base}.txt"))

def main():
    image_folder = "/home/hong_data1/Documents/Delaney/yeast_viability/live/focus_large/images/"
    label_folder = "/home/hong_data1/Documents/Delaney/yeast_viability/live/focus_large/labels/"

    # Load all files containing only class '0'
    class_0_files = load_class_files(label_folder, target_class=0)

    # Randomly select 50% of these files
    selected_files = random.sample(class_0_files, int(float(len(class_0_files) * 0.05)))

    # Move the selected files to the unused folders
    move_files(
        selected_files,
        image_folder,
        label_folder,
        unused_image_folder="/home/hong_data1/Documents/Delaney/yeast_viability/live/focus_large/un_images/",
        unused_label_folder="/home/hong_data1/Documents/Delaney/yeast_viability/live/focus_large/un_labels/"
    )

    print(f"Moved {len(selected_files)} images and corresponding label files to unused_live_images and unused_live_labels.")

if __name__ == "__main__":
    main()