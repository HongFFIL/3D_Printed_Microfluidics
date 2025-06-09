from ultralytics import YOLO
from utils.helpers import generate_rgb_stack_image
import os
import glob
import shutil
import yaml
from PIL import Image
from tqdm import tqdm

class YoloTrainer:
    def __init__(self, model_config):
        self.model = YOLO(model_config)

    def simple_train(self, dataset_path, epochs, batch_size, patience, label_smoothing, STACK=True):
        # Prepare data.yaml path
        data_yaml = os.path.join(dataset_path, "data.yaml")
        if not os.path.exists(data_yaml):
            print(f"data.yaml not found in {dataset_path}.")
            return

        # Generate temporary dataset with RGB stacked images
        temp_dataset_path = os.path.join('datasets/temp')
        if os.path.exists(temp_dataset_path):
            shutil.rmtree(temp_dataset_path)
        shutil.copytree(dataset_path, temp_dataset_path)

        # Process images in temp dataset
        splits = ['train', 'valid', 'test']
        for split in splits:
            images_dir = os.path.join(temp_dataset_path, split, 'images')
            if not os.path.exists(images_dir):
                continue  # Skip if split does not exist

            image_files = glob.glob(os.path.join(images_dir, '*.*'))
            for image_file in tqdm(image_files):
                # Generate RGB stacked image
                rgb_stack = generate_rgb_stack_image(image_file)
                
                # Create a new filename with a .jpg extension
                base_name = os.path.splitext(os.path.basename(image_file))[0]  # Remove original extension
                jpg_file = os.path.join(images_dir, f"{base_name}.jpg")  # Add .jpg extension
                
                # Save the RGB stacked image as a JPG
                Image.fromarray(rgb_stack).save(jpg_file, format='JPEG')
                
                # Remove the original file
                if os.path.exists(image_file) and image_file != jpg_file:
                    os.remove(image_file)

        # Update data.yaml to point to temp dataset path if necessary
        temp_data_yaml = os.path.join(temp_dataset_path, 'data.yaml')
        with open(data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
        data_config['path'] = temp_dataset_path  # Update path to temp dataset
        with open(temp_data_yaml, 'w') as f:
            yaml.dump(data_config, f)

        # Start training on temp dataset
        self.model.train(
            data=temp_data_yaml,
            epochs=epochs,
            batch=batch_size,
            patience=patience,
            label_smoothing=label_smoothing
        )

        # Clean up temp dataset
        shutil.rmtree(temp_dataset_path)
