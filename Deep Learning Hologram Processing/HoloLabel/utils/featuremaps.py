# utils/featuremaps.py

import torch
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from utils.helpers import generate_rgb_stack_image

import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

from PyQt5.QtCore import QObject, pyqtSignal, QThread

import math

# Load pre-trained ResNet model
resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet.to(device)

resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove the final classification layer
resnet.eval()

# Define a transformation pipeline
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # ResNet input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ResNet normalization
                         std=[0.229, 0.224, 0.225]),
])

def extract_features(image_array):
    # Convert NumPy array to PIL image
    image = Image.fromarray(image_array)
    
    # Apply transformations
    image = transform(image).unsqueeze(0)  # Add batch dimension#
    image = image.to(device)


    # Extract features
    with torch.no_grad():
        features = resnet(image)
    
    features = features.cpu()
    return features.squeeze().numpy()  # Remove unnecessary dimensions


class FeatureExtractionWorker(QObject):
    finished = pyqtSignal(list, list, list, list)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, image_paths, labels):
        super().__init__()
        self.image_paths = image_paths
        self.labels = labels

    def run(self):
        features_list = []
        labels_list = []
        image_paths_list = []
        bounding_boxes_list = []

        for idx, image_path in enumerate(self.image_paths):
            try:
                # Load the image
                image = generate_rgb_stack_image(image_path)
                label_info_list = self.labels[idx]

                if not label_info_list:
                    # No labels for this image
                    # Decide how to handle this case (e.g., skip or assign a default label)
                    continue  # Skipping images without labels
                else:
                    for label_info in label_info_list:
                        label = label_info['class_name']
                        bbox = label_info['bbox']  # This is a QRectF object

                        # Extract coordinates from QRectF
                        x = bbox.x()
                        y = bbox.y()
                        width = bbox.width()
                        height = bbox.height()

                        x1 = int(x)
                        y1 = int(y)
                        x2 = int(x + width)
                        y2 = int(y + height)

                        # Ensure coordinates are within image bounds
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(image.shape[1], x2)
                        y2 = min(image.shape[0], y2)

                        # Crop the image
                        cropped_image = image[y1:y2, x1:x2]

                        # Extract features from the cropped image
                        features = extract_features(cropped_image)

                        # Append results
                        features_list.append(features)
                        labels_list.append(label)
                        image_paths_list.append(image_path)
                        bounding_boxes_list.append([x1, y1, x2, y2])

                # Emit progress signal
                self.progress.emit(idx + 1)

            except Exception as e:
                self.error.emit(str(e))
                return


        self.finished.emit(features_list, labels_list, image_paths_list, bounding_boxes_list)

    def extract_features_for_label(self, label, image_array):
        # (Same code as before)
        # Get the bounding box
        rect = label['bbox']
        x = max(0, int(rect.x()))
        y = max(0, int(rect.y()))
        w = int(rect.width())
        h = int(rect.height())

        # Ensure the bounding box is within image bounds
        x_end = min(x + w, image_array.shape[1])
        y_end = min(y + h, image_array.shape[0])

        if x_end <= x or y_end <= y:
            return None  # Invalid bounding box

        # Crop the particle from the image
        cropped_image = image_array[y:y_end, x:x_end]

        # If the image is grayscale, convert to RGB
        if len(cropped_image.shape) == 2 or cropped_image.shape[2] == 1:
            cropped_image = np.stack((cropped_image,) * 3, axis=-1)

        # Extract features
        features = extract_features(cropped_image)

        return features

class SelectFromCollection:
    def __init__(self, ax, collection, data, labels, image_paths, bounding_boxes):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.data = data  # X_reduced
        self.labels = labels  # y
        self.image_paths = image_paths
        self.bounding_boxes = bounding_boxes
        self.ind = []

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.cid = self.canvas.mpl_connect('key_press_event', self.onkeypress)

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.data))[0]
        print("Selected points:", self.ind)
        self.show_selected_data()

    def show_selected_data(self):
        selected_indices = self.ind
        selected_labels = [self.labels[i] for i in selected_indices]
        selected_image_paths = [self.image_paths[i] for i in selected_indices]
        selected_bounding_boxes = [self.bounding_boxes[i] for i in selected_indices]

        # Display class labels
        class_counts = {}
        for label in selected_labels:
            class_counts[label] = class_counts.get(label, 0) + 1
        message = "Selected data point labels:\n"
        for label, count in class_counts.items():
            message += f"Class {label}: {count} points\n"
        print(message)

        # Display images in a gallery
        num_images = len(selected_image_paths)
        if num_images == 0:
            print("No images selected.")
            return

        images = []
        for img_path, bbox in zip(selected_image_paths, selected_bounding_boxes):
            img = Image.open(img_path)
            x1, y1, x2, y2 = bbox
            cropped_img = img.crop((x1, y1, x2, y2))
            images.append(cropped_img)

        # Show combined gallery
        self.show_combined_gallery(images, selected_labels)

        """ cols = min(5, num_images)
        rows = num_images // cols + int(num_images % cols > 0)
        fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows),
                                gridspec_kw={'wspace': 0.01, 'hspace': 0.01})
        axes = axes.flatten()
        for idx, (ax, img_path, bbox, label) in enumerate(zip(axes, selected_image_paths,
                                                            selected_bounding_boxes, selected_labels)):
            img = plt.imread(img_path)
            x1, y1, x2, y2 = bbox
            cropped_img = img[y1:y2, x1:x2]
            ax.imshow(cropped_img)
            ax.axis('off')
            # Overlay class label directly on the image
            ax.text(0.5, 0.05, f'{label}', fontsize=10, color='white',
                    ha='center', va='bottom', transform=ax.transAxes,
                    bbox=dict(facecolor='black', alpha=0.5, pad=0.5))
        # Hide unused axes
        for ax in axes[num_images:]:
            ax.axis('off')
        plt.tight_layout()
        plt.show() """

    def show_combined_gallery(self, images, labels):
        # Determine grid size
        num_images = len(images)
        cols = int(math.ceil(math.sqrt(num_images)))
        rows = int(math.ceil(num_images / cols))
        thumbnail_size = (100, 100)  # Adjust size as needed

        # Create a new image with a white background
        combined_width = cols * thumbnail_size[0]
        combined_height = rows * thumbnail_size[1]
        combined_image = Image.new('RGB', (combined_width, combined_height), color='white')

        draw = ImageDraw.Draw(combined_image)
        font = ImageFont.load_default()

        for idx, (img, label) in enumerate(zip(images, labels)):
            # Resize image to thumbnail size
            img = img.resize(thumbnail_size)
            x_offset = (idx % cols) * thumbnail_size[0]
            y_offset = (idx // cols) * thumbnail_size[1]
            combined_image.paste(img, (x_offset, y_offset))
            # Overlay class label directly on the image
            text_position = (x_offset + 5, y_offset + 5)
            draw.text(text_position, str(label), font=font, fill='black')

        fig, ax = plt.subplots(figsize=(15, 8))
        ax.imshow(combined_image)
        ax.axis('off')
        plt.tight_layout()

        # Enable mouse wheel zoom and pan
        self.add_zoom_and_pan(fig, ax)

        plt.show()

    def add_zoom_and_pan(self, fig, ax):
        # Enable mouse wheel zoom and pan
        class ZoomPan:
            def __init__(self):
                self.press = None
                self.cur_xlim = None
                self.cur_ylim = None
                self.xpress = None
                self.ypress = None

            def zoom_factory(self, ax, base_scale=1.1):
                def zoom(event):
                    if event.inaxes != ax:
                        return
                    cur_xlim = ax.get_xlim()
                    cur_ylim = ax.get_ylim()

                    xdata = event.xdata
                    ydata = event.ydata

                    if xdata is None or ydata is None:
                        return

                    if event.button == 'up':
                        # Zoom in
                        scale_factor = 1 / base_scale
                    elif event.button == 'down':
                        # Zoom out
                        scale_factor = base_scale
                    else:
                        # Unknown event
                        scale_factor = 1

                    new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
                    new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

                    relx = (xdata - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0])
                    rely = (ydata - cur_ylim[0]) / (cur_ylim[1] - cur_ylim[0])

                    ax.set_xlim([xdata - relx * new_width, xdata + (1 - relx) * new_width])
                    ax.set_ylim([ydata - rely * new_height, ydata + (1 - rely) * new_height])
                    ax.figure.canvas.draw_idle()

                fig.canvas.mpl_connect('scroll_event', zoom)

            def pan_factory(self, ax):
                def on_press(event):
                    if event.inaxes != ax:
                        return
                    if event.button != 1:
                        return  # Only respond to left mouse button
                    self.cur_xlim = ax.get_xlim()
                    self.cur_ylim = ax.get_ylim()
                    self.press = event.xdata, event.ydata
                    self.xpress, self.ypress = event.xdata, event.ydata

                def on_release(event):
                    self.press = None
                    ax.figure.canvas.draw_idle()

                def on_motion(event):
                    if self.press is None:
                        return
                    if event.inaxes != ax:
                        return
                    dx = event.xdata - self.xpress
                    dy = event.ydata - self.ypress
                    self.cur_xlim -= dx
                    self.cur_ylim -= dy
                    ax.set_xlim(self.cur_xlim)
                    ax.set_ylim(self.cur_ylim)
                    ax.figure.canvas.draw_idle()

                fig.canvas.mpl_connect('button_press_event', on_press)
                fig.canvas.mpl_connect('button_release_event', on_release)
                fig.canvas.mpl_connect('motion_notify_event', on_motion)

        zp = ZoomPan()
        zp.zoom_factory(ax, base_scale=1.1)
        zp.pan_factory(ax)


    def onkeypress(self, event):
        if event.key == 'escape':
            self.lasso.disconnect_events()
            self.canvas.draw_idle()
            self.canvas.mpl_disconnect(self.cid)
