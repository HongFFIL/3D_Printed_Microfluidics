# gui/label_tool.py

import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QLabel, QGraphicsScene, QGraphicsView,
    QVBoxLayout, QWidget, QPushButton, QSlider, QHBoxLayout, QLineEdit, QComboBox,
    QAction, QMessageBox, QSplitter, QGraphicsRectItem, QGraphicsItem,
    QGraphicsItemGroup, QGraphicsTextItem, QCheckBox, QDialog,
    QCompleter, QGroupBox, QFormLayout, QInputDialog, QColorDialog, QProgressDialog,
    QTableWidget, QTableWidgetItem,
)
from PyQt5.QtGui import QPixmap, QPen, QColor, QFont, QPainter, QPainterPath
from PyQt5.QtCore import Qt, QRectF, QPointF, pyqtSignal, QThread
from collections import defaultdict

from gui.graphics_view import GraphicsView

import os
import glob

from model_training.predict import ModelPredictor
from model_training.train import YoloTrainer
from gui.training_dialog import InitialTrainingDialog, PredictionDialog
import threading
import yaml
import shutil

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from utils.helpers import (
    generate_rgb_stack_image, generate_focal_plane, array_to_pixmap,
    norm_to_pixel_rect, pixel_to_norm_rect, select_n_random_images,
    load_image_array, save_image_array
)

from utils.sizing import (
    crop_particle,
    segment_particle,
    calculate_size_metrics,
    setup_predictor,
)

from utils.focusmetrics import (
    find_best_focus_plane,
    obtain_infocus_image,
)

from utils.featuremaps import FeatureExtractionWorker, SelectFromCollection

from utils.enhancement import Enhancement

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
from sklearn.preprocessing import StandardScaler

class LabelTool(QMainWindow):
    prediction_completed = pyqtSignal(list)
    training_completed = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Labeling Tool")
        self.setGeometry(100, 100, 1200, 800)

        self.class_names = ["WBC", "RBC", "Debris", "EC", "PA", "PD", "EF", "BS", "CJ"]

        # Initialize variables
        self.image_paths = []  # Paths of loaded images
        self.current_image = None  # Holds the currently displayed image (QPixmap)
        self.current_array = None
        self.current_image_index = 0  # Index of the currently displayed image
        self.labels = []  # List to store labels for each image
        self.current_class = "0"  # Default class
        self.start_point = QPointF()
        self.end_point = QPointF()
        self.rect_item = None

        self.edit_mode = False  # Default to panning mode
        self.measurement_mode = False  # Default to panning mode
        self.single_sizing = True

        self.image_cache = {}  # Key: image index, Value: QPixmap
        self.cache_size = 3  # Number of images to keep in cache (adjust as needed)


        # Key parameters with default values
        self.camera_pixel_size = 3.45  # um/px
        self.magnification = 10.0
        self.index_of_refraction = 1.33
        self.wavelength = 405  # nm
        self.depth_min = 0.0  # um
        self.depth_max = 100.0  # um
        self.step_size = 10.0  # um

        self.class_colors = {
            "WBC": QColor(0, 255, 0),    # Green
            "RBC": QColor(255, 0, 0),    # Red
            "Debris": QColor(0, 0, 255),  # Blue
            "0": QColor(255,255,255),
        }

        self.current_depth = self.depth_min

        self.enhancement_enabled = False  # Flag to track if enhancement is enabled
        self.stack_size = 20  # You can allow users to set this value via the GUI
        self.enhancement = None  # Will hold the Enhancement object when enabled

        # Setup UI components
        self.init_ui()
        self.prediction_completed.connect(self.on_prediction_completed)
        self.training_completed.connect(self.on_training_completed)
        self.set_parameters("./configs/default_parameters.yaml")

        self.predictor = None
        self.init_segment_anything_model()

    def init_ui(self):
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)

        # Create graphics view and scene for image display
        self.graphics_scene = QGraphicsScene()
        self.graphics_view = GraphicsView(self)
        self.graphics_view.setScene(self.graphics_scene)
        main_splitter.addWidget(self.graphics_view)

        # Enable panning
        self.graphics_view.setDragMode(QGraphicsView.NoDrag)

        # Create side panel for controls
        side_panel_widget = QWidget()
        side_panel = QVBoxLayout()
        side_panel_widget.setLayout(side_panel)
        main_splitter.addWidget(side_panel_widget)

        main_splitter.setSizes([800, 400])  # Adjust values as needed
        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 1)

        # Add main_splitter to central widget
        main_layout = QHBoxLayout()
        main_layout.addWidget(main_splitter)
        central_widget.setLayout(main_layout)

        # Class selection with autocomplete
        self.class_input = QLineEdit()
        self.class_input.setText("0")

        self.class_history = ["WBC", "RBC", "Debris", "EC", "PA", "PD", "EF", "BS", "CJ"]
        self.completer = QCompleter(self.class_history)
        self.class_input.setCompleter(self.completer)
        self.class_input.editingFinished.connect(self.update_class_history)


        # Create groups
        parameters_group = QGroupBox("Parameters")
        parameters_layout = QFormLayout()
        parameters_group.setLayout(parameters_layout)

        mode_group = QGroupBox("Modes")
        mode_layout = QVBoxLayout()
        mode_group.setLayout(mode_layout)

        navigation_group = QGroupBox("Navigation")
        navigation_layout = QVBoxLayout()
        navigation_group.setLayout(navigation_layout)

        ml_group = QGroupBox("ML Model")
        ml_layout = QVBoxLayout()
        ml_group.setLayout(ml_layout)
        
        # Input fields for key parameters
        self.pixel_size_input = QLineEdit(str(self.camera_pixel_size))
        self.magnification_input = QLineEdit(str(self.magnification))
        self.ior_input = QLineEdit(str(self.index_of_refraction))
        self.wavelength_input = QLineEdit(str(self.wavelength))
        self.depth_min_input = QLineEdit(str(self.depth_min))
        self.depth_max_input = QLineEdit(str(self.depth_max))
        self.step_size_input = QLineEdit(str(self.step_size))

        # Refresh button to update parameters
        refresh_btn = QPushButton("Refresh Parameters")
        refresh_btn.clicked.connect(self.refresh_parameters)

        # Toggle for displaying reconstructed images
        self.display_reconstructed_image = False
        self.toggle_reconstruction_btn = QPushButton("Display Reconstructed Image")
        self.toggle_reconstruction_btn.setCheckable(True)
        self.toggle_reconstruction_btn.clicked.connect(self.toggle_reconstruction_display)

        # Toggle for displaying tripleStack image
        self.display_tripleStack_image = False
        self.toggle_tripleStack_btn = QPushButton("Display RGB Hologram")
        self.toggle_tripleStack_btn.setCheckable(True)
        self.toggle_tripleStack_btn.clicked.connect(self.toggle_tripleStack_display)

        # Toggle Display for enhancement
        self.enhancement_checkbox = QCheckBox("Enable Background Enhancement")
        self.enhancement_checkbox.stateChanged.connect(self.toggle_enhancement)


        # Mode Toggles
        self.edit_mode = False  # Default to panning mode
        self.toggle_edit_btn = QPushButton("Enable Edit Mode")
        self.toggle_edit_btn.setCheckable(True)
        self.toggle_edit_btn.clicked.connect(self.toggle_edit_mode)

        self.measurement_mode = False
        self.toggle_measurement_btn = QPushButton("Enable Measurement Mode")
        self.toggle_measurement_btn.setCheckable(True)
        self.toggle_measurement_btn.clicked.connect(self.toggle_measurement_mode)
        
        # Measurement shape selection
        self.measurement_shape = 'Line'  # Default shape
        self.shape_dropdown = QComboBox()
        self.shape_dropdown.addItems(['Line', 'Rectangle', 'Ellipse'])
        self.shape_dropdown.currentTextChanged.connect(self.change_measurement_shape)

        # Sliders
        self.image_slider = QSlider(Qt.Horizontal)
        self.image_slider.setMinimum(0)
        self.image_slider.setMaximum(0)  # Will be updated after images are loaded
        self.image_slider.setValue(0)
        self.image_slider.valueChanged.connect(self.change_image)

        self.depth_slider = QSlider(Qt.Horizontal)
        self.depth_slider.setMinimum(0)
        self.depth_slider.setMaximum(int((self.depth_max - self.depth_min) / self.step_size))
        self.depth_slider.setValue(0)
        self.depth_slider.valueChanged.connect(self.update_depth)

        # Create labels for the sliders
        self.image_slider_label = QLabel('Image: 0/0')
        self.depth_slider_label = QLabel('Depth: 0.0')

        # Image Slider Layout
        image_slider_layout = QHBoxLayout()
        image_slider_layout.addWidget(QLabel('Image:'))
        image_slider_layout.addWidget(self.image_slider)
        image_slider_layout.addWidget(self.image_slider_label)

        # Depth Slider Layout
        depth_slider_layout = QHBoxLayout()
        depth_slider_layout.addWidget(QLabel('Depth:'))
        depth_slider_layout.addWidget(self.depth_slider)
        depth_slider_layout.addWidget(self.depth_slider_label)

        # Parameters group
        parameters_layout.addRow("Camera Pixel Size (um/px):", self.pixel_size_input)
        parameters_layout.addRow("Magnification:", self.magnification_input)
        parameters_layout.addRow("Index of Refraction:", self.ior_input)
        parameters_layout.addRow("Wavelength (nm):", self.wavelength_input)
        parameters_layout.addRow("Depth Minimum (um):", self.depth_min_input)
        parameters_layout.addRow("Depth Maximum (um):", self.depth_max_input)
        parameters_layout.addRow("Step Size (um):", self.step_size_input)
        parameters_layout.addRow(refresh_btn)
        parameters_layout.addRow(self.toggle_reconstruction_btn)
        parameters_layout.addRow(self.toggle_tripleStack_btn)
        parameters_layout.addRow(self.enhancement_checkbox)

        self.focus_metric_graphs = QCheckBox("Focus Metric Graphs")
        parameters_layout.addRow(self.focus_metric_graphs)

        side_panel.addWidget(parameters_group)

        # Navigation group
        navigation_layout.addLayout(depth_slider_layout)
        navigation_layout.addLayout(image_slider_layout)

        side_panel.addWidget(navigation_group)

        # Modes group
        mode_layout.addWidget(self.toggle_edit_btn)
        mode_layout.addWidget(self.toggle_measurement_btn)
        mode_layout.addWidget(QLabel("Enter Class:"))
        mode_layout.addWidget(self.class_input)
        mode_layout.addWidget(QLabel("Select Measurement Shape:"))
        mode_layout.addWidget(self.shape_dropdown)

        side_panel.addWidget(mode_group)
        
        # Machine Learning Group
        # Model selection
        self.model_path_input = QLineEdit("model_training/models/detection/YoloV8/04172024.pt")
        self.model_select_btn = QPushButton("Select Model")
        self.model_select_btn.clicked.connect(self.select_model)
        ml_layout.addWidget(QLabel("Model Path:"))
        ml_layout.addWidget(self.model_path_input)
        ml_layout.addWidget(self.model_select_btn)

        # Confidence threshold
        self.confidence_input = QLineEdit("0.25")
        self.confidence_input.editingFinished.connect(self.on_confidence_changed)
        ml_layout.addWidget(QLabel("Confidence Threshold:"))
        ml_layout.addWidget(self.confidence_input)

        """ # IoU threshold
        self.iou_input = QLineEdit("0.45")
        ml_layout.addWidget(QLabel("IoU Threshold:"))
        ml_layout.addWidget(self.iou_input)

        # Agnostic NMS toggle
        self.agnostic_nms_checkbox = QCheckBox("Agnostic NMS")
        ml_layout.addWidget(self.agnostic_nms_checkbox) """

        self.import_as_labels_checkbox = QCheckBox("Permanent Labels")
        ml_layout.addWidget(self.import_as_labels_checkbox)

        # Run inference button
        self.inference_mode = False  # Default to off
        self.inference_mode_checkbox = QCheckBox("Enable Inference Mode")
        self.inference_mode_checkbox.stateChanged.connect(self.toggle_inference_mode)
        ml_layout.addWidget(self.inference_mode_checkbox)


        side_panel.addWidget(ml_group)

        # Add stretch to push everything upwards
        side_panel.addStretch()


        # Menu for additional options
        self.create_menu()

        # Status bar
        self.statusBar().showMessage("Ready")


    def create_menu(self):
        # Create a menu bar
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("File")

        # Load Image Folder
        load_image_folder_action = QAction("Load Image Folder", self)
        load_image_folder_action.triggered.connect(self.load_image_folder)
        file_menu.addAction(load_image_folder_action)

        # Load Labels
        load_labels_action = QAction("Load Labels", self)
        load_labels_action.triggered.connect(self.load_labels)
        file_menu.addAction(load_labels_action)

        # Load Image
        load_images_action = QAction("Load Image", self)
        load_images_action.triggered.connect(self.load_images)
        file_menu.addAction(load_images_action)

        # Load Label for Current Image
        load_label_current_action = QAction("Load Label for Current Image", self)
        load_label_current_action.triggered.connect(self.load_label_for_current_image)  
        file_menu.addAction(load_label_current_action)

        # Save Labels
        save_labels_action = QAction("Save Labels", self)
        save_labels_action.triggered.connect(self.save_labels)
        file_menu.addAction(save_labels_action)

        # Save Image
        save_image_action = QAction("Save Image", self)
        save_image_action.triggered.connect(self.save_image)
        file_menu.addAction(save_image_action)

        # Load Parameters
        load_params_action = QAction("Load Parameters", self)
        load_params_action.triggered.connect(self.load_parameters)
        file_menu.addAction(load_params_action)

        # Save Parameters
        save_params_action = QAction("Save Parameters", self)
        save_params_action.triggered.connect(self.save_parameters)
        file_menu.addAction(save_params_action)

        # Edit menu
        edit_menu = menu_bar.addMenu("Edit")

        # Set Class Colors
        set_class_colors_action = QAction("Set Class Colors", self)
        set_class_colors_action.triggered.connect(self.set_class_colors)
        edit_menu.addAction(set_class_colors_action)

        # Delete Label
        delete_label_action = QAction("Delete Label", self)
        delete_label_action.triggered.connect(self.delete_label)
        edit_menu.addAction(delete_label_action)

        # View menu
        view_menu = menu_bar.addMenu("View")

        # Draw Scale Bar
        scale_bar_action = QAction("Draw Scale Bar", self)
        scale_bar_action.triggered.connect(self.draw_scale_bar)
        view_menu.addAction(scale_bar_action)
        
        # Training menu
        training_menu = self.menuBar().addMenu("Training")

        # Initial Training action
        initial_training_action = QAction("Initial Training", self)
        initial_training_action.triggered.connect(self.initial_training)
        training_menu.addAction(initial_training_action)

        # Prediction on Unlabeled Data action
        prediction_action = QAction("Prediction on Unlabeled Data", self)
        prediction_action.triggered.connect(self.predict_on_unlabeled_data)
        training_menu.addAction(prediction_action)

        # Analysis menu
        analysis_menu = self.menuBar().addMenu('Analysis')

        # Feature Map action
        generate_feature_map_action = QAction('Generate Feature Map', self)
        generate_feature_map_action.triggered.connect(self.generate_feature_map)
        analysis_menu.addAction(generate_feature_map_action)

        load_feature_maps_action = QAction('Load and Plot Feature Maps', self)
        load_feature_maps_action.triggered.connect(self.load_and_plot_feature_maps)
        analysis_menu.addAction(load_feature_maps_action)

        # Sizing 
        self.automate_sizing_action = QAction('Automatic Sizing', self)
        self.automate_sizing_action.triggered.connect(self.automate_sizing)
        analysis_menu.addAction(self.automate_sizing_action)

        # Info Menu
        self.info_menu = self.menuBar().addMenu('Info')
        self.count_labels_action = QAction('Number of Labels', self)
        self.count_labels_action.triggered.connect(self.count_labels)
        self.info_menu.addAction(self.count_labels_action)

    def init_segment_anything_model(self):
        self.sam_predictor = setup_predictor()

    def toggle_edit_mode(self, checked):
        if checked:
            self.edit_mode = True
            # Disable other modes
            self.measurement_mode = False
            self.toggle_measurement_btn.setChecked(False)
            self.toggle_edit_btn.setText("Disable Edit Mode")
            self.toggle_measurement_btn.setText("Enable Measurement Mode")
            self.statusBar().showMessage("Edit mode enabled")
        else:
            self.edit_mode = False
            self.toggle_edit_btn.setText("Enable Edit Mode")
            self.statusBar().showMessage("Edit mode disabled")

    def toggle_measurement_mode(self, checked):
        if checked:
            self.measurement_mode = True
            # Disable other modes
            self.edit_mode = False
            self.toggle_edit_btn.setChecked(False)
            self.toggle_measurement_btn.setText("Disable Measurement Mode")
            self.toggle_edit_btn.setText("Enable Edit Mode")
            self.statusBar().showMessage("Measurement mode enabled")
        else:
            self.measurement_mode = False
            self.toggle_measurement_btn.setText("Enable Measurement Mode")
            self.statusBar().showMessage("Measurement mode disabled")

    ############### Reconsturciton, RGB, and Enhancement ###############
    def update_depth(self, value):
        self.current_depth = self.depth_min + value * self.step_size
        self.depth_slider_label.setText(f'Depth: {self.current_depth:.2f}')
        self.statusBar().showMessage(f"Current depth: {self.current_depth} um")
        if self.display_reconstructed_image:
            self.change_focal_plane()

    def change_focal_plane(self):
        if not self.display_reconstructed_image:
            return  # Do nothing if reconstruction is not toggled on

        # Get the current depth
        self.current_depth = self.depth_min + self.depth_slider.value() * self.step_size

        # Retrieve parameters
        pixel_size = float(self.pixel_size_input.text())
        magnification = float(self.magnification_input.text())
        resolution = pixel_size / magnification  # um/px

        wavelength_nm = float(self.wavelength_input.text())  # nm
        index_of_refraction = float(self.ior_input.text())
        wavelength = wavelength_nm / index_of_refraction / 1000  # Convert to um

        recon_image = generate_focal_plane(self.image_paths[self.current_image_index], resolution, wavelength, self.current_depth)
        pixmap = array_to_pixmap(recon_image, RGB=False)

        # Update the displayed image
        self.current_image = pixmap
        self.update_image_display()

    def triple_stack_current(self):
        if not self.display_tripleStack_image:
            return
        
        stack = generate_rgb_stack_image(self.image_paths[self.current_image_index])
        pixmap = array_to_pixmap(stack, RGB=True)

        # Update the displayed image
        self.current_image = pixmap
        self.update_image_display()

    def toggle_enhancement(self, state):
        if state == Qt.Checked:
            # Enable enhancement
            self.enhancement_enabled = True
            # Initialize the Enhancement object
            img_height = self.current_image.height()
            img_width = self.current_image.width()
            self.enhancement = Enhancement(
                imgsz=(img_width, img_height),
                stack_size=self.stack_size
            )
            # Preload images to fill the stack
            self.preload_enhancement_stack()
        else:
            # Disable enhancement
            self.enhancement_enabled = False
            self.enhancement = None

        self.update_image_display()

    def preload_enhancement_stack(self):
        start_index = self.current_image_index
        end_index = min(start_index + self.stack_size, len(self.image_paths))
        for idx in range(start_index, end_index):
            image_path = self.image_paths[idx]
            image_array = load_image_array(image_path)
            # Warm up the enhancement stack
            self.enhancement.moving_window_enhance(image_array)

    ############### Model Inference ###############
    def select_model(self):
        options = QFileDialog.Options()
        model_file, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "Model Files (*.pt);;All Files (*)", options=options)
        if model_file:
            self.model_path_input.setText(model_file)
            self.statusBar().showMessage(f"Model selected: {model_file}")

            # Create a ModelPredictor instance
            try:
                self.predictor = ModelPredictor(model_file)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load model: {e}")
                return
            
    def toggle_inference_mode(self, state):
        self.inference_mode = state == Qt.Checked
        if self.inference_mode:
            self.statusBar().showMessage("Inference mode enabled")
            # Run inference on the current image
            self.run_model_inference()
        else:
            self.statusBar().showMessage("Inference mode disabled")
            # Remove inference labels if necessary
            self.clear_inference_labels()
            self.update_image_display()

    def on_confidence_changed(self):
        if self.inference_mode:
            self.run_model_inference()

    def run_model_inference(self):
        # Get configuration
        try:
            conf_threshold = float(self.confidence_input.text())
        except ValueError:
            QMessageBox.warning(self, "Error", "Please enter valid numerical values for thresholds.")
            return

        # Use the RGB hologram image for inference
        image_array = generate_rgb_stack_image(self.image_paths[self.current_image_index])

        if getattr(self, 'predictor', None) is None:
            # Create a ModelPredictor instance
            try:
                self.predictor = ModelPredictor(self.model_path_input.text())
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load model: {e}")
                return

        # Run inference
        try:
            results = self.predictor.predict(
                image=image_array,
                conf_threshold=conf_threshold,
            )
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Inference failed: {e}")
            return

        # Process results
        self.inference_results = results[0]  # Assuming single image
        self.display_inference_results()


    def display_inference_results(self):
        # Draw inference results on the image
        if not self.inference_results:
            return

        if self.import_as_labels_checkbox.isChecked():
            # Convert inference results to labels
            current_index = self.current_image_index
            # Remove previous inference labels
            self.clear_inference_labels()

            # Now add new inference labels
            for box in self.inference_results.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

                rect = QRectF(QPointF(x1, y1), QPointF(x2, y2))
                class_id = int(box.cls)
                class_name = self.inference_results.names[class_id]

                self.labels[current_index].append({
                    'class_name': class_name,
                    'bbox': rect,
                    'source': 'inference'  # Mark as inference label
                })

            # Redraw labels
            self.update_image_display()
        else:
            # Remove previous inference drawings
            self.load_current_image()  # Reload the image to remove previous drawings
            image = self.current_image.copy()
            painter = QPainter(image)
            painter.setPen(QPen(QColor(255, 0, 0), 2))

            # Iterate over detections
            for box in self.inference_results.boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Draw rectangle
                painter.drawRect(x1, y1, x2 - x1, y2 - y1)

                # Draw label
                class_id = int(box.cls)
                class_name = self.inference_results.names[class_id]
                painter.setFont(QFont('Arial', 10))
                painter.drawText(x1, y1 - 5, class_name)

            painter.end()

            # Update the displayed image
            self.current_image = image
            self.update_image_display()

    def clear_inference_labels(self):
        current_index = self.current_image_index
        if self.labels and len(self.labels) > current_index:
            self.labels[current_index] = [label for label in self.labels[current_index] if label.get('source') != 'inference']

    ############### Saving and Loading ###############
    def load_images(self):
        options = QFileDialog.Options()
        file_names, _ = QFileDialog.getOpenFileNames(
            self, "Open Image Files", "", "Image Files (*.pgm *.png *.jpg *.bmp *.tif)", options=options)
        if file_names:
            self.image_paths = file_names
            self.current_image_index = 0
            self.labels = [[] for _ in self.image_paths]
            self.image_cache.clear()  # Clear the cache
            self.change_image(self.current_image_index)
            self.update_image_display()
            self.statusBar().showMessage(f"Loaded {len(self.image_paths)} images.")

            # Update the image slider minimum, maximum, and value
            self.image_slider.setMinimum(0)
            self.image_slider.setMaximum(len(self.image_paths) - 1)
            self.image_slider.setValue(0)


    def load_labels(self):
        if not self.image_paths:
            QMessageBox.warning(self, "Warning", "No images loaded. Please load images first.")
            return
        
        options = QFileDialog.Options()
        labels_directory = QFileDialog.getExistingDirectory(self, "Select Labels Directory", options=options)
        
        if not labels_directory:
            QMessageBox.warning(self, "Warning", "No labels directory selected.")
            return
        
        for idx, image_path in enumerate(self.image_paths):
            image_filename = os.path.basename(image_path)
            label_filename = os.path.splitext(image_filename)[0] + '.txt'
            label_file_path = os.path.join(labels_directory, label_filename)
            
            if os.path.exists(label_file_path):
                with open(label_file_path, 'r') as file:
                    labels = []
                    for line in file:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id, x_center_norm, y_center_norm, width_norm, height_norm = map(float, parts)
                            class_id = int(class_id)
                            class_name = self.class_history[class_id] if class_id < len(self.class_history) else str(class_id)

                            rect = norm_to_pixel_rect(
                                self.current_image.width(),
                                self.current_image.height(),
                                x_center_norm,
                                y_center_norm,
                                width_norm,
                                height_norm,
                            )

                            labels.append({
                                'class_name': class_name,
                                'bbox': rect
                            })
                    self.labels[idx] = labels  # Update labels for the image
            else:
                self.labels[idx] = []  # No labels for this image

        self.update_image_display()
        self.statusBar().showMessage("Labels loaded successfully.")

    def load_image_folder(self):
        options = QFileDialog.Options()
        images_directory = QFileDialog.getExistingDirectory(self, "Select Images Directory", options=options)
        
        if images_directory:
            # Get all image files in the directory
            supported_formats = ['*.pgm', '*.png', '*.jpg', '*.bmp', '*.tif']
            image_files = []
            for fmt in supported_formats:
                image_files.extend(glob.glob(os.path.join(images_directory, fmt)))
            if image_files:
                self.image_paths = sorted(image_files)
                self.current_image_index = 0
                self.labels = [[] for _ in self.image_paths]
                self.image_cache.clear()  # Clear the cache
                self.change_image(self.current_image_index)
                self.update_image_display()

                self.statusBar().showMessage(f"Loaded {len(self.image_paths)} images from folder.")
                # Update the image slider minimum, maximum, and value
                self.image_slider.setMinimum(0)
                self.image_slider.setMaximum(len(self.image_paths) - 1)
                self.image_slider.setValue(0)  # Reset slider to the first image
            else:
                QMessageBox.warning(self, "Warning", "No images found in the selected directory.")
        else:
            QMessageBox.warning(self, "Warning", "No directory selected.")


    def load_label_for_current_image(self):
        if not self.image_paths:
            QMessageBox.warning(self, "Warning", "No images loaded. Please load images first.")
            return
        
        options = QFileDialog.Options()
        label_file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Label File", "", "Text Files (*.txt)", options=options)
        
        if label_file_path:
            idx = self.current_image_index
            with open(label_file_path, 'r') as file:
                labels = []
                for line in file:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x_center_norm, y_center_norm, width_norm, height_norm = map(float, parts)
                        class_id = int(class_id)
                        class_name = self.class_history[class_id] if class_id < len(self.class_history) else str(class_id)
                        
                        rect = norm_to_pixel_rect(
                                self.current_image.width(),
                                self.current_image.height(),
                                x_center_norm,
                                y_center_norm,
                                width_norm,
                                height_norm,
                            )
                        
                        labels.append({
                            'class_name': class_name,
                            'bbox': rect
                        })
                self.labels[idx] = labels  # Update labels for the current image
                self.update_image_display()
                self.statusBar().showMessage("Label loaded for current image.")
        else:
            QMessageBox.warning(self, "Warning", "No label file selected.")


    def save_labels(self):
        options = QFileDialog.Options()
        labels_directory = QFileDialog.getExistingDirectory(self, "Select Labels Directory", options=options)
        if not labels_directory:
            QMessageBox.warning(self, "Warning", "No directory selected.")
            return

        for idx, labels in enumerate(self.labels):
            image_path = self.image_paths[idx]
            image_filename = os.path.basename(image_path)
            label_filename = os.path.splitext(image_filename)[0] + '.txt'
            label_file_path = os.path.join(labels_directory, label_filename)

            with open(label_file_path, 'w') as f:
                for label in labels:
                    class_name = label['class_name']
                    if class_name in self.class_names:
                        class_id = self.class_names.index(class_name)
                    else:
                        # Add new class to class_names
                        self.class_names.append(class_name)
                        class_id = self.class_names.index(class_name)

                    rect = label['bbox']

                    x_center_norm, y_center_norm, width_norm, height_norm = pixel_to_norm_rect(
                        rect,
                        self.current_image.width(),
                        self.current_image.height(),
                    )
                    
                    f.write(f"{class_id} {x_center_norm} {y_center_norm} {width_norm} {height_norm}\n")

        # Update progress
        QMessageBox.information(self, "Info", "Labels saved successfully.")

    def save_image(self):
        options = QFileDialog.Options()
        image_file, _ = QFileDialog.getSaveFileName(self, "Save Parameters File", "", "YAML Files (*.jpg);;All Files (*)", options=options)
        if image_file:
            try:
                self.current_image.save(image_file)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to save image: {e}")

    ################ Edit Mode ################
    def change_class(self, index):
        self.current_class = index
        self.statusBar().showMessage(f"Current class set to: {self.class_dropdown.currentText()}")


    def delete_label(self):
        current_index = self.current_image_index
        if self.labels and self.labels[current_index]:
            self.labels[current_index].pop()
            self.statusBar().showMessage(f"Deleted the last label from image {current_index + 1}")
            self.update_image_display()
        else:
            QMessageBox.information(self, "Info", "No labels to delete for this image")

    def redraw_labels(self):        
        if self.labels and len(self.labels) > self.current_image_index and self.labels[self.current_image_index]:
            for label in self.labels[self.current_image_index]:
                rect = label['bbox']
                class_name = label['class_name']
                color = self.class_colors.get(class_name, QColor(0, 0, 0))  # Default to black if class not found
                
                pen = QPen(color, 2)
                rect_item = QGraphicsRectItem(rect)
                rect_item.setPen(pen)
                rect_item.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsFocusable)
                
                # Create the text label
                text_item = QGraphicsTextItem(class_name)
                text_item.setDefaultTextColor(color)
                font = QFont()
                font.setPointSize(10)
                text_item.setFont(font)
                text_item.setPos(rect.x(), rect.y() - 20)
                
                # Group the rectangle and text
                group = QGraphicsItemGroup()
                group.addToGroup(rect_item)
                group.addToGroup(text_item)
                group.setFlags(
                    QGraphicsItem.ItemIsSelectable |
                    QGraphicsItem.ItemIsFocusable
                )
                
                # Add the group to the scene
                self.graphics_scene.addItem(group)
                
                # Store the group in the label for reference
                label['graphics_item'] = group
        else:
            pass


    def refresh_parameters(self):
        # Update key parameters from user input
        try:
            self.camera_pixel_size = float(self.pixel_size_input.text())
            self.magnification = float(self.magnification_input.text())
            self.index_of_refraction = float(self.ior_input.text())
            self.wavelength = float(self.wavelength_input.text())
            self.depth_min = float(self.depth_min_input.text())
            self.depth_max = float(self.depth_max_input.text())
            self.step_size = float(self.step_size_input.text())

            # Update the depth slider range
            self.depth_slider.setMinimum(0)
            max_steps = int((self.depth_max - self.depth_min) / self.step_size)
            self.depth_slider.setMaximum(max_steps)
            self.depth_slider.setValue(0)
            self.current_depth = self.depth_min

            self.current_array = None  # Reset image array to force re-conversion
            self.update_depth(self.depth_slider.value())  # Update depth
            if self.display_reconstructed_image:
                self.change_focal_plane()

            self.statusBar().showMessage("Parameters updated successfully") 
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter valid numerical values for the parameters")

    def draw_scale_bar(self):
        # Implement functionality to draw scale bars or circles
        # For example, draw a line representing 10 um at the bottom right corner
        pass

    def load_current_image(self):
        # Check if image is in cache
        if self.current_image_index in self.image_cache:
            self.current_image = self.image_cache[self.current_image_index]
        else:
            # Load image from file
            image_path = self.image_paths[self.current_image_index]
            self.current_image = QPixmap(image_path)
            # Add to cache
            self.image_cache[self.current_image_index] = self.current_image
            # Manage cache size
            self.manage_cache()

        # if 

        # Reset image array
        self.current_array = None



    def manage_cache(self):
        # Remove images outside the cache range
        cache_indices = range(max(0, self.current_image_index - 1), min(len(self.image_paths), self.current_image_index + 2))
        keys_to_remove = [key for key in self.image_cache.keys() if key not in cache_indices]
        for key in keys_to_remove:
            del self.image_cache[key]

    def change_image(self, value):
        self.current_image_index = value
        self.image_slider_label.setText(f'Image: {value + 1}/{len(self.image_paths)}')
        self.load_current_image()

        if self.display_reconstructed_image:
            self.change_focal_plane()
        if self.display_tripleStack_image:
            self.triple_stack_current()
        else:
            self.update_image_display()

        self.statusBar().showMessage(f"Displaying image {self.current_image_index + 1}/{len(self.image_paths)}")

        if self.inference_mode:
            self.run_model_inference()


    def clear(self):
        # Get a list of all current items in the scene
        all_items = self.graphics_scene.items()
        
        # Extract items to keep from self.graphics_view.measurement_items
        items_to_keep = set()
        for item, text in self.graphics_view.measurement_items:
            items_to_keep.add(item)
            items_to_keep.add(text)

        # Remove and delete items that are not in measurement_items
        for item in all_items:
            self.graphics_scene.removeItem(item)
            if item not in items_to_keep:
                del item  # Explicitly delete the item

    def update_image_display(self):
        if self.current_image:
            self.clear()

            if self.enhancement_enabled and self.enhancement is not None:
                image_array = load_image_array(self.image_paths[self.current_image_index])
                enhanced_image = self.enhancement.moving_window_enhance(image_array, to_cpu=True)
                enhanced_pixmap = array_to_pixmap(enhanced_image, RGB=False)
                self.graphics_scene.addPixmap(enhanced_pixmap)

            else:
                self.graphics_scene.addPixmap(self.current_image)

            self.redraw_labels() 
            self.redraw_measurements()  # Re-add measurements to the scene
        else:
            self.clear()
    
    def redraw_measurements(self):
        for item, text in self.graphics_view.measurement_items:
            self.graphics_view.scene().addItem(item)
            self.graphics_view.scene().addItem(text)


    def change_measurement_shape(self, text):
        self.measurement_shape = text
        self.statusBar().showMessage(f"Measurement shape set to: {text}")

    def set_parameters(self, params_file):
        if params_file:
            try:
                with open(params_file, 'r') as f:
                    params = yaml.safe_load(f)
                self.pixel_size_input.setText(str(params.get('camera_pixel_size', self.camera_pixel_size)))
                self.magnification_input.setText(str(params.get('magnification', self.magnification)))
                self.ior_input.setText(str(params.get('index_of_refraction', self.index_of_refraction)))
                self.wavelength_input.setText(str(params.get('wavelength', self.wavelength)))
                self.depth_min_input.setText(str(params.get('depth_min', self.depth_min)))
                self.depth_max_input.setText(str(params.get('depth_max', self.depth_max)))
                self.step_size_input.setText(str(params.get('step_size', self.step_size)))
                # Load class history
                self.class_history = params.get('classes', self.class_history)
                self.completer.model().setStringList(self.class_history)
                # Load class colors
                class_colors = params.get('class_colors', {})
                self.class_colors = {class_name: QColor(color_str) for class_name, color_str in class_colors.items()}
                self.refresh_parameters()
                self.statusBar().showMessage("Parameters loaded successfully.")

                if not self.current_image is None:
                    self.redraw_labels()  # Update labels with new class colors
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load parameters: {e}")

    def load_parameters(self):
        options = QFileDialog.Options()
        params_file, _ = QFileDialog.getOpenFileName(self, "Load Parameters File", "", "YAML Files (*.yaml);;All Files (*)", options=options)
        self.set_parameters(params_file)

    
    def save_parameters(self):
        options = QFileDialog.Options()
        params_file, _ = QFileDialog.getSaveFileName(self, "Save Parameters File", "", "YAML Files (*.yaml);;All Files (*)", options=options)
        if params_file:
            try:
                params = {
                    'camera_pixel_size': float(self.pixel_size_input.text()),
                    'magnification': float(self.magnification_input.text()),
                    'index_of_refraction': float(self.ior_input.text()),
                    'wavelength': float(self.wavelength_input.text()),
                    'depth_min': float(self.depth_min_input.text()),
                    'depth_max': float(self.depth_max_input.text()),
                    'step_size': float(self.step_size_input.text()),
                    'classes': self.class_history,
                    'class_colors': {class_name: color.name() for class_name, color in self.class_colors.items()}
                }
                with open(params_file, 'w') as f:
                    yaml.dump(params, f)
                self.statusBar().showMessage("Parameters saved successfully.")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to save parameters: {e}")


    def update_class_history(self):
        class_name = self.class_input.text().strip()
        if class_name and class_name not in self.class_history:
            self.class_history.append(class_name)
            self.completer.model().setStringList(self.class_history)
        self.current_class = class_name
        self.statusBar().showMessage(f"Current class set to: {class_name}")
    
    def set_class_colors(self):
        updated_colors = self.class_colors.copy()
        class_name = self.current_class

        color = self.class_colors.get(class_name, QColor(0, 0, 0))
        new_color = QColorDialog.getColor(color, self, f"Select Color for Class '{class_name}'")
        if new_color.isValid():
            updated_colors[class_name] = new_color
        else:
            updated_colors[class_name] = color  # Keep existing color if dialog canceled

        self.class_colors = updated_colors
        self.statusBar().showMessage("Class colors updated.")
        self.redraw_labels()  # Update the display with new colors

    def toggle_reconstruction_display(self, checked):
        self.display_reconstructed_image = checked
        self.display_tripleStack_image = False
        self.toggle_tripleStack_btn.setChecked(False)
        if checked:
            self.toggle_reconstruction_btn.setText("Display Original Image")
            self.toggle_tripleStack_btn.setText("Display RGB Hologram")
            self.change_focal_plane()
        else:
            self.toggle_reconstruction_btn.setText("Display Reconstructed Image")
            self.load_current_image()
            self.update_image_display()
    
    def toggle_tripleStack_display(self, checked):
        self.display_tripleStack_image = checked
        self.display_reconstructed_image = False
        self.toggle_reconstruction_btn.setChecked(False)
        if checked:
            self.toggle_tripleStack_btn.setText("Display Gray Hologram")
            self.toggle_reconstruction_btn.setText("Display Reconstructed Image")
            self.triple_stack_current()
        else:
            self.toggle_tripleStack_btn.setText("Display RGB Hologram")
            self.load_current_image()
            self.update_image_display()


    def stack_params_changed(self):
        # Check if the parameters used to generate the image stack have changed
        current_params = (float(self.pixel_size_input.text()),
                        float(self.magnification_input.text()),
                        float(self.wavelength_input.text()),
                        float(self.ior_input.text()),
                        float(self.depth_min_input.text()),
                        float(self.depth_max_input.text()),
                        float(self.step_size_input.text()))
        if hasattr(self, 'last_stack_params'):
            return current_params != self.last_stack_params
        else:
            return True  # No previous parameters, so assume they have changed

    def change_label_class(self, item_group):
        # Find the label associated with this item_group
        current_index = self.current_image_index
        for label in self.labels[current_index]:
            if label.get('graphics_item') == item_group:
                # Prompt user for new class
                new_class, ok = QInputDialog.getText(self, "Change Class", "Enter new class name:", text=label['class_name'])
                if ok and new_class:
                    label['class_name'] = new_class
                    # Update the display
                    self.update_image_display()
                break
    
    def delete_label_item(self, item_group):
        current_index = self.current_image_index
        labels = self.labels[current_index]
        for label in labels:
            if label.get('graphics_item') == item_group:
                labels.remove(label)
                self.update_image_display()
                self.statusBar().showMessage("Label deleted.")
                break
    
    ################ Training #################
    ######### Training from Dataset ###########
    def initial_training(self):
        dialog = InitialTrainingDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            # Get training parameters
            model_type = dialog.model_type_combo.currentText()
            model_size = dialog.model_size_combo.currentText()
            dataset_path = dialog.dataset_path_input.text()
            epochs = int(dialog.epochs_input.text())
            batch_size = int(dialog.batch_size_input.text())
            patience = int(dialog.patience_input.text())
            label_smoothing = float(dialog.label_smoothing_input.text())

            # Validate inputs
            if not os.path.exists(dataset_path):
                QMessageBox.warning(self, "Error", "Dataset path does not exist.")
                return

            # Start training in a separate thread to avoid blocking the UI
            self.start_training(model_type, model_size, dataset_path, epochs, batch_size, patience, label_smoothing)

    def start_training(self, model_type, model_size, dataset_path, epochs, batch_size, patience, label_smoothing):
        # Show a message that training has started
        self.statusBar().showMessage("Training started...")

        training_thread = threading.Thread(target=self.run_training_process, args=(
            model_type, model_size, dataset_path, epochs, batch_size, patience, label_smoothing))
        training_thread.start()

    def run_training_process(self, model_type, model_size, dataset_path, epochs, batch_size, patience, label_smoothing):
        try:
            # Build the model path or configuration based on the model type and size
            model_config = f"{model_type.lower()}{model_size}.pt"
            trainer = YoloTrainer(model_config)
            trainer.simple_train(dataset_path, epochs, batch_size, patience, label_smoothing)

            # Training completed
            self.statusBar().showMessage("Training completed.")
            self.training_completed.emit(True)

        except Exception as e:
            # Handle exceptions and display error messages
            self.statusBar().showMessage(f"Training failed: {e}")
            self.training_completed.emit(False)

    def on_training_completed(self, status):
        if status:
            QMessageBox.information(self, "Training", "Training completed successfully.")
        else:
            QMessageBox.warning(self, "Error", f"Training failed")

    ######### Prediction from Model ###########
    def start_prediction(self, model_path, images_folder, conf_threshold, num_images_to_select, low_conf_threshold):
        self.statusBar().showMessage("Prediction started...")

        prediction_thread = threading.Thread(target=self.run_prediction_process, args=(
            model_path, images_folder, conf_threshold, num_images_to_select, low_conf_threshold))
        prediction_thread.start()

    def predict_on_unlabeled_data(self):
        dialog = PredictionDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            # Get prediction parameters
            model_path = dialog.model_path_input.text()
            images_folder = dialog.images_folder_input.text()
            conf_threshold = float(dialog.confidence_input.text())
            num_images_to_select = int(dialog.num_images_input.text())
            low_conf_threshold = float(dialog.low_conf_threshold_input.text())

            # Validate inputs
            if not os.path.exists(model_path):
                QMessageBox.warning(self, "Error", "Model file does not exist.")
                return
            if not os.path.exists(images_folder):
                QMessageBox.warning(self, "Error", "Images folder does not exist.")
                return

            # Start prediction in a separate thread
            self.start_prediction(model_path, images_folder, conf_threshold, num_images_to_select, low_conf_threshold)

    def run_prediction_process(self, model_path, images_folder, conf_threshold, num_images_to_select, low_conf_threshold):
        try:
            predictor = ModelPredictor(model_path)

            # Get list of all images
            all_image_files = []
            supported_formats = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tif', '*.pgm']
            for fmt in supported_formats:
                all_image_files.extend(glob.glob(os.path.join(images_folder, fmt)))

            selected_images = select_n_random_images(all_image_files, num_images_to_select)

            # Create new dataset folder
            dataset_root = 'datasets/prediction'
            dataset_dir = dataset_root
            index = 1
            while os.path.exists(dataset_dir):
                dataset_dir = f"{dataset_root}{index}"
                index += 1

            images_dir = os.path.join(dataset_dir, 'images')
            labels_dir = os.path.join(dataset_dir, 'labels')
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)

            # Copy selected images to new dataset folder
            for img_path in selected_images:
                shutil.copy(img_path, images_dir)

            # Update images_folder and labels_folder to new dataset paths
            images_folder = images_dir
            labels_folder = labels_dir
            self.labels_folder = labels_folder  # Store labels folder

            # Get list of images in new dataset folder
            image_files = []
            for fmt in supported_formats:
                image_files.extend(glob.glob(os.path.join(images_folder, fmt)))

            if not image_files:
                self.prediction_completed.emit([])
                return

            # Generate RGB stack images and run inference
            # all_results = []
            """ for img_path in image_files:
                stack = generate_rgb_stack_image(img_path)
                prediction = predictor.predict(stack, conf_threshold)
                result = prediction[0]
                all_results.append(result) """

            # Prepare list to hold images to label
            images_to_label = []

            # Process and filter predictions
            for img_path in image_files:
                stack = generate_rgb_stack_image(img_path)
                prediction = predictor.predict(stack, conf_threshold)
                result = prediction[0]

                label_path = os.path.join(labels_folder, os.path.splitext(os.path.basename(img_path))[0] + '.txt')
                uncertainties = []
                with open(label_path, 'w') as f:
                    for box in result.boxes:
                        cls = int(box.cls)
                        conf = box.conf[0].item()
                        uncertainties.append(1 - conf)  # Store uncertainty if needed
                        x_center = box.xywhn[0][0].item()
                        y_center = box.xywhn[0][1].item()
                        width = box.xywhn[0][2].item()
                        height = box.xywhn[0][3].item()
                        f.write(f"{cls} {x_center} {y_center} {width} {height}\n")

                # Determine if image should be labeled
                if result.boxes:
                    # Get confidences
                    confidences = result.boxes.conf.cpu().numpy()
                    if all(conf < low_conf_threshold for conf in confidences):
                        images_to_label.append(img_path)
                    else:
                        # Move image to high_conf folder
                        high_conf_dir = os.path.join(dataset_dir, 'high_conf')
                        hc_images = os.path.join(high_conf_dir, "images")
                        hc_labels = os.path.join(high_conf_dir, "labels")
                        os.makedirs(hc_images, exist_ok=True)
                        os.makedirs(hc_labels, exist_ok=True)

                        shutil.move(img_path, hc_images)
                        if os.path.exists(label_path):
                            shutil.move(label_path, hc_labels)
                else:
                    # No detections, move to background
                    background_dir = os.path.join(dataset_dir, 'background')
                    bk_images = os.path.join(background_dir, "images")
                    bk_labels = os.path.join(background_dir, "labels")
                    os.makedirs(bk_images, exist_ok=True)
                    os.makedirs(bk_labels, exist_ok=True)

                    shutil.move(img_path, bk_images)
                    if os.path.exists(label_path):
                        shutil.move(label_path, bk_labels)

            # Emit the signal with the images to label
            self.prediction_completed.emit(images_to_label)

        except Exception as e:
            self.prediction_completed.emit([])
            print(f"Prediction failed: {e}")


    def on_prediction_completed(self, sorted_images):
        if not sorted_images:
            # Handle the case where prediction failed or no images were found
            self.statusBar().showMessage("Prediction failed or no images found.")
            QMessageBox.warning(self, "Prediction", "Prediction failed or no images found.")
            return

        self.statusBar().showMessage("Prediction completed.")

        # Ask the user if they want to load high-uncertainty images
        load_images = QMessageBox.question(
            self, "Load Images",
            "Prediction completed successfully.\nDo you want to load the images with highest uncertainty for labeling?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)

        if load_images == QMessageBox.Yes:
            self.sorted_images = sorted_images
            self.load_high_uncertainty_images()

    def load_high_uncertainty_images(self):
        if hasattr(self, 'sorted_images') and self.sorted_images:
            self.image_paths = self.sorted_images
            self.current_image_index = 0
            self.labels = []
            for img_path in self.image_paths:
                image_filename = os.path.basename(img_path)
                label_filename = os.path.splitext(image_filename)[0] + '.txt'
                label_file_path = os.path.join(self.labels_folder, label_filename)

                labels = []
                if os.path.exists(label_file_path):
                    with open(label_file_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id, x_center_norm, y_center_norm, width_norm, height_norm = map(float, parts[:5])
                                class_id = int(class_id)
                                class_name = self.class_history[class_id] if class_id < len(self.class_history) else str(class_id)

                                # Convert normalized coordinates to pixel coordinates
                                image_width = self.current_image.width()
                                image_height = self.current_image.height()
                                x_center = x_center_norm * image_width
                                y_center = y_center_norm * image_height
                                width = width_norm * image_width
                                height = height_norm * image_height
                                x = x_center - width / 2
                                y = y_center - height / 2
                                rect = QRectF(x, y, width, height)

                                labels.append({
                                    'class_name': class_name,
                                    'bbox': rect
                                })
                self.labels.append(labels)
            self.image_cache.clear()
            self.load_current_image()
            self.update_image_display()
            self.statusBar().showMessage("Loaded images with predictions for labeling.")

            # Update the image slider
            self.image_slider.setMinimum(0)
            self.image_slider.setMaximum(len(self.image_paths) - 1)
            self.image_slider.setValue(0)
        else:
            QMessageBox.warning(self, "Warning", "No sorted images available. Please run prediction first.")

    ######## Sizing ##########
    def size_particle(self, item_group):
        # Find the label associated with this item_group
        current_index = self.current_image_index
        for label in self.labels[current_index]:
            if label.get('graphics_item') == item_group:
                # Proceed to size the particle
                self.perform_sizing(label)
                break
    
    def perform_sizing(self, label):
        # Get the bounding box
        rect = label['bbox']
        # Expand the bounding box by a factor of n
        n = 2
        expanded_rect = rect.adjusted(
            -rect.width() / n, -rect.height() / n,
            rect.width() / n, rect.height() / n
        )

        # Ensure the expanded rectangle is within image bounds
        expanded_rect = expanded_rect.intersected(
            QRectF(0, 0, self.current_image.width(), self.current_image.height())
        )

        # Load the image as a NumPy array
        image_array = load_image_array(self.image_paths[self.current_image_index])

        # Crop the particle from the image
        cropped_image = crop_particle(image_array, expanded_rect)

        # Get reconstruction parameters
        reconstruction_params = self.get_reconstruction_params()
        resolution = reconstruction_params['resolution']
        wavelength = reconstruction_params['wavelength']
        min_depth = reconstruction_params['z_start']
        z_step = reconstruction_params['z_step']
        num_z_planes = reconstruction_params['num_planes']
        img_size = cropped_image.shape

        # Find the best focal plane using the refactored focus metrics
        focus_results = find_best_focus_plane(
            source_img=cropped_image,
            img_size=img_size,
            num_z_planes=num_z_planes,
            min_depth=min_depth,
            z_step=z_step,
            resolution=resolution,
            wavelength=wavelength,
            graphs=self.focus_metric_graphs.isChecked()
        )

        # Access the results
        best_z_values = focus_results['best_z_values']
        best_z = best_z_values.get('brenner')  # Replace 'wavelet' with the preferred metric

        if best_z is None:
            if self.single_sizing:
                QMessageBox.warning(self, "Sizing Error", "No focus found, try reducing step size.")
            return None
        
        best_z -= 0
        slider_value = int((best_z - self.depth_min) / self.step_size)
        self.depth_slider.setValue(slider_value)

        # Reconstruct the image at the best focal plane
        reconstructed_image = obtain_infocus_image(cropped_image, resolution, wavelength, best_z, img_size)

        # Optionally display reconstructed image
        # cv2.imshow("Reconstructed Image", reconstructed_image)
        # cv2.waitKey(0)

        # Segment the particle using Segment Anything
        mask = segment_particle(reconstructed_image, self.sam_predictor)

        # Optionally display mask
        # cv2.imshow("Segmentation Mask", mask * 255)
        # cv2.waitKey(0)

        # Calculate size metrics
        metrics = calculate_size_metrics(mask, self.get_pixel_size(), best_z)

        # Display the contour and size metrics
        if self.single_sizing:
            self.display_sizing_results(mask, metrics, expanded_rect)

        return metrics

    
    def display_sizing_results(self, mask, metrics, expanded_rect):
        if metrics is None:
            QMessageBox.warning(self, "Sizing Error", "Could not compute size metrics.")
            return

        # Draw the contour on the image
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Create an overlay image
        overlay = self.current_image.copy()
        painter = QPainter(overlay)
        painter.setPen(QPen(QColor(0, 255, 0), 2))

        # Offset contours based on expanded_rect position
        for contour in contours:
            shifted_contour = contour + np.array([int(expanded_rect.x()), int(expanded_rect.y())])
            points = [QPointF(p[0][0], p[0][1]) for p in shifted_contour]
            if points:
                path = QPainterPath()
                path.moveTo(points[0])
                for pt in points[1:]:
                    path.lineTo(pt)
                path.closeSubpath()
                painter.drawPath(path)

        painter.end()
        self.current_image = overlay
        self.update_image_display()

        # Display size metrics in a popup
        metrics_text = "\n".join(f"{key}: {value:.2f}" for key, value in metrics.items())
        QMessageBox.information(self, "Size Metrics", metrics_text)

    def get_reconstruction_params(self):
        pixel_size = float(self.pixel_size_input.text())
        magnification = float(self.magnification_input.text())
        resolution = pixel_size / magnification  # µm/px

        wavelength_nm = float(self.wavelength_input.text())  # nm
        index_of_refraction = float(self.ior_input.text())
        wavelength = wavelength_nm / index_of_refraction / 1000  # Convert to µm

        z_start = float(self.depth_min_input.text())
        z_step = float(self.step_size_input.text())
        num_planes = int((float(self.depth_max_input.text()) - z_start) / z_step) + 1

        phi = wavelength * np.pi / (resolution ** 2)

        params = {
            'resolution': resolution,
            'wavelength': wavelength,
            'phi': phi,
            'z_start': z_start,
            'z_step': z_step,
            'num_planes': num_planes
        }
        return params
    

    def get_pixel_size(self):
        pixel_size = float(self.pixel_size_input.text()) / float(self.magnification_input.text())
        return pixel_size
    
    def automate_sizing(self):
        # Confirm with the user
        reply = QMessageBox.question(
            self,
            'Automate Sizing',
            'This will perform sizing on all labeled particles in all images. Continue?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.No:
            return

        self.single_sizing = False
        total_images = len(self.image_paths)
        progress_dialog = QProgressDialog('Performing automated sizing...', 'Cancel', 0, total_images, self)
        progress_dialog.setWindowTitle('Automated Sizing')
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.show()

        # List to collect metrics
        all_metrics = []

        for index in range(total_images):
            if progress_dialog.wasCanceled():
                break
            self.current_image_index = index
            self.change_image(index)
            labels = self.labels[index]
            for label in labels:
                metrics = self.perform_sizing(label)
                if metrics is not None:
                    # Add additional information like image index or file name
                    metrics['Image'] = self.image_paths[index]
                    metrics['Class'] = label['class_name']
                    all_metrics.append(metrics)
            progress_dialog.setValue(index + 1)
            QApplication.processEvents()
        
        progress_dialog.close()
        QMessageBox.information(self, 'Automated Sizing', 'Sizing completed for all labeled particles.')

        # Present the collected metrics
        self.present_metrics(all_metrics)
        self.single_sizing = True

    def present_metrics(self, all_metrics):
        df = pd.DataFrame(all_metrics)

        # Create a dialog to display the table
        dialog = QDialog(self)
        dialog.setWindowTitle("Sizing Metrics")
        layout = QVBoxLayout(dialog)

        # Create the table widget
        table = QTableWidget()
        table.setRowCount(len(all_metrics))

        # Determine the table columns from the keys of the metrics dictionaries
        if all_metrics:
            columns = list(all_metrics[0].keys())
            table.setColumnCount(len(columns))
            table.setHorizontalHeaderLabels(columns)

            # Populate the table
            for row, metrics in enumerate(all_metrics):
                for col, key in enumerate(columns):
                    value = metrics[key]
                    table_item = QTableWidgetItem(f"{value:.2f}" if isinstance(value, float) else str(value))
                    table.setItem(row, col, table_item)
        else:
            table.setColumnCount(0)

        layout.addWidget(table)

        # Add Save button
        save_button = QPushButton("Save to CSV")
        save_button.clicked.connect(lambda: self.save_metrics_dataframe_to_csv(df))
        layout.addWidget(save_button)

        dialog.setLayout(layout)
        dialog.resize(800, 600)
        dialog.exec_()

    def save_metrics_dataframe_to_csv(self, df):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Metrics", "", "CSV Files (*.csv)")
        if not file_path:
            return
        try:
            df.to_csv(file_path, index=False, float_format='%.2f')
            QMessageBox.information(self, "Save Successful", f"Metrics saved to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"An error occurred while saving: {str(e)}")

    ############ INFO MENU #################
    def count_labels(self):
        # Initialize a dictionary to hold class counts
        class_counts = defaultdict(int)
        images_with_labels = 0
        
        # Iterate over all images and their labels
        for image_labels in self.labels:
            if image_labels:
                images_with_labels += 1
                for label in image_labels:
                    class_name = label.get('class_name', 'Unknown')
                    class_counts[class_name] += 1

        # Create a dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Label Counts")
        layout = QVBoxLayout()

        # Add total labels info
        total_labels = sum(class_counts.values())
        total_labels_label = QLabel(f"Total Images with Labels: {images_with_labels}\nTotal Labels: {total_labels}")
        layout.addWidget(total_labels_label)

        # Create a table
        table = QTableWidget()
        table.setRowCount(len(class_counts))
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(['Class Name', 'Count'])
        table.verticalHeader().setVisible(False)

        for row, (class_name, count) in enumerate(sorted(class_counts.items())):
            table.setItem(row, 0, QTableWidgetItem(str(class_name)))
            table.setItem(row, 1, QTableWidgetItem(str(count)))

        table.resizeColumnsToContents()
        layout.addWidget(table)
        dialog.setLayout(layout)
        dialog.exec_()

    ######### Feature Mapping ############
    def generate_feature_map(self):
        # Confirm with the user
        reply = QMessageBox.question(
            self,
            'Generate Feature Map',
            'This will extract features from all labeled particles and generate a feature map. Continue?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.No:
            return

        # Create a QThread
        self.thread = QThread()
        # Create a worker object
        self.worker = FeatureExtractionWorker(self.image_paths, self.labels)
        # Move the worker to the thread
        self.worker.moveToThread(self.thread)
        # Connect signals and slots
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_feature_extraction_finished)
        self.worker.progress.connect(self.update_progress)
        self.worker.error.connect(self.on_worker_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        # Start the thread
        self.thread.start()

        # Show a progress dialog
        self.progress_dialog = QProgressDialog('Extracting features...', 'Cancel', 0, len(self.image_paths), self)
        self.progress_dialog.setWindowTitle('Feature Extraction')
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.canceled.connect(self.thread.quit)
        self.progress_dialog.show()

    def update_progress(self, value):
        self.progress_dialog.setValue(value)
        QApplication.processEvents()

    def on_feature_extraction_finished(self, features_list, labels_list, image_paths, bounding_boxes_list):
        self.progress_dialog.close()
        self.save_features_and_labels(features_list, labels_list, image_paths, bounding_boxes_list)
        # self.visualize_features(features_list, labels_list, image_paths, bounding_boxes_list)

    def on_worker_error(self, error_message):
        self.progress_dialog.close()
        QMessageBox.critical(self, 'Error', f'An error occurred: {error_message}')

    def save_features_and_labels(self, features_list, labels_list, image_paths, bounding_boxes_list):
        # Ask the user if they want to save the features
        reply = QMessageBox.question(
            self,
            'Save Features',
            'Do you want to save the extracted features for future use?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.No:
            return

        # Prompt the user for a file path
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Features",
            "",
            "NumPy Files (*.npz);;All Files (*)",
            options=options
        )
        if not file_path:
            return

        # Save the features, labels, and image paths
        try:
            np.savez_compressed(file_path,
                                features=np.array(features_list),
                                labels=np.array(labels_list),
                                image_paths=np.array(image_paths),
                                bounding_boxes=np.array(bounding_boxes_list))
            QMessageBox.information(self, 'Save Successful', f'Features saved to {file_path}')
        except Exception as e:
            QMessageBox.warning(self, 'Save Failed', f'An error occurred while saving: {e}')


    def visualize_features(self, features_list, labels_list, image_paths):
        if not features_list:
            QMessageBox.warning(self, 'No Features', 'No features were extracted.')
            return

        X = np.array(features_list)
        y = np.array(labels_list)

        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Ask the user which method to use
        method, ok = QInputDialog.getItem(
            self,
            'Select Method',
            'Choose dimensionality reduction method:',
            ['PCA', 't-SNE'],
            0,
            False
        )
        if not ok:
            return

        # Ask the user if they want interactivity
        interactive_plot = QMessageBox.question(
            self,
            'Interactive Plot',
            'Do you want to enable interactivity (select data points on the plot)?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if interactive_plot == QMessageBox.Yes:
            n_components = 2
        else:
            n_components = 3

        if method == 'PCA':
            reducer = PCA(n_components=n_components)
            X_reduced = reducer.fit_transform(X_scaled)
            title = f'PCA of Particle Features ({n_components}D)'
        elif method == 't-SNE':
            reducer = TSNE(n_components=n_components, perplexity=30, n_iter=300)
            X_reduced = reducer.fit_transform(X_scaled)
            title = f't-SNE of Particle Features ({n_components}D)'
        else:
            QMessageBox.warning(self, 'Invalid Method', 'Invalid dimensionality reduction method selected.')
            return

        if n_components == 2:
            # Plot the reduced features
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis', alpha=0.7)
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_title(title)
            plt.tight_layout()

            selector = SelectFromCollection(ax, scatter, X_reduced, y, image_paths)

            plt.show()
        else:
            # 3D Plot
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y, cmap='viridis', alpha=0.7)
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_zlabel('Component 3')
            ax.set_title(title)
            plt.tight_layout()
            plt.show()


    def load_and_plot_feature_maps(self):
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Load Feature Maps",
            "",
            "NumPy Files (*.npz);;All Files (*)",
            options=options
        )
        if not files:
            return

        all_features_list = []
        all_labels_list = []
        file_identifiers = []
        all_image_paths_list = []
        all_bounding_boxes_list = []

        for file_path in files:
            try:
                data = np.load(file_path, allow_pickle=True)
                features_list = data['features']
                labels_list = data['labels']
                image_paths_list = data['image_paths']
                if 'bounding_boxes' in data:
                    bounding_boxes_list = data['bounding_boxes']
                else:
                    QMessageBox.warning(self, 'Missing Data', f"'bounding_boxes' not found in {file_path}. Skipping this file.")
                    continue  # Skip this file

                # Append data only after successful loading
                all_features_list.append(features_list)
                all_labels_list.append(labels_list)
                all_image_paths_list.append(image_paths_list)
                all_bounding_boxes_list.append(bounding_boxes_list)
                identifier = os.path.basename(file_path)
                file_identifiers.append(identifier)
            except Exception as e:
                QMessageBox.warning(self, 'Load Failed', f'An error occurred while loading {file_path}:\n{e}')
                continue  # Skip this file

        if not all_features_list:
            QMessageBox.warning(self, 'No Data', 'No feature maps were loaded successfully.')
            return

        self.visualize_multiple_feature_maps(
            all_features_list,
            all_labels_list,
            all_image_paths_list,
            all_bounding_boxes_list,
            file_identifiers
        )

    def visualize_multiple_feature_maps(self, all_features_list, all_labels_list, all_image_paths_list, all_bounding_boxes_list, identifiers):
        # Flatten the lists
        combined_features = np.concatenate(all_features_list, axis=0)
        combined_labels = np.concatenate(all_labels_list, axis=0)
        combined_image_paths = np.concatenate(all_image_paths_list, axis=0)
        combined_bounding_boxes = np.concatenate(all_bounding_boxes_list, axis=0)

        # Optionally, create a list to identify which feature belongs to which file
        feature_sources = []
        for idx, features in enumerate(all_features_list):
            feature_sources.extend([identifiers[idx]] * len(features))

        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(combined_features)

        # Ask the user which method to use
        method, ok = QInputDialog.getItem(
            self,
            'Select Method',
            'Choose dimensionality reduction method:',
            ['PCA', 't-SNE'],
            0,
            False
        )
        if not ok:
            return

        # Ask the user how many dimensions to plot
        n_components, ok = QInputDialog.getInt(
            self,
            'Number of Dimensions',
            'Enter the number of dimensions to plot (2 or 3):',
            value=2,
            min=2,
            max=3
        )
        if not ok:
            return

        if method == 'PCA':
            reducer = PCA(n_components=n_components)
            X_reduced = reducer.fit_transform(X_scaled)
            title = f'PCA of Particle Features ({n_components}D)'

            # Plot the explained variance ratios
            if n_components >= 2:
                explained_variance = reducer.explained_variance_ratio_

                plt.figure(figsize=(8, 5))
                plt.bar(range(1, len(explained_variance) + 1), explained_variance * 100, alpha=0.7, align='center')
                plt.step(range(1, len(np.cumsum(explained_variance)) + 1), np.cumsum(explained_variance) * 100, where='mid')
                plt.xlabel('Principal Component')
                plt.ylabel('Explained Variance (%)')
                plt.title('Explained Variance by Principal Components')
                plt.grid(True)
                plt.show()

                # Ask the user if they want to see the loadings
                show_loadings = QMessageBox.question(
                    self,
                    'Show Loadings',
                    'Do you want to see the feature loadings for the principal components?',
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if show_loadings == QMessageBox.Yes:
                    # Assume you have a list of feature names
                    feature_names = [f'Feature {i+1}' for i in range(X_scaled.shape[1])]

                    # Get the loadings
                    loadings = reducer.components_.T * np.sqrt(reducer.explained_variance_)

                    # Create a DataFrame for loadings
                    import pandas as pd
                    pc_labels = [f'PC{i+1}' for i in range(n_components)]
                    loadings_df = pd.DataFrame(loadings, index=feature_names, columns=pc_labels)

                    # Plot loadings for each principal component
                    for i in range(n_components):
                        plt.figure(figsize=(10, 6))
                        loadings_df.iloc[:, i].plot(kind='bar')
                        plt.title(f'Feature Loadings for PC{i+1}')
                        plt.ylabel('Loading Value')
                        plt.xlabel('Features')
                        plt.grid(True)
                        plt.tight_layout()
                        plt.show()
        elif method == 't-SNE':
            reducer = TSNE(n_components=n_components, perplexity=30, n_iter=300)
            X_reduced = reducer.fit_transform(X_scaled)
            title = f't-SNE of Particle Features ({n_components}D)'
        else:
            QMessageBox.warning(self, 'Invalid Method', 'Invalid dimensionality reduction method selected.')
            return

        # *** K-Means Clustering Addition ***
        # Ask the user if they want to apply k-means clustering
        apply_kmeans = QMessageBox.question(
            self,
            'Apply K-Means Clustering',
            'Do you want to apply k-means clustering to the reduced data?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if apply_kmeans == QMessageBox.Yes:
            # Ask for the number of clusters
            k, ok = QInputDialog.getInt(
                self,
                'Number of Clusters',
                'Enter the number of clusters for k-means:',
                min=2,
                max=20,
                step=1
            )
            if not ok:
                return

            # Perform k-means clustering
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(X_reduced)
        else:
            cluster_labels = None

        # Update color options
        color_options = ['Class Labels', 'Feature Map Files']
        if cluster_labels is not None:
            color_options.append('Clusters')

        # Ask the user how to color the data points
        color_option, ok = QInputDialog.getItem(
            self,
            'Color Option',
            'Color data points by:',
            color_options,
            0,
            False
        )
        if not ok:
            return

        if n_components == 2:
            # 2D Plot with interactivity
            fig, ax = plt.subplots(figsize=(10, 6))

            if color_option == 'Class Labels':
                unique_classes = np.unique(combined_labels)
                for class_name in unique_classes:
                    idxs = np.where(combined_labels == class_name)
                    ax.scatter(X_reduced[idxs, 0], X_reduced[idxs, 1],
                            label=class_name, alpha=0.7)
                ax.legend()
            elif color_option == 'Feature Map Files':
                unique_sources = np.unique(feature_sources)
                for source in unique_sources:
                    idxs = [i for i, s in enumerate(feature_sources) if s == source]
                    ax.scatter(X_reduced[idxs, 0], X_reduced[idxs, 1],
                            label=source, alpha=0.7)
                ax.legend()
            elif color_option == 'Clusters' and cluster_labels is not None:
                unique_clusters = np.unique(cluster_labels)
                for cluster in unique_clusters:
                    idxs = np.where(cluster_labels == cluster)
                    ax.scatter(X_reduced[idxs, 0], X_reduced[idxs, 1],
                            label=f'Cluster {cluster}', alpha=0.7)
                ax.legend()
            else:
                QMessageBox.warning(self, 'Invalid Option', 'Invalid color option selected.')
                return

            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_title(title)
            plt.tight_layout()

            # Make sure to draw the canvas before initializing the selector
            fig.canvas.draw_idle()

            # Disconnect any existing lasso selectors
            if hasattr(self, 'selector') and self.selector is not None:
                self.selector.disconnect_events()
                del self.selector

            # Enable selection tool
            self.selector = SelectFromCollection(ax, ax.collections[0], X_reduced, combined_labels,
                                                combined_image_paths, combined_bounding_boxes)

            plt.show()

        elif n_components == 3:
            # 3D Plot
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')

            if color_option == 'Class Labels':
                unique_classes = np.unique(combined_labels)
                for class_name in unique_classes:
                    idxs = np.where(combined_labels == class_name)
                    ax.scatter(X_reduced[idxs, 0], X_reduced[idxs, 1], X_reduced[idxs, 2],
                            label=class_name, alpha=0.7)
                ax.legend()
            elif color_option == 'Feature Map Files':
                unique_sources = np.unique(feature_sources)
                for source in unique_sources:
                    idxs = [i for i, s in enumerate(feature_sources) if s == source]
                    ax.scatter(X_reduced[idxs, 0], X_reduced[idxs, 1], X_reduced[idxs, 2],
                            label=source, alpha=0.7)
                ax.legend()
            elif color_option == 'Clusters' and cluster_labels is not None:
                unique_clusters = np.unique(cluster_labels)
                for cluster in unique_clusters:
                    idxs = np.where(cluster_labels == cluster)
                    ax.scatter(X_reduced[idxs, 0], X_reduced[idxs, 1], X_reduced[idxs, 2],
                            label=f'Cluster {cluster}', alpha=0.7)
                ax.legend()
            else:
                QMessageBox.warning(self, 'Invalid Option', 'Invalid color option selected.')
                return

            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_zlabel('Component 3')
            ax.set_title(title)
            plt.tight_layout()

            plt.show()
        else:
            QMessageBox.warning(self, 'Invalid Dimensions', 'Number of dimensions must be 2 or 3.')
            return


def main():
    app = QApplication(sys.argv)
    window = LabelTool()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()