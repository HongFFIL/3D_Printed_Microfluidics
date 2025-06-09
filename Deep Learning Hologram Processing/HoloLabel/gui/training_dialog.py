# Import necessary modules
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QComboBox, QLineEdit, QPushButton, QFileDialog, QFormLayout, QHBoxLayout

class InitialTrainingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Initial Training Configuration")

        # Create layout
        layout = QFormLayout()

        # Model type selection
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["YOLOv8", "YOLOv10", "YOLOv11"])
        layout.addRow("Model Type:", self.model_type_combo)

        # Model size selection
        self.model_size_combo = QComboBox()
        self.model_size_combo.addItems(["n", "s", "m", "l", "x"])
        layout.addRow("Model Size:", self.model_size_combo)

        # Dataset path selection
        self.dataset_path_input = QLineEdit()
        self.dataset_browse_button = QPushButton("Browse")
        self.dataset_browse_button.clicked.connect(self.browse_dataset)
        dataset_layout = QHBoxLayout()
        dataset_layout.addWidget(self.dataset_path_input)
        dataset_layout.addWidget(self.dataset_browse_button)
        layout.addRow("Dataset Path:", dataset_layout)

        # Training parameters
        self.epochs_input = QLineEdit("100")
        layout.addRow("Epochs:", self.epochs_input)

        self.batch_size_input = QLineEdit("16")
        layout.addRow("Batch Size:", self.batch_size_input)

        self.patience_input = QLineEdit("10")
        layout.addRow("Patience:", self.patience_input)

        self.label_smoothing_input = QLineEdit("0.0")
        layout.addRow("Label Smoothing:", self.label_smoothing_input)

        # Start training button
        self.start_button = QPushButton("Start Training")
        self.start_button.clicked.connect(self.accept)
        layout.addWidget(self.start_button)

        self.setLayout(layout)

    def browse_dataset(self):
        options = QFileDialog.Options()
        dataset_dir = QFileDialog.getExistingDirectory(self, "Select Dataset Directory", options=options)
        if dataset_dir:
            self.dataset_path_input.setText(dataset_dir)

class PredictionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Prediction on Unlabeled Data")

        # Create layout
        layout = QFormLayout()

        # Model path selection
        self.model_path_input = QLineEdit()
        self.model_browse_button = QPushButton("Browse")
        self.model_browse_button.clicked.connect(self.browse_model)
        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_path_input)
        model_layout.addWidget(self.model_browse_button)
        layout.addRow("Model Path:", model_layout)

        # Images folder selection
        self.images_folder_input = QLineEdit()
        self.images_browse_button = QPushButton("Browse")
        self.images_browse_button.clicked.connect(self.browse_images_folder)
        images_layout = QHBoxLayout()
        images_layout.addWidget(self.images_folder_input)
        images_layout.addWidget(self.images_browse_button)
        layout.addRow("Images Folder:", images_layout)

        # Confidence threshold
        self.confidence_input = QLineEdit("0.25")
        layout.addRow("Confidence Threshold:", self.confidence_input)

        self.low_conf_threshold_input = QLineEdit("0.9")
        layout.addRow("Low Confidence Threshold:", self.low_conf_threshold_input)


        self.num_images_input = QLineEdit("100")
        layout.addRow("Number of Images to Process:", self.num_images_input)

        # Start prediction button
        self.start_button = QPushButton("Start Prediction")
        self.start_button.clicked.connect(self.accept)
        layout.addWidget(self.start_button)

        self.setLayout(layout)

    def browse_model(self):
        options = QFileDialog.Options()
        model_file, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "Model Files (*.pt);;All Files (*)", options=options)
        if model_file:
            self.model_path_input.setText(model_file)

    def browse_images_folder(self):
        options = QFileDialog.Options()
        images_dir = QFileDialog.getExistingDirectory(self, "Select Images Directory", options=options)
        if images_dir:
            self.images_folder_input.setText(images_dir)
