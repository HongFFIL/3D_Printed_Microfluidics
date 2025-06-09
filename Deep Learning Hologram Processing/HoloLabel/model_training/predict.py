# model_training/predict.py
from ultralytics import YOLO

class ModelPredictor:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.names = self.model.names

    def predict(self, image, conf_threshold=0.5, iou_threshold=0.7, agnostic_nms=False):
        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            agnostic_nms=agnostic_nms,
            verbose=False
        )
        return results

