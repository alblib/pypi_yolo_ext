import unittest
from src.yolo_ext import yolo_model_truncate_classes
from ultralytics import YOLO


class TestYOLOModelTruncateClasses(unittest.TestCase):
    def test_result(self):
        self.model = YOLO('yolo11n.pt')
        self.model = yolo_model_truncate_classes(self.model, 'person')
        self.model.export(format='onnx')
