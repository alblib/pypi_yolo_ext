import unittest
from src.yolo_ext.slice_class import yolo_model_slice_class
from ultralytics import YOLO


class TestYOLOModelSliceClass(unittest.TestCase):
    def test_result(self):
        self.model = YOLO('yolo11n.pt')
        self.model = yolo_model_slice_class(self.model, 'person')
        self.model.export(format='onnx')
