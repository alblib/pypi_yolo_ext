import unittest, os
from src.yolo_ext import yolo_model_truncate_classes
from ultralytics import YOLO


class TestYOLOModelTruncateClasses(unittest.TestCase):
    def test_result(self):
        self.model = YOLO('yolo11n.pt')
        self.trunc_model = yolo_model_truncate_classes(self.model, 'person')

        # Test if export successful
        self.trunc_model.export(format='onnx')

        # Test prediction and compare
        orig_predictions = self.trunc_model.predict(
            [os.path.join('data', name) for name in os.listdir('data')])
        trun_predictions = self.trunc_model.predict(
            [os.path.join('data', name) for name in os.listdir('data')])
        for pred in orig_predictions:
            pred.show()
        for pred in trun_predictions:
            pred.show()