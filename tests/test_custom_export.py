import unittest
from src.yolo_ext import yolo_model_custom_export_onnx
from ultralytics import YOLO


class TestYOLOModelCustomExportONNX(unittest.TestCase):
    def test_result(self):
        self.model = YOLO('yolo11n.pt')
        yolo_model_custom_export_onnx(
            self.model,
            'test.onnx'
        )

    def test_result2(self):
        self.model = YOLO('yolo11n.pt')
        yolo_model_custom_export_onnx(
            self.model,
            'test.onnx',
            quantize=True
        )

    def test_simplest(self):
        self.model = YOLO('yolo11s.pt')
        yolo_model_custom_export_onnx(
            self.model,
            'simplest.onnx',
            imgsz=320,
            quantize=True,
            truncate_for_classes=['person']
        )
