import unittest
from src.yolo_ext import yolo_model_truncate_classes, imagenet_static_qdq_quantization
from ultralytics import YOLO


class TestYOLOModelStaticQDQQuantization(unittest.TestCase):
    def test_result(self):
        self.model = YOLO('yolo11n.pt')
        self.model.export(format='onnx')
        imagenet_static_qdq_quantization(
            'yolo11n.onnx',
            'yolo11n_q8.onnx',
            640,
            256
        )
